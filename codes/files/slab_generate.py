"""Generate (and orthogonality-filter) all symmetric slabs for several bulks.

Split out of ``face_build.py``: the ``generate_all_slabs`` enumeration has an
unpredictable runtime (it scales with the number of Miller planes and symmetry
operations), so it is run as its own CalcJob *before* the relaxations. ALL bulk
structures for a formula are handled by a SINGLE job (the model is loaded once
and reused), rather than one generation job per bulk. This runner does NOT relax
anything -- per bulk it only computes the bulk per-atom energy (``epa``, needed
downstream for the surface energy) and emits the orthogonal slabs for the
SurfaceBuilderWorkChain to chunk and submit for relaxation.

Robustness (added 2026-06-15) -- why even FCC metals used to OOM / hit walltime
-----------------------------------------------------------------------------
The bulks reaching this runner are MLIP-relaxed, so they carry ~0.001-0.01 A of
numerical noise. At the tight default ``symprec=0.01`` spglib then mis-reads a
perfect FCC metal (Ag, Cu, ...) as a large ``P1`` cell, and ``generate_all_slabs``
explodes: in P1 every atomic layer is a distinct termination, so it enumerates
dozens of Miller planes x many spurious shifts and (with ``max_normal_search``)
builds large oriented supercells -- holding every slab of all bulks in RAM in
one shared CalcJob. Measured on Ag, ``max_index=2``: a clean Fm-3m cell -> 6
slabs in ~3 s / ~100 MB, but the same metal as a noisy 32-atom P1 cell times out
(>180 s) and OOMs. Three guards fix this without changing the slab metadata
contract (still pymatgen ``Slab`` -> ``pmg_to_ase`` with the oriented cell):

  1. ``_standardize_bulk`` walks a loose->tight symprec ladder and keeps the
     SMALLEST conventional cell that preserves the reduced composition, so a
     noisy metal collapses back to its 4-atom Fm-3m cell (symprec 0.05 suffices).
  2. ``max_normal_search=1`` (was ``=max_miller_idx``): orthogonality is already
     enforced afterwards by ``process_slab``, so a costly normal search is waste.
  3. A size+symmetry skip (``MAX_CONV_ATOMS`` & ``LOWSYM_SG_MAX``) and a per-bulk
     ``GEN_TIMEOUT_S`` walltime cap turn a pathological cell into a clean per-bulk
     skip instead of taking down the whole 10-bulk job. The guard skips ONLY
     large *and* low-symmetry cells, so legitimate high-symmetry oxides (e.g. a
     Fd-3m pyrochlore, 88-atom conventional cell) still pass. Bulks marked
     ``from_manifest`` (deliberately produced via the ``DBComposition.stable_struct``
     manifest -- SQS cells are P1 by construction and can exceed the atom cap)
     BYPASS this skip; the walltime cap remains their safety net.

See ``docs/slab_generation.md`` for the full diagnosis and benchmark.

Input (``input_structures.json``, staged via the ``file`` namespace)
    [{"uuid": "<structure uuid>", "structure": <pymatgen Structure.as_dict()>,
      "from_manifest": <bool, optional: bypass the size+symmetry skip>},
     ...]

Submitted on the gnome@v100 code (the generic v100 Python code) via the reused
GNoMECalculation + generic ``sqs_parser`` -- only ``output.json`` is retrieved.

Output (``output.json``, parsed into output_dict by the generic ``sqs_parser``)
    {
      "results": [
        {
          "uuid":    "<structure uuid>",
          "epa":     <bulk energy per atom, eV>,
          "n_total": <slabs generated before filtering>,
          "n_orth":  <orthogonal slabs kept>,
          "slabs":   [<ase.io.jsonio-encoded orthogonal Slab>, ...]  # info
                                                                     # carries
                                                                     # slab meta
        },
        ...
      ]
    }

Each slab is serialised with ``ase.io.jsonio`` (the same encoding adsorbates.py
uses) so the ``info`` dict -- miller_index, shift, scale_factor and the
oriented_unit_cell -- survives the trip to ``slab_relax.py``.
"""

import json
import signal
import argparse
from contextlib import contextmanager
import numpy as np
from ase.io import jsonio
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import Slab, generate_all_slabs
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# --- slab-generation safety bounds (see module docstring "Robustness") -------
# Loose->tight symprec ladder used to absorb MLIP relaxation noise so a noisy
# near-symmetric cell collapses to its small high-symmetry conventional cell.
SYMPREC_LADDER = (0.1, 0.05, 0.01)
# Skip slab generation for a standardized cell that is BOTH large AND
# low-symmetry: that is the OOM/hang regime, and such a cell is not a
# meaningful slab target. High-symmetry oxides (large but ordered) are kept.
MAX_CONV_ATOMS = 60          # atom-count threshold for the skip
LOWSYM_SG_MAX = 15           # spacegroup numbers 1-15 = triclinic + monoclinic
# Per-bulk walltime cap on generate_all_slabs; a slower cell is skipped, not
# allowed to take down the shared multi-bulk job.
GEN_TIMEOUT_S = 3600 * 2 # timeout for generate_all_slabs call


class SlabGenTimeout(Exception):
    """Raised when generate_all_slabs exceeds GEN_TIMEOUT_S for one bulk."""


@contextmanager
def _time_limit(seconds):
    """SIGALRM-based walltime guard (runner is single-threaded / main thread)."""
    def _handler(signum, frame):
        raise SlabGenTimeout(f"slab generation exceeded {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _standardize_bulk(structure):
    """Conventional standard cell, absorbing MLIP relaxation noise.

    Walk SYMPREC_LADDER (loose -> tight) and return the SMALLEST conventional
    cell whose reduced composition still matches the input -- so a noisy FCC
    metal collapses to its 4-atom Fm-3m cell, but we never over-merge into a
    different stoichiometry. Falls back to the tight symprec=0.01 conventional
    cell (the original behaviour) if no ladder rung preserves the composition.
    """
    target = structure.composition.reduced_formula
    best = None
    for symprec in SYMPREC_LADDER:
        try:
            conv = SpacegroupAnalyzer(
                structure, symprec=symprec, angle_tolerance=5
            ).get_conventional_standard_structure()
        except Exception:  # noqa: BLE001 -- bad symprec rung, try the next
            continue
        if conv.composition.reduced_formula != target:
            continue
        if best is None or conv.num_sites < best.num_sites:
            best = conv
    if best is None:
        best = SpacegroupAnalyzer(
            structure, symprec=0.01, angle_tolerance=5
        ).get_conventional_standard_structure()
    return best


def pmg_to_ase(slab):
    """pymatgen Slab -> ASE Atoms, stashing slab metadata in Atoms.info."""
    ase_atoms = AseAtomsAdaptor.get_atoms(slab)
    ase_atoms.info.update({
        "miller_index": tuple(slab.miller_index),
        "shift": slab.shift,
        "scale_factor": slab.scale_factor,
        "oriented_unit_cell": slab.oriented_unit_cell.as_dict(),
    })
    return ase_atoms


def process_slab(slab, target_vacuum=10.0, angle_tol=1.0):
    """Keep only (near-)orthogonal slabs; snap alpha/beta to 90 deg and set the
    vacuum along c. Returns the processed pymatgen Slab, or None if skewed."""
    a_len, b_len, c_len = slab.lattice.abc
    alpha, beta, gamma = slab.lattice.angles  # degrees

    if abs(alpha - 90) > angle_tol or abs(beta - 90) > angle_tol:
        return None

    new_lattice = Lattice.from_parameters(a=a_len, b=b_len, c=c_len,
                                          alpha=90, beta=90, gamma=gamma)

    cart_coords = np.array(slab.cart_coords)
    z_coords = cart_coords[:, 2]
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    slab_thickness = z_max - z_min

    new_c = slab_thickness + target_vacuum
    scale_factor = new_c / c_len
    new_matrix = new_lattice.matrix.copy()
    new_matrix[2] *= scale_factor
    new_lattice = Lattice(new_matrix)

    return Slab(
        lattice=new_lattice,
        species=[site.specie for site in slab],
        coords=[site.coords for site in slab],
        miller_index=slab.miller_index,
        oriented_unit_cell=slab.oriented_unit_cell,
        shift=slab.shift,
        scale_factor=slab.scale_factor,
        coords_are_cartesian=True,
        to_unit_cell=True,
        site_properties=slab.site_properties,
    )


def generate_for_structure(calc, structure, max_miller_idx, from_manifest=False):
    """epa + orthogonal slabs for one bulk; returns the per-structure result.

    ``from_manifest=True`` marks a bulk the producing workchain deliberately
    requested surfaces for (``DBComposition.stable_struct`` manifest, e.g. an
    SQS cell -- P1 by construction, and above MAX_CONV_ATOMS for larger
    supercells). The size+symmetry skip below exists to catch ACCIDENTAL
    large-P1 cells from relaxation noise, so it is bypassed for manifest
    bulks; the GEN_TIMEOUT_S walltime cap still applies to them.
    """
    nat = structure.num_sites

    # Bulk per-atom energy (epa), passed to every relax chunk of this structure.
    atoms_bulk = AseAtomsAdaptor.get_atoms(structure)
    atoms_bulk.calc = calc
    epa = atoms_bulk.get_potential_energy() / nat

    # Standardize first so MLIP relaxation noise does not leave the cell as a
    # large P1 cell (which makes generate_all_slabs explode -- see docstring).
    conv_struct = _standardize_bulk(structure)
    sg_number = SpacegroupAnalyzer(conv_struct, symprec=0.1, angle_tolerance=5).get_space_group_number()

    # Skip ONLY large + low-symmetry cells: that is the OOM/hang regime and not
    # a meaningful slab target. Ordered oxides (large but high-symmetry) pass.
    # Manifest bulks (deliberate, e.g. SQS) bypass the skip -- the walltime
    # guard around generate_all_slabs remains their safety net.
    if conv_struct.num_sites > MAX_CONV_ATOMS and sg_number <= LOWSYM_SG_MAX:
        if not from_manifest:
            print(f"Skipping slab generation: standardized cell has "
                  f"{conv_struct.num_sites} atoms at spacegroup #{sg_number} "
                  f"(low-symmetry & > MAX_CONV_ATOMS={MAX_CONV_ATOMS}); epa: {epa}")
            return {'epa': epa, 'n_total': 0, 'n_orth': 0, 'slabs': []}
        print(f"Manifest bypass: standardized cell has {conv_struct.num_sites} "
              f"atoms at spacegroup #{sg_number} (would be skipped as "
              f"low-symmetry & > MAX_CONV_ATOMS={MAX_CONV_ATOMS}); generating "
              f"anyway under the {GEN_TIMEOUT_S}s walltime guard")

    # max_normal_search=1 (cheap): orthogonality is enforced afterwards by
    # process_slab, so a costly search (=max_miller_idx) is wasted work. The
    # walltime guard converts a pathological cell into a clean per-bulk skip.
    with _time_limit(GEN_TIMEOUT_S):
        slabs = generate_all_slabs(conv_struct,
                                   max_index=max_miller_idx,
                                   min_slab_size=4,
                                   min_vacuum_size=10,
                                   max_normal_search=1,
                                   in_unit_planes=True)

    # Only keep orthogonal slabs (relaxation happens later, in chunks).
    orth_slabs = [process_slab(s) for s in slabs]
    orth_slabs = [s for s in orth_slabs if s is not None]

    print(f"Standardized cell: {conv_struct.num_sites} atoms (spacegroup "
          f"#{sg_number}); total slabs generated: {len(slabs)}; "
          f"orthogonal after filtering: {len(orth_slabs)}; epa: {epa}")

    return {
        'epa': epa,
        'n_total': len(slabs),
        'n_orth': len(orth_slabs),
        'slabs': [jsonio.encode(pmg_to_ase(s)) for s in orth_slabs],
    }


def run_slab_generate(calc, max_miller_idx):
    """Generate slabs for every bulk in input_structures.json in one job."""
    with open('input_structures.json', 'r') as f:
        entries = json.load(f)

    results = []
    for entry in entries:
        uuid_str = entry["uuid"]
        structure = Structure.from_dict(entry["structure"])
        print(f"--- generating slabs for uuid={uuid_str} ---")
        try:
            res = generate_for_structure(calc, structure, max_miller_idx,
                                         from_manifest=bool(entry.get("from_manifest")))
        except Exception as e:  # noqa: BLE001 -- log + continue per structure
            print(f"Error generating slabs for uuid={uuid_str}: {e}")
            res = {'epa': None, 'n_total': 0, 'n_orth': 0, 'slabs': []}
        res['uuid'] = uuid_str
        results.append(res)

    with open('output.json', 'w') as f:
        json.dump({'results': results}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ML_model", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--max_miller_idx", type=int)
    args = parser.parse_args()

    from _calculators import make_calculator
    calc = make_calculator(args.ML_model, model=args.model, model_path=args.model_path,
                           device=args.device, task_name=args.task_name)

    run_slab_generate(calc, args.max_miller_idx)
