"""Generate + MLIP-relax Special Quasirandom Structures for alloy + surface-defect screens.

Entry-point script run as ``aiida.py`` by the SQS CalcJob
(``codes/sqs/calculation.py``). The docs/csp_icet+sqs.{md,pdf} note explains
where this stage sits in the larger pipeline. The stage now does both
structure *generation* (icet SQS, optional slabs, optional surface O
vacancies) and a quick *MLIP relaxation pass* (BFGSLineSearch, fmax 0.05,
max 200 steps). Only structures whose relaxation converged are written to
``output.json``; the relaxed energy is stored on each structure under
``properties["predicted_energy"]`` for the downstream convex-hull step.

Input file (``input_structures.json``)
--------------------------------------

A list of *requests*. Each request describes one parent template plus the
configurational degrees of freedom to sample::

    {
      "parent_label": "Y2Ru2O7-pyrochlore",
      "structure":    <pymatgen Structure.as_dict()>,   # parent unit cell
      "sublattices": {
        # icet labels sublattices 'A','B',... in order of distinct
        # allowed-species sets it encounters. Use the SAME labels here
        # and in composition_grid below.
        "A": {"sites": "Y",  "species": ["Y", "Gd", "Bi"]},
        "B": {"sites": "Ru", "species": ["Ru", "Ir", "Os"]}
      },
      "composition_grid": {
        "A": [{"Y": 1.0},
              {"Y": 0.5, "Gd": 0.5},
              {"Bi": 0.5, "Y": 0.5}],
        "B": [{"Ru": 1.0},
              {"Ru": 0.5, "Ir": 0.5},
              {"Ir": 1.0}]
      },
      "supercell":   [2, 2, 2],   # max search size (volume = product)
      "n_per_comp":  3,           # SQS samples per composition (different seeds)
      "sqs": {
        "cutoffs":  [7.0, 4.0],   # pair, triplet cutoffs in A (icet)
        "n_steps":  5000,         # MC steps per SQS
        "T_start":  5.0,
        "T_stop":   0.001,
        "include_smaller_cells": false
      },
      "surfaces": [               # OPTIONAL: slab variants to build per SQS
        {"miller": [1,1,1], "min_thickness": 10.0, "vacuum": 15.0}
      ],
      "defects": {                # OPTIONAL: surface defects (experimentally
                                  # relevant for OER, HER, CO2RR active sites)
        "oxygen_vacancy": {
          "concentrations": [0.0, 0.125, 0.25],  # fraction of surface O
          "n_per_conc":     2
        }
      }
    }

Output file (``output.json``)
-----------------------------

Only relaxation-converged structures are written. The MLIP relaxed energy
travels on the pymatgen Structure dict under
``properties["predicted_energy"]`` (set from ``atoms.info`` before
serialisation), so the downstream convex-hull step has everything it needs
in one file. Any molecular references shipped alongside the request (e.g.
``o2.vasp`` for the O-vacancy formation energy) are relaxed with the same
MLIP and emitted as per-molecule energies under ``molecular_refs``; the O2
value is also exposed at the top level as ``mu_O2`` for the SQSWorkChain
analysis consumer::

    {
      "structures": [
        {
          "@module": "pymatgen.core.structure",
          "@class":  "Structure",
          "lattice": {...}, "sites": [...],
          "properties": {
            "predicted_energy": -123.456,           # eV, MLIP relaxed
            ...                                     # spacegroup, unit_cell, ...
          }
        }, ...
      ],
      "metadata":   [{
          "parent_label":    "Y2Ru2O7-pyrochlore",
          "kind":            "bulk_sqs",            # or "slab", "slab_Ovac"
          "composition":     {"A": {...}, "B": {...}},
          "n_parent_atoms":  88,
          "natoms":          40,
          "miller":          null,                   # set for slabs
          "slab_thickness":  null,
          "vacuum":          null,
          "vacancy_frac":    0.0,
          "sqs_seed":        17
      }, ...],
      "molecular_refs": {"O2": -9.86, ...},          # eV per molecule
      "mu_O2":          -9.86                        # alias for convenience
    }

Tunable parameters in this file are the experimentally relevant axes:
composition fractions on each cation sublattice, supercell size (target),
surface orientation + thickness + vacuum, and surface oxygen-vacancy
concentration. The technical SQS knobs (cluster cutoffs, MC steps, annealing
temperatures) carry sensible defaults but are exposed for users that need
them.

The downstream consumer is the convex-hull step in ``MainWorkChain``
(workchains/main.py), which uses ``properties["predicted_energy"]`` per
(parent, composition, supercell, surface, defect-set) and assembles the
per-composition convex hull directly.
"""

import os
import sys
import json
import argparse
import random
import numpy as np
from ase.io import read as ase_read
from ase.optimize import BFGSLineSearch
from icet import ClusterSpace
from icet.tools.structure_generation import generate_sqs
from itertools import product
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _pmg_dict_to_ase(struct_dict):
    return AseAtomsAdaptor().get_atoms(Structure.from_dict(struct_dict))


def _ase_to_pmg_dict(atoms):
    return AseAtomsAdaptor().get_structure(atoms).as_dict()


def _build_chemical_symbols(parent_atoms, sublattices):
    """Map parent_atoms positions to icet's per-site allowed-species list.

    Atoms whose symbol matches a sublattice's ``sites`` selector get that
    sublattice's allowed species list; everything else is fixed (single-
    element list). The ``species`` list always contains the parent symbol
    even if the user forgot to list it.
    """
    chem = []
    for atom in parent_atoms:
        assigned = None
        for label, spec in sublattices.items():
            site_sel = spec["sites"]
            site_match = (
                atom.symbol == site_sel
                if isinstance(site_sel, str)
                else atom.symbol in site_sel
            )
            if site_match:
                allowed = list(spec["species"])
                if atom.symbol not in allowed:
                    allowed = [atom.symbol] + allowed
                # icet expects sorted-by-symbol allowed list for stable
                # sublattice identity
                chem.append(sorted(allowed))
                assigned = label
                break
        if assigned is None:
            chem.append([atom.symbol])
    return chem


def _comp_grid_iter(composition_grid):
    """Cartesian product over per-sublattice composition options.

    Yields dicts like ``{'A': {'Y': 0.5, 'Gd': 0.5}, 'B': {'Ru': 1.0}}``.
    Single-sublattice case yields a flat dict.
    """
    labels = sorted(composition_grid.keys())
    options = [composition_grid[k] for k in labels]
    for combo in product(*options):
        if len(labels) == 1:
            yield dict(combo[0])
        else:
            yield {lab: dict(c) for lab, c in zip(labels, combo)}


def _icet_target_concentrations(comp_point, sublattice_species):
    """User comp dict -> the dict shape icet expects.

    icet requires every allowed species on a sublattice to appear in the
    concentrations dict, even at 0.0. We zero-fill against
    ``sublattice_species`` so the user can write ``{"Y": 1.0}`` instead of
    ``{"Y": 1.0, "Gd": 0.0, "Bi": 0.0}``.
    """
    labels = sorted(sublattice_species.keys())
    if len(labels) == 1 and isinstance(next(iter(comp_point.values())), (int, float)):
        # single sublattice: comp_point is {species: frac}
        full = {sp: 0.0 for sp in sublattice_species[labels[0]]}
        full.update(comp_point)
        return full
    # multi sublattice: comp_point is {label: {species: frac}}
    out = {}
    for lab in labels:
        full = {sp: 0.0 for sp in sublattice_species[lab]}
        full.update(comp_point.get(lab, {}))
        out[lab] = full
    return out


def _comp_label(comp_point):
    """Compact human label for filenames / metadata sort keys."""
    if not comp_point:
        return "pure"
    if all(isinstance(v, (int, float)) for v in comp_point.values()):
        return "-".join(f"{k}{v:.2f}" for k, v in sorted(comp_point.items()))
    return "_".join(
        f"{lab}({'-'.join(f'{k}{v:.2f}' for k,v in sorted(d.items()))})"
        for lab, d in sorted(comp_point.items())
    )


# ---------------------------------------------------------------------------
# bulk SQS
# ---------------------------------------------------------------------------

def _is_pure_composition(comp_point):
    """Every sublattice has a single species at fraction 1.0."""
    if not comp_point:
        return True
    if all(isinstance(v, (int, float)) for v in comp_point.values()):
        return any(abs(f - 1.0) < 1e-9 for f in comp_point.values()) and \
               sum(comp_point.values()) <= 1.0 + 1e-9
    return all(
        any(abs(f - 1.0) < 1e-9 for f in subdict.values())
        for subdict in comp_point.values()
    )


def _pure_endpoint_atoms(parent_atoms, sublattices, comp_point, supercell):
    """Build the parent supercell with each sublattice set to its lone species."""
    repeat = tuple(supercell) if isinstance(supercell, (list, tuple)) else \
             (int(round(supercell ** (1/3))), int(round(supercell ** (1/3))),
              int(round(supercell ** (1/3))))
    repeat = tuple(max(1, r) for r in repeat)
    decorated = parent_atoms.copy()
    for lab, spec in sublattices.items():
        site_sel = spec["sites"]
        if isinstance(comp_point.get(lab), dict):
            target_sp = next(sp for sp, f in comp_point[lab].items() if f > 0.999)
        elif lab in comp_point and isinstance(comp_point[lab], (int, float)):
            target_sp = lab  # single-sublattice flat dict; lab IS species
        else:
            target_sp = next(sp for sp, f in comp_point.items() if f > 0.999)
        for i, atom in enumerate(decorated):
            site_match = (
                atom.symbol == site_sel
                if isinstance(site_sel, str)
                else atom.symbol in site_sel
            )
            if site_match:
                decorated[i].symbol = target_sp
    return decorated.repeat(repeat)


def generate_bulk_sqs(parent_atoms, sublattices, comp_point, supercell,
                      sqs_opts, n_per_comp, base_seed):
    """Generate ``n_per_comp`` SQS Atoms for one (parent, composition).

    ``supercell`` is interpreted in units of icet's *primitive cell of the
    parent* (icet derives that internally), not the parent cell the user
    passed in -- so passing ``[2,2,2]`` means "search up to 8 primitive cells",
    not "exactly 2x2x2 of my parent". Use a single int if you only care
    about the total volume budget.
    """
    chem = _build_chemical_symbols(parent_atoms, sublattices)
    cs = ClusterSpace(parent_atoms, cutoffs=sqs_opts["cutoffs"],
                      chemical_symbols=chem)

    if isinstance(supercell, (list, tuple)):
        max_size = int(np.prod(supercell))
    else:
        max_size = int(supercell)
    include_smaller = bool(sqs_opts.get("include_smaller_cells", True))

    if _is_pure_composition(comp_point):
        # icet has nothing to optimise; just decorate the parent supercell.
        return [(_pure_endpoint_atoms(parent_atoms, sublattices,
                                      comp_point, supercell), base_seed)]

    sublattice_species = {}
    for lab, spec in sublattices.items():
        seed = [spec["sites"]] if isinstance(spec["sites"], str) else list(spec["sites"])
        sublattice_species[lab] = sorted(set(list(spec["species"]) + seed))
    target = _icet_target_concentrations(comp_point, sublattice_species)
    out = []
    for k in range(n_per_comp):
        seed = base_seed + k
        try:
            sqs = generate_sqs(
                cluster_space=cs,
                max_size=max_size,
                target_concentrations=target,
                include_smaller_cells=include_smaller,
                T_start=sqs_opts.get("T_start", 5.0),
                T_stop=sqs_opts.get("T_stop", 1e-3),
                n_steps=sqs_opts.get("n_steps", 5000),
                random_seed=seed,
            )
        except Exception as exc:
            sys.stderr.write(
                f"SQS failed for comp {comp_point} seed {seed}: {exc}\n")
            continue
        out.append((sqs, seed))
    return out


# ---------------------------------------------------------------------------
# slabs + surface oxygen vacancies
# ---------------------------------------------------------------------------

def build_slabs(sqs_atoms, surface_specs):
    """For each requested miller plane, build the (most symmetric) slab.

    Returns list of (Atoms, {"miller":..., "thickness":..., "vacuum":...}).
    Termination selection is intentionally minimal here: we pick the slab
    SlabGenerator returns first (typically the highest-symmetry one). The
    downstream relax stage handles termination ranking.
    """
    if not surface_specs:
        return []
    pmg = AseAtomsAdaptor().get_structure(sqs_atoms)
    slabs = []
    for spec in surface_specs:
        miller = tuple(spec["miller"])
        sg = SlabGenerator(
            initial_structure=pmg,
            miller_index=miller,
            min_slab_size=float(spec.get("min_thickness", 10.0)),
            min_vacuum_size=float(spec.get("vacuum", 15.0)),
            center_slab=True,
            in_unit_planes=False,
            primitive=False,
            lll_reduce=True,
        )
        all_slabs = sg.get_slabs(symmetrize=False)
        if not all_slabs:
            continue
        slab = all_slabs[0]
        slabs.append((
            AseAtomsAdaptor().get_atoms(slab),
            {"miller": list(miller),
             "thickness": float(spec.get("min_thickness", 10.0)),
             "vacuum": float(spec.get("vacuum", 15.0))},
        ))
    return slabs


def _surface_oxygen_indices(slab_atoms, depth=2.5):
    """Indices of O atoms within ``depth`` A of either slab face."""
    z = slab_atoms.positions[:, 2]
    z_min, z_max = z.min(), z.max()
    out = []
    for i, atom in enumerate(slab_atoms):
        if atom.symbol != "O":
            continue
        if (z[i] - z_min) < depth or (z_max - z[i]) < depth:
            out.append(i)
    return out


def introduce_surface_vacancies(slab_atoms, frac, n_samples, base_seed):
    """Random O-vacancy configurations on the slab surface(s).

    Random sampling (not SQS) on the vacancy degree of freedom: cheap, good
    enough as the seed set for the relax stage, which will dedup near-
    duplicates via spglib + StructureMatcher.
    """
    surf_idx = _surface_oxygen_indices(slab_atoms)
    if not surf_idx or frac <= 0.0:
        return [(slab_atoms.copy(), 0.0, base_seed)]
    n_vac = int(round(frac * len(surf_idx)))
    if n_vac == 0:
        return [(slab_atoms.copy(), 0.0, base_seed)]
    out = []
    for k in range(n_samples):
        seed = base_seed + k
        rng = random.Random(seed)
        victims = rng.sample(surf_idx, n_vac)
        modified = slab_atoms.copy()
        del modified[sorted(victims, reverse=True)]
        out.append((modified, n_vac / len(surf_idx), seed))
    return out


# ---------------------------------------------------------------------------
# per-request driver
# ---------------------------------------------------------------------------

def process_request(req, request_index):
    """Walk one parent + its composition / surface / defect axes."""
    parent_label = req.get("parent_label", f"parent_{request_index}")
    parent_atoms = _pmg_dict_to_ase(req["structure"])
    sublattices = req["sublattices"]
    composition_grid = req["composition_grid"]
    supercell = req.get("supercell", 2)
    n_per_comp = int(req.get("n_per_comp", 1))
    sqs_opts = dict(req.get("sqs", {}))
    sqs_opts.setdefault("cutoffs", [7.0, 4.0])
    sqs_opts.setdefault("n_steps", 5000)
    surface_specs = req.get("surfaces", []) or []
    defect_spec = (req.get("defects") or {}).get("oxygen_vacancy", None)

    structures = []
    metadata = []

    base_seed = 1000 * request_index + 1

    for comp_idx, comp_point in enumerate(_comp_grid_iter(composition_grid)):
        sqs_set = generate_bulk_sqs(
            parent_atoms=parent_atoms,
            sublattices=sublattices,
            comp_point=comp_point,
            supercell=supercell,
            sqs_opts=sqs_opts,
            n_per_comp=n_per_comp,
            base_seed=base_seed + 100 * comp_idx,
        )

        for sqs_atoms, sqs_seed in sqs_set:
            # bulk
            structures.append(_ase_to_pmg_dict(sqs_atoms))
            metadata.append({
                "parent_label": parent_label,
                "kind": "bulk_sqs",
                "composition": comp_point,
                "n_parent_atoms": len(parent_atoms),
                "natoms": len(sqs_atoms),
                "miller": None,
                "slab_thickness": None,
                "vacuum": None,
                "vacancy_frac": 0.0,
                "sqs_seed": sqs_seed,
            })

            # slabs
            slabs = build_slabs(sqs_atoms, surface_specs)
            for slab_atoms, slab_info in slabs:
                if defect_spec is None:
                    variants = [(slab_atoms.copy(), 0.0, sqs_seed)]
                else:
                    variants = []
                    for conc in defect_spec.get("concentrations", [0.0]):
                        variants.extend(introduce_surface_vacancies(
                            slab_atoms=slab_atoms,
                            frac=float(conc),
                            n_samples=int(defect_spec.get("n_per_conc", 1)),
                            base_seed=sqs_seed * 10,
                        ))

                for v_atoms, v_frac, v_seed in variants:
                    structures.append(_ase_to_pmg_dict(v_atoms))
                    metadata.append({
                        "parent_label": parent_label,
                        "kind": "slab_Ovac" if v_frac > 0 else "slab",
                        "composition": comp_point,
                        "n_parent_atoms": len(parent_atoms),
                        "natoms": len(v_atoms),
                        "miller": slab_info["miller"],
                        "slab_thickness": slab_info["thickness"],
                        "vacuum": slab_info["vacuum"],
                        "vacancy_frac": v_frac,
                        "sqs_seed": v_seed,
                    })

    return structures, metadata

def relax_requests(calculator, structures, metadata):
    res_data = []
    res_structure = []
    for structure, meta in zip(structures, metadata):
        atoms = AseAtomsAdaptor.get_atoms(Structure.from_dict(structure))
        atoms.calc = calculator
        relax = BFGSLineSearch(atoms)
        relax.run(fmax=0.05, steps=200)
        if relax.converged():
            atoms.info['predicted_energy'] = atoms.get_potential_energy()
            res_data.append(meta)
            res_structure.append(AseAtomsAdaptor.get_structure(atoms).as_dict())

    return res_structure, res_data


# ---------------------------------------------------------------------------
# molecular references (computed inline against the same MLIP)
# ---------------------------------------------------------------------------

# Atom count per molecule for every gas-phase reference cell we might ship in
# alongside the SQS request. ase.formula.Formula would derive these, but we
# avoid the extra dependency by listing them explicitly.
_REF_ATOMS_PER_MOLECULE = {"O2": 2, "H2": 2, "N2": 2, "Cl2": 2,
                           "H2O": 3, "CO2": 3, "H2O2": 4}


def relax_molecular_reference(calculator, ref_path, formula,
                              fmax=0.05, max_steps=200):
    """MLIP-relax one molecular reference cell and return per-molecule energy.

    The molecular_references/*.vasp cells pack several copies of a molecule
    (o2.vasp = 4 O2, h2o.vasp = 8 H2O, ...); the total relaxed energy is
    divided by that count so downstream code can use it as the chemical
    potential of a single molecule. Returns None (with a stderr note) if the
    file is missing, ASE refuses to read it, or the relaxation does not
    converge -- callers should treat None as "no reference available" rather
    than abort.
    """
    if not os.path.exists(ref_path):
        sys.stderr.write(f"mol-ref: {ref_path} not shipped, skipping\n")
        return None
    try:
        atoms = ase_read(ref_path)
    except Exception as exc:
        sys.stderr.write(f"mol-ref: cannot read {ref_path}: {exc}\n")
        return None

    atoms.calc = calculator
    relax = BFGSLineSearch(atoms, maxstep=0.1, logfile="opt_molref.log")
    try:
        relax.run(fmax=fmax, steps=max_steps)
    except Exception as exc:
        sys.stderr.write(f"mol-ref: {formula} relax raised: {exc}\n")
        return None
    if not relax.converged():
        sys.stderr.write(
            f"mol-ref: {formula} did not converge in {max_steps} steps\n")
        return None

    atoms_per_mol = _REF_ATOMS_PER_MOLECULE.get(formula, 1)
    n_molecules = max(1, round(len(atoms) / atoms_per_mol))
    return float(atoms.get_potential_energy()) / n_molecules


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # The shared get_cmdline helper unconditionally appends --ML_model /
    # --device (and friends). Accept and ignore them: structure generation
    # has no calculator.
    parser.add_argument("--ML_model", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--input_structures", type=str, default="input_structures.json")

    args = parser.parse_args()

    with open(args.input_structures, "r") as f:
        requests = json.load(f)

    if not isinstance(requests, list):
        raise ValueError(
            "input_structures.json must be a JSON list of request dicts")

    metadata = []
    structures = []
    for i, req in enumerate(requests):
        s, m = process_request(req, i)
        structures.extend(s)
        metadata.extend(m)

    from _calculators import make_calculator
    calc = make_calculator(args.ML_model, model=args.model, model_path=args.model_path, device=args.device, task_name=args.task_name)

    structures, metadata = relax_requests(calc, structures, metadata)

    # Relax any molecular references the CalcJob shipped alongside the
    # request (currently just o2.vasp for the O-vacancy formation energy).
    # Per-molecule values land in output.json under "molecular_refs"; mu_O2
    # is also exposed at the top level for the SQSWorkChain.create_hull
    # consumer.
    molecular_refs = {}
    for formula, filename in [("O2", "o2.vasp")]:
        e_per_mol = relax_molecular_reference(calc, filename, formula)
        if e_per_mol is not None:
            molecular_refs[formula] = e_per_mol

    output = {"structures": structures, "metadata": metadata, "molecular_refs": molecular_refs}
    if "O2" in molecular_refs:
        output["mu_O2"] = molecular_refs["O2"]

    with open("output.json", "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
