"""Relax one chunk of pre-generated slabs and score their surface energy.

The second half of the old ``face_build.py`` pipeline, split out so the
SurfaceBuilderWorkChain can submit many balanced chunks (<= 50 slabs each) in
parallel instead of relaxing every slab inside a single, unpredictable job.
Runs on the gnome@v100 code via the reused GNoMECalculation + generic
``sqs_parser``.

This runner does NOT pick the best surfaces -- it relaxes its chunk and returns
*every* converged slab with its surface formation energy. The global top-N
(max_num_surf) selection happens in the workchain, after all of a structure's
chunks come back, so the selection is global rather than per-chunk.

Input (``input_structures.json``, staged via the ``file`` namespace) -- one of
    [{"slab": <ase.io.jsonio-encoded orthogonal Slab>,
      "uuid": <bulk structure uuid>,        # attribution: WHICH bulk this
      "epa":  <that bulk's energy/atom>,    # slab came from rides WITH the
      "index": <global input position>},    # slab (chunks mix bulks now)
     ...]
    [<encoded Slab>, ...]                   # legacy single-bulk format;
                                            # --epa from the CLI applies

The per-slab uuid/index are ECHOED into every output record: downstream
stages (adsorbates -> photocat gap filter -> manual HSE) identify slabs by
the bulk uuid they came from, so attribution must never rely on input
position (non-converged slabs are dropped) or on job topology.

CLI: --fmax/--max_steps; --epa only as the legacy-format fallback.

Output (``output.json``, parsed into output_dict by the generic ``sqs_parser``)
    {
      "slabs": [
        {"slab": <pymatgen Slab.as_dict()>, "surface_formation_energy": <eV/A^2>,
         "uuid": <bulk uuid or None>, "index": <input position>},
        ...
      ],
      "n_total":  <slabs in this chunk>,
      "n_failed": <slabs that errored or did not converge>,
      "failed_slabs": [{"uuid", "index", "reason"}, ...]
    }
"""

import json
import argparse
from ase.io import jsonio
from ase.optimize.bfgslinesearch import BFGSLineSearch
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.core.surface import Slab


def ase_to_pmg(atoms):
    """ASE Atoms (with slab metadata in .info) -> pymatgen Slab."""
    structure = AseAtomsAdaptor.get_structure(atoms)
    miller_index = tuple(atoms.info["miller_index"])
    shift = atoms.info.get("shift", 0.0)
    scale_factor = atoms.info.get("scale_factor", None)
    oriented_unit_cell = Structure.from_dict(atoms.info["oriented_unit_cell"])
    energy = atoms.info["energy"]

    return Slab(
        lattice=structure.lattice,
        species=structure.species,
        coords=structure.frac_coords,
        miller_index=miller_index,
        oriented_unit_cell=oriented_unit_cell,
        shift=shift,
        scale_factor=scale_factor,
        to_unit_cell=True,
        site_properties=structure.site_properties,
        energy=energy,
    )


def _normalize_items(raw, cli_epa):
    """Accept both payload formats -> [(enc, uuid, epa, index), ...].

    New format: dicts carrying per-slab uuid/epa/index (chunks mix bulks).
    Legacy format: bare encoded slabs, one bulk per job, epa from the CLI.
    """
    items = []
    for pos, entry in enumerate(raw):
        if isinstance(entry, dict) and "slab" in entry:
            items.append((entry["slab"], entry.get("uuid"),
                          entry.get("epa", cli_epa), entry.get("index", pos)))
        else:
            items.append((entry, None, cli_epa, pos))
    return items


def run_slab_relax(calc, epa, fmax, max_steps):
    """Relax this chunk of slabs; emit every converged slab + surface energy,
    each echoing the uuid/index of the bulk it came from."""
    with open('input_structures.json', 'r') as f:
        raw = json.load(f)
    items = _normalize_items(raw, epa)

    slab_data = []
    failed_slabs = []

    for enc, uuid, slab_epa, index in items:
        try:
            if slab_epa is None:
                raise ValueError("no epa for this slab (neither per-item nor --epa)")
            atoms = jsonio.decode(enc)
            atoms.calc = calc
            relax = BFGSLineSearch(atoms, maxstep=0.1, logfile="log.opt")
            relax.run(fmax=fmax, steps=max_steps)

            if relax.converged():
                n_slab = len(atoms)
                area = atoms.cell.areas()[2]
                energy = atoms.get_potential_energy()
                surface_energy = (energy - (n_slab * slab_epa)) / (2.0 * area)
                atoms.info['energy'] = energy
                atoms.info['surface_formation_energy'] = surface_energy
                slab_data.append({
                    "slab": ase_to_pmg(atoms).as_dict(),
                    "surface_formation_energy": surface_energy,
                    "uuid": uuid,
                    "index": index,
                })
            else:
                failed_slabs.append({"uuid": uuid, "index": index,
                                     "reason": "not converged"})
                print(f"Warning: slab {index} (uuid {uuid}, miller "
                      f"{atoms.info.get('miller_index')}) did not converge")
        except Exception as e:  # noqa: BLE001 -- log + continue per slab
            failed_slabs.append({"uuid": uuid, "index": index,
                                 "reason": f"{type(e).__name__}: {e}"})
            print(f"Error relaxing slab {index} (uuid {uuid}): {e}")

    with open('output.json', 'w') as f:
        json.dump({
            "slabs": slab_data,
            "n_total": len(items),
            "n_failed": len(failed_slabs),
            "failed_slabs": failed_slabs,
        }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ML_model", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--epa", type=float, default=None)  # legacy payloads only
    parser.add_argument("--fmax", type=float)
    parser.add_argument("--max_steps", type=int)
    args = parser.parse_args()

    from _calculators import make_calculator
    calc = make_calculator(args.ML_model, model=args.model, model_path=args.model_path,
                           device=args.device, task_name=args.task_name)

    run_slab_relax(calc, args.epa, args.fmax, args.max_steps)
