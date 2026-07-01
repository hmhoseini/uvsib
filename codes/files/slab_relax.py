"""Relax one chunk of pre-generated slabs and score their surface energy.

The second half of the old ``face_build.py`` pipeline, split out so the
SurfaceBuilderWorkChain can submit many balanced chunks (<= 250 slabs each) in
parallel instead of relaxing every slab inside a single, unpredictable job.
Runs on the gnome@v100 code via the reused GNoMECalculation + generic
``sqs_parser``.

This runner does NOT pick the best surfaces -- it relaxes its chunk and returns
*every* converged slab with its surface formation energy. The global top-N
(max_num_surf) selection happens in the workchain, after all of a structure's
chunks come back, so the selection is global rather than per-chunk.

Input (``input_structures.json``, staged via the ``file`` namespace)
    [<ase.io.jsonio-encoded orthogonal Slab>, ...]   # this chunk's slabs;
                                                     # info carries slab metadata
CLI: --epa (bulk energy per atom from slab_generate) + --fmax/--max_steps.

Output (``output.json``, parsed into output_dict by the generic ``sqs_parser``)
    {
      "slabs": [
        {"slab": <pymatgen Slab.as_dict()>, "surface_formation_energy": <eV/A^2>},
        ...
      ],
      "n_total":  <slabs in this chunk>,
      "n_failed": <slabs that errored or did not converge>
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


def run_slab_relax(calc, epa, fmax, max_steps):
    """Relax this chunk of slabs; emit every converged slab + surface energy."""
    with open('input_structures.json', 'r') as f:
        encoded_slabs = json.load(f)

    slab_data = []
    num_failed = 0

    for idx, enc in enumerate(encoded_slabs):
        try:
            atoms = jsonio.decode(enc)
            atoms.calc = calc
            relax = BFGSLineSearch(atoms, maxstep=0.1, logfile="log.opt")
            relax.run(fmax=fmax, steps=max_steps)

            if relax.converged():
                n_slab = len(atoms)
                area = atoms.cell.areas()[2]
                energy = atoms.get_potential_energy()
                surface_energy = (energy - (n_slab * epa)) / (2.0 * area)
                atoms.info['energy'] = energy
                atoms.info['surface_formation_energy'] = surface_energy
                slab_data.append({
                    "slab": ase_to_pmg(atoms).as_dict(),
                    "surface_formation_energy": surface_energy,
                })
            else:
                num_failed += 1
                print(f"Warning: slab {idx} "
                      f"(miller {atoms.info.get('miller_index')}) did not converge")
        except Exception as e:  # noqa: BLE001 -- log + continue per slab
            num_failed += 1
            print(f"Error relaxing slab {idx}: {e}")

    with open('output.json', 'w') as f:
        json.dump({
            "slabs": slab_data,
            "n_total": len(encoded_slabs),
            "n_failed": num_failed,
        }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ML_model", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--epa", type=float)
    parser.add_argument("--fmax", type=float)
    parser.add_argument("--max_steps", type=int)
    args = parser.parse_args()

    from _calculators import make_calculator
    calc = make_calculator(args.ML_model, model=args.model, model_path=args.model_path,
                           device=args.device, task_name=args.task_name)

    run_slab_relax(calc, args.epa, args.fmax, args.max_steps)
