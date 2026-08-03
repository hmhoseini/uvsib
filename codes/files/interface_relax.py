"""MLIP relaxation of built electrode|electrolyte interfaces.

Stage 3 of the solid-state cell path (docs/interfaces.md). Serves BOTH modes:

  * ``active_learning`` -- the relaxation trajectory is the point. A junction
    built by lattice matching starts with deliberately bad contacts (two
    lattices forced together across a fixed gap), and the walk from there to a
    physical contact spans exactly the configurations an interface MLIP has
    never seen. Frames are subsampled along that walk and written out for DFT
    labelling.
  * ``production`` -- only the converged endpoint matters; it is what the NEB
    stage runs on.

WHY NOT codes/files/relax.py. That runner wraps every structure in a
``FrechetCellFilter``, i.e. it relaxes the CELL. For an interface that is
wrong twice over: the in-plane lattice IS the Zur-McGill coherent match and
must be held fixed, and a slab with vacuum has no meaningful cell relaxation
in the surface normal -- the vacuum would simply be squeezed out. Here the
cell is frozen and only the ions move.

FROZEN BULK. ``active_mask`` from the builder marks the atoms within
ACTIVE_THICKNESS of the junction. The rest are held with FixAtoms: the physics
is at the contact, and relaxing hundreds of bulk atoms both wastes the budget
and adds soft modes that make the later NEB harder to converge. Pass
``relax_all: true`` to override.

SUBSAMPLING IS BY FORCE DECADE, NOT BY STEP. Consecutive optimiser steps
differ by fractions of an Angstrom, so an evenly-strided trajectory is a set of
near-duplicates -- the same correlation trap the training-data README warns
about for MD. Frames are instead selected to spread evenly in log(f_max), which
covers the range from "atoms are clashing" to "converged" with roughly equal
weight per decade.

Input (``input_structures.json``)
    {"params": {"fmax": 0.02, "max_steps": 400, "n_frames": 10,
                "relax_all": false, "mode": "active_learning"},
     "interfaces": [{"uuid": "...", "structure": <Structure.as_dict()>,
                     "active_mask": [bool, ...], "label": "..."}, ...]}

Output (``output.json``)
    {"results": [{"uuid": ..., "label": ..., "converged": bool,
                  "n_steps": int, "fmax_final": float,
                  "energy": float, "structure": <relaxed, as_dict()>,
                  "frames": [{"structure": ..., "fmax": float,
                              "energy": float, "step": int}, ...]}, ...]}
"""

import json
import argparse

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import FIRE
from pymatgen.core import Lattice, Structure

DEFAULT_FMAX = 0.02
DEFAULT_MAX_STEPS = 400
DEFAULT_N_FRAMES = 10


def pmg_to_ase(s):
    return Atoms(symbols=[str(site.specie) for site in s.sites],
                 scaled_positions=s.frac_coords,
                 cell=s.lattice.matrix, pbc=True)


def ase_to_pmg(a):
    return Structure(Lattice(a.cell.array.tolist()),
                     a.get_chemical_symbols(),
                     a.get_scaled_positions().tolist(),
                     coords_are_cartesian=False)


def select_by_force_decade(traj, n_frames):
    """Pick n_frames spread evenly in log10(f_max) over the trajectory.

    Returns indices into ``traj`` (list of (step, fmax, energy, atoms)).
    The first (most distorted) and last (converged) frames are always kept:
    they bracket the range the model has to cover.
    """
    if not traj:
        return []
    if len(traj) <= n_frames:
        return list(range(len(traj)))
    f = np.array([max(t[1], 1e-6) for t in traj])
    logf = np.log10(f)
    targets = np.linspace(logf.max(), logf.min(), n_frames)
    idx = []
    for t in targets:
        j = int(np.argmin(np.abs(logf - t)))
        if j not in idx:
            idx.append(j)
    for must in (0, len(traj) - 1):
        if must not in idx:
            idx.append(must)
    return sorted(idx)


def relax_one(calc, entry, params):
    """Relax one interface at FIXED CELL; return the record for output.json."""
    struct = Structure.from_dict(entry["structure"])
    atoms = pmg_to_ase(struct)
    atoms.calc = calc

    mask = entry.get("active_mask")
    if mask and not params.get("relax_all", False):
        if len(mask) != len(atoms):
            raise ValueError(
                f"active_mask has {len(mask)} entries for {len(atoms)} atoms "
                f"({entry.get('label')}) -- refusing to guess which atoms to "
                f"freeze")
        frozen = [i for i, active in enumerate(mask) if not active]
        if frozen:
            atoms.set_constraint(FixAtoms(indices=frozen))
    else:
        frozen = []

    traj = []

    def record():
        f = atoms.get_forces()
        if frozen:                      # ignore constrained atoms in f_max
            f = np.delete(f, frozen, axis=0)
        fmax = float(np.linalg.norm(f, axis=1).max()) if len(f) else 0.0
        traj.append((len(traj), fmax, float(atoms.get_potential_energy()),
                     atoms.copy()))

    # NO cell filter: the in-plane lattice is the coherent ZSL match.
    opt = FIRE(atoms, logfile="opt.log")
    opt.attach(record, interval=1)
    opt.run(fmax=params.get("fmax", DEFAULT_FMAX),
            steps=params.get("max_steps", DEFAULT_MAX_STEPS))

    if not traj:
        record()
    converged = bool(opt.converged())

    frames = []
    if params.get("mode", "active_learning") == "active_learning":
        for j in select_by_force_decade(traj, params.get("n_frames",
                                                         DEFAULT_N_FRAMES)):
            step, fmax, energy, a = traj[j]
            frames.append({"structure": ase_to_pmg(a).as_dict(),
                           "fmax": fmax, "energy": energy, "step": step})

    return {"uuid": entry.get("uuid"),
            "label": entry.get("label"),
            "converged": converged,
            "n_steps": len(traj),
            "fmax_final": traj[-1][1],
            "energy": traj[-1][2],
            "n_frozen": len(frozen),
            "structure": ase_to_pmg(atoms).as_dict(),
            "frames": frames}


def run_interface_relax(calc, input_path="input_structures.json",
                        output_path="output.json"):
    with open(input_path) as fh:
        req = json.load(fh)
    params = req.get("params", {})
    entries = req.get("interfaces", [])
    if not entries:
        raise ValueError("input_structures.json carries no 'interfaces'")

    results = []
    for entry in entries:
        try:
            res = relax_one(calc, entry, params)
        except Exception as exc:
            # loud, and never silently absent from the output
            res = {"uuid": entry.get("uuid"), "label": entry.get("label"),
                   "converged": False, "error":
                   f"{type(exc).__name__}: {exc}", "frames": []}
            print(f"{entry.get('label')}: FAILED {res['error']}", flush=True)
        else:
            print(f"{res['label']}: {'converged' if res['converged'] else 'NOT converged'} "
                  f"in {res['n_steps']} steps, fmax {res['fmax_final']:.4f}, "
                  f"{len(res['frames'])} frame(s) kept, {res['n_frozen']} frozen",
                  flush=True)
        results.append(res)

    with open(output_path, "w") as fh:
        json.dump({"results": results}, fh)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ML_model", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--task_name", type=str, default=None)
    args = parser.parse_args()

    from _calculators import make_calculator
    calc = make_calculator(args.ML_model, model=args.model,
                           model_path=args.model_path, device=args.device,
                           task_name=args.task_name)
    run_interface_relax(calc)
