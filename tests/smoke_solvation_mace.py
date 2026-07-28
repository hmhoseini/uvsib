#!/usr/bin/env python3
"""Standalone solvation-frames smoke: CuAu(111) + *O + explicit water film
with a local MACE-MP, end-to-end through codes/files/solvation_frames.py.

Validates the whole harvest chain on real physics: water packing survives
MLIP MD at 300 K (film stays molecular, no spontaneous dissociation),
H-transfer pairs are found, the shared NEB engine produces *O + H -> *OH
bands, and the output.json frame records carry full attribution.

Sanity expectations (foundation MACE-MP, PBE-class):
  - waters stay intact through pre-relax + MD (water_units count constant-ish)
  - H-transfer barriers land somewhere physical-ish (0.1 - 1.5 eV); exact
    numbers are NOT trusted -- that is the whole point of the fine-tune.

Not collected by pytest. Usage:
    python tests/smoke_solvation_mace.py [--device cuda] [--model medium]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "codes", "files"))
import _solvate                    # noqa: E402
import solvation_frames as sf      # noqa: E402

import numpy as np                 # noqa: E402
from ase import Atoms              # noqa: E402
from ase.build import fcc111       # noqa: E402


def build_task():
    """CuAu(111) 3x3x4 slab (alternating layers) + *O in an fcc hollow."""
    slab = fcc111("Cu", size=(3, 3, 4), a=3.75, vacuum=13.0)
    symbols = np.array(slab.get_chemical_symbols(), dtype=object)
    z = slab.positions[:, 2]
    layers = np.unique(np.round(z, 3))
    for i, zl in enumerate(layers):
        if i % 2 == 1:
            symbols[np.abs(z - zl) < 1e-2] = "Au"
    slab.set_chemical_symbols(list(symbols))
    slab.pbc = [True, True, True]

    z_top = z.max()
    cell = slab.get_cell()
    hollow = (cell[0] + cell[1]) * (1.0 / 3.0)
    ads = [hollow[0], hollow[1], z_top + 1.25]
    scene = slab + Atoms("O", positions=[ads])
    return {
        "structure": sf.ase_to_pmg_dict(scene),
        "ads_coord": list(map(float, ads)),
        "surface_id": -1, "bulk_uuid": None,
        "composition": "CuAu", "miller_index": [1, 1, 1],
        "reaction": "OER", "reaction_path": "default",
        "tag": "smoke_cuau111",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="medium")
    args = ap.parse_args()

    from mace.calculators import mace_mp
    calc = mace_mp(model=args.model, device=args.device,
                   default_dtype="float64")

    workdir = os.path.expanduser("~/.cache/uvsib_smoke/solvation")
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    params = {
        "thickness": 5.0, "gap": 2.3, "seed": 11,
        "pre_fmax": 0.5, "pre_steps": 60,
        "md_temperature_K": 300.0, "md_timestep_fs": 0.5,
        "md_friction": 0.02, "equil_steps": 400,
        "snapshot_stride": 200, "n_snapshots": 2,
        "pairs_per_snapshot": 1, "pair_max_dist": 4.0,
        "n_images": 5, "neb_fmax": 0.10, "neb_max_steps": 180,
        "free_radius": 5.5, "max_frames_per_task": 60,
    }
    with open("input_structures.json", "w") as f:
        json.dump({"params": params, "tasks": [build_task()]}, f)

    t0 = time.time()
    sf.run_solvation_frames(calc, model_tag=f"MACE-MP-{args.model}")
    dt = time.time() - t0

    out = json.load(open("output.json"))
    print(f"\n=== smoke summary ({dt:.0f} s) ===")
    print(f"tasks: {out['n_tasks']}  failed: {len(out['failed_tasks'])}  "
          f"frames: {out['n_frames']}")
    kinds = {}
    for fr in out["frames"]:
        kinds[fr["kind"]] = kinds.get(fr["kind"], 0) + 1
    print("kinds:", kinds)
    barriers = sorted({round(fr["meta"]["barrier_fwd"], 3)
                       for fr in out["frames"]
                       if fr["kind"] != "md_snapshot"})
    print("forward barriers (eV):", barriers)
    # water integrity through the MD (from the snapshots)
    from pymatgen.core import Structure
    for fr in out["frames"]:
        if fr["kind"] == "md_snapshot":
            atoms = sf.pmg_to_ase(Structure.from_dict(fr["structure"]))
            n_w = len(_solvate.water_units(atoms))
            print(f"snapshot {fr['meta']['snapshot']}: {n_w} intact waters")
    if out["failed_tasks"]:
        raise SystemExit(f"FAILED tasks: {out['failed_tasks']}")
    print("smoke OK; workdir:", workdir)


if __name__ == "__main__":
    main()
