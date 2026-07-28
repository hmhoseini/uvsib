"""Fine-tune frame harvester for the solvated catalysis NEB chemistry.

Standalone by design (the active-learning loop runs WITHOUT the full uvsib
chain, on a substrate set frozen from a completed run): give it adsorbed
slabs + a model, get back DFT-ready training frames covering exactly the
chemistry the foundation models have not seen -- metal/water interfaces and
*O + H -> *OH transfer paths, including the (extrapolating) TS regions.

Per task (one adsorbed slab, *O located from ads_coord):
  1. pack an explicit water film (codes/files/_solvate.py, seeded),
  2. loose pre-relax (kills packing strain), Langevin NVT equilibration,
  3. sample n_snapshots along the MD           -> kind "md_snapshot",
  4. per snapshot, enumerate (donor H2O, *O) pairs, build endpoint pairs by
     editing ONE parent, freeze everything outside the reactive region and
     run the shared (CI-)NEB engine (codes/files/neb.py),
  5. harvest EVERY band image (converged or not -- non-converged bands are
     still valid training geometries)           -> kinds "neb_endpoint",
                                                   "neb_image".

Input (input_structures.json): either a bare list of tasks or
    {"params": {...overrides...}, "tasks": [...]}
with each task
    {"structure": <pmg Structure.as_dict of the adsorbed slab>,
     "ads_coord": [x, y, z],                  # locates the *O acceptor
     "surface_id": ..., "bulk_uuid": ...,     # attribution, echoed on
     "composition": ..., "miller_index": ..., # every frame this task
     "reaction": ..., "reaction_path": ...,   # produces
     "tag": ...}

Output: output.json
    {"frames": [{"structure": <pmg dict>, "kind": ..., "energy_model": ...,
                 "task": {attribution}, "meta": {...}}, ...],
     "n_frames": ..., "n_tasks": ..., "failed_tasks": [{tag, reason}],
     "params": {...}, "model": ...}
plus frames.extxyz for eyeballing. A task that raises is recorded and the
run continues (fail loudly per task, never silently).

Ingest into the DB with `python -m uvsib.db.ingest_frames output.json
--batch <name>`; export the DFT batch with `export_all.py --finetune-frames`.

`--from-export <export.json>` builds input_structures.json from a
run_dir/export_all.py export instead (needs --with-structures exports).
"""
import argparse
import json
import sys
import traceback

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import write as ase_write
from ase.md.langevin import Langevin
from ase.optimize import FIRE
from ase import units as ase_units
from pymatgen.core import Lattice, Structure

import _solvate
from neb import run_neb_pair


DEFAULT_PARAMS = {
    # water film
    "thickness": 6.0, "gap": 2.3, "n_waters": None, "seed": 7,
    # pre-relax (strain removal only)
    "pre_fmax": 0.6, "pre_steps": 120,
    # MD sampling
    "md_temperature_K": 300.0, "md_timestep_fs": 0.5, "md_friction": 0.02,
    "equil_steps": 1500, "snapshot_stride": 400, "n_snapshots": 3,
    # H-transfer pairs
    "pairs_per_snapshot": 2, "pair_max_dist": 3.5, "oh_bond": 0.98,
    # reactive-region NEB
    "free_radius": 6.0, "slab_free_depth": 4.0,
    "n_images": 5, "neb_fmax": 0.08, "neb_max_steps": 250,
    "spring": 0.1, "climb": True,
    # endpoint-collapse guard: if the transferred H ends up closer than this
    # to its start after endpoint pre-relax, the product state fell back into
    # the reactant basin (*OH not a minimum on this PES) -- re-run the band
    # with the CONSTRUCTED endpoints so the TS region is still sampled
    "collapse_threshold": 0.7,
    # budget
    "max_frames_per_task": 80,
}


def ase_to_pmg_dict(atoms):
    lattice = Lattice(atoms.cell.array.tolist())
    return Structure(lattice, atoms.get_chemical_symbols(),
                     atoms.get_scaled_positions().tolist(),
                     coords_are_cartesian=False).as_dict()


def pmg_to_ase(structure):
    return Atoms(symbols=[str(s.specie) for s in structure.sites],
                 scaled_positions=structure.frac_coords,
                 cell=structure.lattice.matrix, pbc=True)


def _task_attribution(task):
    return {key: task.get(key) for key in
            ("tag", "surface_id", "bulk_uuid", "composition",
             "miller_index", "reaction", "reaction_path")}


def _snapshot_energy(atoms):
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        return None


def run_task(task, calc, params, task_index):
    """One adsorbed slab -> list of frame records (raises on task failure)."""
    structure = Structure.from_dict(task["structure"])
    bare = pmg_to_ase(structure)
    acceptor = _solvate.nearest_index(bare, task["ads_coord"], symbol="O",
                                      max_dist=1.2)

    solvated = _solvate.pack_water(
        bare, thickness=params["thickness"], gap=params["gap"],
        n_waters=params["n_waters"],
        seed=int(params["seed"]) + 1000 * task_index)

    # slab anchor: everything more than slab_free_depth below the slab top
    # stays frozen through pre-relax, MD and every NEB
    z_top = bare.positions[:, 2].max()
    slab_fixed = [i for i in range(len(bare))
                  if bare.positions[i, 2] < z_top - params["slab_free_depth"]]
    n_bare = len(bare)

    solvated.calc = calc
    solvated.set_constraint(FixAtoms(indices=slab_fixed))
    FIRE(solvated, logfile="md.log").run(fmax=params["pre_fmax"],
                                         steps=params["pre_steps"])

    dyn = Langevin(solvated, timestep=params["md_timestep_fs"] * ase_units.fs,
                   temperature_K=params["md_temperature_K"],
                   friction=params["md_friction"], logfile="md.log")
    dyn.run(params["equil_steps"])

    frames = []
    attribution = _task_attribution(task)

    for snap_i in range(params["n_snapshots"]):
        if snap_i:
            dyn.run(params["snapshot_stride"])
        snapshot = solvated.copy()
        snapshot.set_constraint()
        frames.append({
            "structure": ase_to_pmg_dict(snapshot), "kind": "md_snapshot",
            "energy_model": _snapshot_energy(solvated),
            "task": attribution,
            "meta": {"snapshot": snap_i, "n_slab_atoms": n_bare,
                     "acceptor": int(acceptor)},
        })

        pairs = _solvate.find_h_transfer_pairs(
            snapshot, acceptor, max_dist=params["pair_max_dist"],
            k=params["pairs_per_snapshot"])
        if not pairs:
            print(f"task {task_index} snapshot {snap_i}: no donor water "
                  f"within {params['pair_max_dist']} A of the acceptor",
                  flush=True)
        for pair_i, pair in enumerate(pairs):
            initial, final = _solvate.make_h_transfer_endpoints(
                snapshot, pair["h"], acceptor, bond=params["oh_bond"])
            centers = [pair["h"], pair["water_o"], acceptor]
            fixed = sorted(set(
                _solvate.freeze_far_atoms(snapshot, centers,
                                          params["free_radius"]))
                | set(slab_fixed))
            tag = f"t{task_index}_s{snap_i}_p{pair_i}"
            result = run_neb_pair(
                initial.copy(), final.copy(), calc,
                n_images=params["n_images"],
                fmax=params["neb_fmax"], max_steps=params["neb_max_steps"],
                spring=params["spring"], climb=params["climb"],
                prerelax_endpoints=True, fixed=fixed, logfile="neb.log")
            # collapse guard: after endpoint pre-relax the transferred H
            # must actually have moved between the two relaxed endpoints
            h_i = pair["h"]
            h_moved = float(np.linalg.norm(
                result["images"][-1].positions[h_i]
                - result["images"][0].positions[h_i]))
            collapsed = h_moved < params["collapse_threshold"]
            if collapsed:
                # product basin does not exist on this PES here -- rerun the
                # band between the CONSTRUCTED endpoints (no pre-relax): the
                # barriers are then path-scan values, not true barriers, but
                # the TS-region geometries (the training gold) are sampled
                print(f"task {task_index} {tag}: endpoint collapse "
                      f"(H moved {h_moved:.2f} A) -> rerunning with "
                      "constructed endpoints", flush=True)
                result = run_neb_pair(
                    initial.copy(), final.copy(), calc,
                    n_images=params["n_images"],
                    fmax=params["neb_fmax"],
                    max_steps=params["neb_max_steps"],
                    spring=params["spring"], climb=params["climb"],
                    prerelax_endpoints=False, fixed=fixed,
                    logfile="neb.log")
            n_img = len(result["images"])
            for img_i, (image, energy) in enumerate(
                    zip(result["images"], result["energies"])):
                clean = image.copy()
                clean.set_constraint()
                frames.append({
                    "structure": ase_to_pmg_dict(clean),
                    "kind": ("neb_endpoint" if img_i in (0, n_img - 1)
                             else "neb_image"),
                    "energy_model": float(energy),
                    "task": attribution,
                    "meta": {"snapshot": snap_i, "pair": pair_i, "tag": tag,
                             "image_index": img_i,
                             "ts_index": result["ts_index"],
                             "barrier_fwd": result["barrier_fwd"],
                             "barrier_rev": result["barrier_rev"],
                             "converged": result["converged"],
                             "collapsed": collapsed,
                             "d_h_acc": pair["d_h_acc"],
                             "n_fixed": len(fixed)},
                })
            print(f"task {task_index} {tag}: barrier fwd "
                  f"{result['barrier_fwd']:.3f} eV rev "
                  f"{result['barrier_rev']:.3f} eV converged "
                  f"{result['converged']} collapsed {collapsed}", flush=True)

    cap = params["max_frames_per_task"]
    if len(frames) > cap:
        # NEB frames are the gold -- drop md_snapshots first, loudly
        neb_frames = [f for f in frames if f["kind"] != "md_snapshot"]
        md_frames = [f for f in frames if f["kind"] == "md_snapshot"]
        kept = neb_frames[:cap]
        kept += md_frames[:max(cap - len(kept), 0)]
        print(f"task {task_index}: capped {len(frames)} -> {len(kept)} "
              f"frames (max_frames_per_task={cap})", flush=True)
        frames = kept
    return frames


def run_solvation_frames(calc, model_tag):
    with open("input_structures.json", "r") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        params = {**DEFAULT_PARAMS, **(raw.get("params") or {})}
        tasks = raw["tasks"]
    else:
        params, tasks = dict(DEFAULT_PARAMS), raw

    all_frames, failed = [], []
    for i, task in enumerate(tasks):
        tag = task.get("tag", f"task_{i}")
        try:
            all_frames.extend(run_task(task, calc, params, i))
        except Exception as exc:  # noqa: BLE001 -- record + continue per task
            failed.append({"tag": tag,
                           "reason": f"{type(exc).__name__}: {exc}"})
            print(f"task {i} ({tag}) FAILED: {exc}", flush=True)
            traceback.print_exc()

    with open("output.json", "w") as f:
        json.dump({"frames": all_frames, "n_frames": len(all_frames),
                   "n_tasks": len(tasks), "failed_tasks": failed,
                   "params": params, "model": model_tag}, f)

    if all_frames:
        images = []
        for fr in all_frames:
            atoms = pmg_to_ase(Structure.from_dict(fr["structure"]))
            atoms.info.update({"kind": fr["kind"],
                               "tag": fr["meta"].get("tag", ""),
                               "surface_id": fr["task"].get("surface_id")})
            images.append(atoms)
        ase_write("frames.extxyz", images)
    print(f"wrote output.json: {len(all_frames)} frames, "
          f"{len(failed)}/{len(tasks)} tasks failed", flush=True)


# --------------------------------------------------------------- helpers
def make_tasks_from_export(export_path, reaction="OER",
                           reaction_path="default", composition=None,
                           n_surfaces=5):
    """Build the task list from a run_dir/export_all.py export.

    Uses the per-surface BEST (lowest-eta) adsorbate record of the given
    reaction; needs an export made with --with-structures (the adsorbate
    records must embed the relaxed adsorbed-slab structure). Fails loudly
    otherwise -- there is nothing sensible to harvest without geometries.
    """
    with open(export_path) as f:
        data = json.load(f)
    tasks = []
    for comp, v in data.get("compositions", {}).items():
        if composition and comp != composition:
            continue
        best_per_surface = {}
        for a in v.get("adsorbates", []):
            if (a.get("reaction") != reaction
                    or a.get("reaction_path") != reaction_path
                    or a.get("eta") is None):
                continue
            sid = a.get("surface_id")
            if (sid not in best_per_surface
                    or a["eta"] < best_per_surface[sid]["eta"]):
                best_per_surface[sid] = a
        ranked = sorted(best_per_surface.values(), key=lambda a: a["eta"])
        for a in ranked[:n_surfaces]:
            if not a.get("structure"):
                raise ValueError(
                    f"{export_path}: adsorbate record (surface "
                    f"{a.get('surface_id')}, {comp}) has no embedded "
                    "structure -- re-export with --with-structures")
            tasks.append({
                "structure": a["structure"], "ads_coord": a["ads_coord"],
                "surface_id": a.get("surface_id"),
                "bulk_uuid": a.get("bulk_uuid") or a.get("structure_uuid"),
                "composition": comp,
                "miller_index": a.get("miller_index"),
                "reaction": reaction, "reaction_path": reaction_path,
                "tag": f"{comp}_s{a.get('surface_id')}",
            })
    if not tasks:
        raise ValueError(f"{export_path}: no usable adsorbate records for "
                         f"{reaction}/{reaction_path}")
    return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ML_model", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--from-export", type=str, default=None,
                        help="build input_structures.json from an "
                             "export_all.py export (needs --with-structures)")
    parser.add_argument("--reaction", type=str, default="OER")
    parser.add_argument("--reaction_path", type=str, default="default")
    parser.add_argument("--composition", type=str, default=None)
    parser.add_argument("--n_surfaces", type=int, default=5)
    args = parser.parse_args()

    if args.from_export:
        tasks = make_tasks_from_export(
            args.from_export, reaction=args.reaction,
            reaction_path=args.reaction_path, composition=args.composition,
            n_surfaces=args.n_surfaces)
        with open("input_structures.json", "w") as f:
            json.dump(tasks, f)
        print(f"wrote input_structures.json with {len(tasks)} tasks")

    from _calculators import make_calculator
    calc = make_calculator(args.ML_model, model=args.model,
                           model_path=args.model_path, device=args.device,
                           task_name=args.task_name)

    run_solvation_frames(calc, model_tag=args.ML_model or "unknown")
