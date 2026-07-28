"""Tests for the solvation geometry (codes/files/_solvate.py) and the
frame-harvester contract (codes/files/solvation_frames.py, run with EMT --
mechanics only, the physics smoke lives in smoke_solvation_mace.py).

Run from the repo root:  python -m pytest tests/ -q
"""
import json
import os
import sys

import numpy as np
import pytest
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.geometry import find_mic

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "codes", "files"))
import _solvate  # noqa: E402


def _slab(vacuum=14.0, size=(3, 3, 3)):
    slab = fcc111("Cu", size=size, a=3.61, vacuum=vacuum)
    slab.pbc = [True, True, True]
    return slab


def _mic_min_dist(atoms):
    n = len(atoms)
    pos = atoms.positions
    dmin = np.inf
    for i in range(n - 1):
        v = pos[i + 1:] - pos[i]
        _, d = find_mic(v, np.array(atoms.get_cell()), [True, True, True])
        dmin = min(dmin, d.min())
    return dmin


# ------------------------------------------------------------- pack_water
def test_pack_water_counts_positions_and_integrity():
    slab = _slab()
    n_cu = len(slab)
    z_top = slab.positions[:, 2].max()
    solvated = _solvate.pack_water(slab, thickness=5.0, gap=2.3, seed=1)
    n_w = (len(solvated) - n_cu) // 3
    assert n_w >= 4                       # liquid density in that volume
    assert len(solvated) == n_cu + 3 * n_w
    # film sits above the surface, inside the intended range
    wpos = solvated.positions[n_cu:]
    o_z = wpos[0::3, 2]
    assert o_z.min() >= z_top + 2.3 - 1e-9
    assert o_z.max() <= z_top + 2.3 + 5.0 + 1e-9
    # nothing overlaps
    assert _mic_min_dist(solvated) >= 0.9   # intramolecular OH is ~0.96
    # every packed water is intact and found by the bookkeeping
    units = _solvate.water_units(solvated)
    assert len(units) == n_w
    # slab untouched, order preserved
    assert np.allclose(solvated.positions[:n_cu], slab.positions)


def test_pack_water_deterministic_and_seed_dependent():
    slab = _slab()
    a = _solvate.pack_water(slab, thickness=4.0, seed=3)
    b = _solvate.pack_water(slab, thickness=4.0, seed=3)
    c = _solvate.pack_water(slab, thickness=4.0, seed=4)
    assert np.allclose(a.positions, b.positions)
    assert len(a) == len(c)               # same count, different placement
    assert not np.allclose(a.positions, c.positions)


def test_pack_water_fails_loudly_without_vacuum():
    slab = _slab(vacuum=4.0)
    with pytest.raises(ValueError, match="vacuum"):
        _solvate.pack_water(slab, thickness=6.0)


# ------------------------------------------- pairs, endpoints, freezing
def _scene():
    """Cu slab + *O on the surface + one water 3 A above the adsorbate."""
    from ase import Atoms
    slab = _slab()
    z_top = slab.positions[:, 2].max()
    ads_xy = slab.positions[0, :2] + [1.3, 0.75]
    scene = slab + Atoms("O", positions=[[*ads_xy, z_top + 1.3]])
    acceptor = len(scene) - 1
    o_w = scene.positions[acceptor] + [0.4, 0.2, 2.8]
    h1 = o_w + [0.0, 0.24, -0.93]         # points DOWN toward the acceptor
    h2 = o_w + [0.76, -0.2, 0.55]
    scene += Atoms("OH2", positions=[o_w, h1, h2])
    return scene, acceptor, len(scene) - 3, len(scene) - 2


def test_find_h_transfer_pairs_picks_the_downward_h():
    scene, acceptor, water_o, h_down = _scene()
    pairs = _solvate.find_h_transfer_pairs(scene, acceptor, max_dist=3.5, k=3)
    assert pairs, "no pair found"
    assert pairs[0]["h"] == h_down
    assert pairs[0]["water_o"] == water_o
    assert pairs[0]["acceptor"] == acceptor
    assert pairs[0]["d_h_acc"] < 2.5
    # the adsorbate O itself must never appear as a donor water
    assert all(p["water_o"] != acceptor for p in pairs)


def test_make_h_transfer_endpoints_moves_only_the_h():
    scene, acceptor, water_o, h = _scene()
    initial, final = _solvate.make_h_transfer_endpoints(scene, h, acceptor,
                                                        bond=0.98)
    assert len(initial) == len(final) == len(scene)
    assert initial.get_chemical_symbols() == final.get_chemical_symbols()
    moved = np.linalg.norm(final.positions - initial.positions, axis=1)
    assert moved[h] > 0.5
    others = np.delete(moved, h)
    assert others.max() < 1e-12           # literally nothing else moved
    d = final.get_distance(h, acceptor, mic=True)
    assert abs(d - 0.98) < 1e-8


def test_freeze_far_atoms():
    scene, acceptor, water_o, h = _scene()
    centers = [acceptor, water_o, h]
    fixed = _solvate.freeze_far_atoms(scene, centers, free_radius=4.0)
    assert not set(centers) & set(fixed)
    # far-away bottom-layer slab atoms are frozen
    z_min = scene.positions[:, 2].min()
    bottom = [i for i in range(len(scene))
              if abs(scene.positions[i, 2] - z_min) < 0.1]
    d_acc = [np.linalg.norm(scene.positions[i] - scene.positions[acceptor])
             for i in bottom]
    far_bottom = [i for i, dd in zip(bottom, d_acc) if dd > 4.5]
    assert far_bottom and set(far_bottom) <= set(fixed)


def test_nearest_index_locates_the_adsorbate():
    scene, acceptor, water_o, h = _scene()
    coord = scene.positions[acceptor] + [0.05, -0.03, 0.04]
    assert _solvate.nearest_index(scene, coord, symbol="O",
                                  max_dist=0.5) == acceptor
    with pytest.raises(ValueError, match="A from"):
        _solvate.nearest_index(scene, coord + [0, 0, 30], symbol="O",
                               max_dist=0.5)


# ------------------------------------------------------ runner contract
def test_runner_contract_with_emt(tmp_path, monkeypatch):
    """Tiny end-to-end run: frames written with full attribution, poison
    task recorded as failed without killing the run. EMT energetics are
    meaningless -- this checks the bookkeeping contract only."""
    import solvation_frames as sf
    from ase import Atoms

    slab = _slab(size=(2, 2, 3))
    z_top = slab.positions[:, 2].max()
    ads = slab.positions[0] + [1.3, 0.75, 0]
    ads[2] = z_top + 1.3
    scene = slab + Atoms("O", positions=[ads])
    task = {
        "structure": json.loads(json.dumps(
            sf.ase_to_pmg_dict(scene))),
        "ads_coord": ads.tolist(),
        "surface_id": 4711, "bulk_uuid": "aaaa-bbbb",
        "composition": "Cu", "miller_index": [1, 1, 1],
        "reaction": "OER", "reaction_path": "default", "tag": "good",
    }
    poison = dict(task, structure={"bogus": True}, tag="poison")
    params = {"thickness": 3.2, "gap": 2.3, "seed": 5,
              "pre_fmax": 1.5, "pre_steps": 5,
              "equil_steps": 5, "snapshot_stride": 5, "n_snapshots": 1,
              "pairs_per_snapshot": 1, "pair_max_dist": 6.0,
              "n_images": 3, "neb_fmax": 1.5, "neb_max_steps": 5,
              "free_radius": 5.0, "max_frames_per_task": 50}

    monkeypatch.chdir(tmp_path)
    with open("input_structures.json", "w") as f:
        json.dump({"params": params, "tasks": [task, poison]}, f)

    sf.run_solvation_frames(EMT(), model_tag="EMT")

    out = json.load(open("output.json"))
    assert out["n_tasks"] == 2
    assert [ft["tag"] for ft in out["failed_tasks"]] == ["poison"]
    frames = out["frames"]
    assert frames, "no frames harvested"
    kinds = {f["kind"] for f in frames}
    assert "md_snapshot" in kinds
    assert {"neb_endpoint", "neb_image"} <= kinds   # 3 interior + 2 endpoints
    for fr in frames:
        assert fr["task"]["surface_id"] == 4711
        assert fr["task"]["bulk_uuid"] == "aaaa-bbbb"
        assert fr["structure"]["@class"] == "Structure"
    nebs = [f for f in frames if f["kind"] != "md_snapshot"]
    assert all("barrier_fwd" in f["meta"] for f in nebs)
    assert os.path.exists("frames.extxyz")
