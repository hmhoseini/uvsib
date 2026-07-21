#!/usr/bin/env python3
"""Standalone tier-1 battery smoke: LiFePO4 + LiCoO2 with a local MACE-MP.

End-to-end check of the pure modules WITHOUT AiiDA: MP structure ->
battery_enum (supercell + Ewald-ranked vacancy orderings) -> ASE relax with
MACE (cell + positions, FrechetCellFilter) -> batt.battery_summary.

Sanity expectations (MACE-MP is plain PBE -- TM-oxide voltages sit low):
  LiFePO4 : one flat plateau ~2.8-3.3 V (exp 3.45), Q = 170 mAh/g by
            construction, volume change ~ -5..-8%
  LiCoO2  : V_avg ~3.2-4.0 V, stresses the layer-collapse guard at x=0

Not collected by pytest (no test_ prefix). Needs mace-torch and an
MP_API_KEY env var (structures are cached, so MP is contacted once).

Usage:  MP_API_KEY=... python tests/smoke_battery_mace.py [--device cuda]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "workchains"))
import batt            # noqa: E402
import battery_enum    # noqa: E402

from pymatgen.core import Lattice, Structure               # noqa: E402
from pymatgen.io.ase import AseAtomsAdaptor                # noqa: E402
from ase.filters import FrechetCellFilter                  # noqa: E402
from ase.optimize import LBFGS                             # noqa: E402

CACHE = os.path.expanduser("~/.cache/uvsib_smoke")
MATERIALS = {"LiFePO4": "mp-19017", "LiCoO2": "mp-24850"}


def mp_structure(formula, mp_id):
    """Fetch (once) and cache the MP ground-state structure."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{formula}.json")
    if os.path.isfile(path):
        try:
            with open(path) as fh:
                return Structure.from_dict(json.load(fh))
        except Exception:
            os.remove(path)  # corrupt cache from an interrupted run
    from mp_api.client import MPRester
    with MPRester(os.environ["MP_API_KEY"]) as mpr:
        struct = mpr.get_structure_by_material_id(mp_id)
        if struct is None:  # stale id -> ground state by formula
            docs = mpr.materials.summary.search(
                formula=formula, fields=["material_id", "energy_above_hull"])
            best = min(docs, key=lambda d: d.energy_above_hull)
            print(f"   ({mp_id} gone; using {best.material_id})")
            struct = mpr.get_structure_by_material_id(best.material_id)
    with open(path + ".tmp", "w") as fh:
        json.dump(struct.as_dict(), fh)
    os.replace(path + ".tmp", path)  # atomic: no corrupt cache on interrupt
    return struct


def relax(structure, calc, fmax=0.05, steps=200):
    """Cell+position relax; returns (relaxed Structure, energy eV)."""
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calc
    LBFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=fmax, steps=steps)
    return AseAtomsAdaptor.get_structure(atoms), float(atoms.get_potential_energy())


def li_reference(calc):
    """mu_Li from bcc Li relaxed on the same model."""
    li = Structure(Lattice.cubic(3.44), ["Li", "Li"],
                   [[0, 0, 0], [0.5, 0.5, 0.5]])
    relaxed, energy = relax(li, calc)
    return energy / relaxed.num_sites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--max_atoms", type=int, default=60)
    args = ap.parse_args()

    from mace.calculators import mace_mp
    calc = mace_mp(model=args.model, device=args.device,
                   default_dtype="float64")

    mu_li = li_reference(calc)
    print(f"mu_Li = {mu_li:.4f} eV/atom (bcc, relaxed)\n")

    for formula, mp_id in MATERIALS.items():
        t0 = time.time()
        host = mp_structure(formula, mp_id)
        plan = battery_enum.enumerate_deintercalation(
            host, "Li", n_x_steps=2, max_configs_per_x=3,
            supercell_max_atoms=args.max_atoms)
        n_cfg = sum(len(v) for v in plan["configs"].values())
        print(f"== {formula} ({mp_id}): N={plan['n_sites']} Li sites, grid "
              f"{plan['counts']}, {n_cfg} configs, "
              f"ewald_ranked={plan['ewald_ranked']}")

        configs = []
        for k in plan["counts"]:
            for s in plan["configs"][k]:
                relaxed, energy = relax(s, calc)
                configs.append({"structure": relaxed, "energy": energy})
                print(f"   k={k:3d}  E = {energy:12.4f} eV  "
                      f"({relaxed.num_sites} atoms)")

        res = batt.battery_summary(configs, "Li", mu_li)
        print(f"   -> V_avg = {res['avg_voltage']:.3f} V | steps: "
              + ", ".join(f"{s['voltage']:.3f} V (x {s['x_lo']:.2f}-{s['x_hi']:.2f})"
                          for s in res["voltage_profile"]["steps"]))
        print(f"   -> Q = {res['capacity_grav']:.1f} mAh/g, "
              f"{res['capacity_vol']:.0f} mAh/cm3, "
              f"{res['energy_density']:.0f} Wh/kg, "
              f"dV = {res['volume_change_pct']:+.1f}%")
        print(f"   -> flags: {res['flags']}  [{time.time() - t0:.0f} s]\n")


if __name__ == "__main__":
    main()
