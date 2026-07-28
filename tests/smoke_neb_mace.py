#!/usr/bin/env python3
"""Standalone battery-NEB smoke: LiFePO4 Li migration with a local MACE-MP.

Exercises the REAL production path end-to-end without AiiDA:
  battery_enum.build_supercell -> full relax -> batt_neb.enumerate_hops
  -> batt_neb.hop_endpoints_vacancy -> codes/files/neb.py run_neb_pair
  (the exact function the job_type="neb" calcjob runs) -> percolation.

Physics expectations (GGA literature + experiment):
  - Li migrates along the 1D b channel; in-channel hop (~3.0 A) barrier
    ~0.2-0.55 eV; cross-channel hops are > 1 eV.
  - Percolation must come out ONE-dimensional at the in-channel barrier;
    e_m_2d/3d either huge or unreachable.

Usage:  MP_API_KEY=... python tests/smoke_neb_mace.py [--device cuda]
"""
import argparse
import os
import sys
import time

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, "..", "workchains"))
sys.path.insert(0, os.path.join(_here, "..", "codes", "files"))
import batt_neb                      # noqa: E402
import battery_enum                  # noqa: E402
import neb as neb_engine             # noqa: E402  (codes/files/neb.py)

from pymatgen.io.ase import AseAtomsAdaptor            # noqa: E402
from ase.filters import FrechetCellFilter              # noqa: E402
from ase.optimize import LBFGS                         # noqa: E402
from smoke_battery_mace import mp_structure            # noqa: E402


def relax_full(structure, calc, fmax=0.05, steps=300):
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calc
    LBFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=fmax, steps=steps)
    return AseAtomsAdaptor.get_structure(atoms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="LiFePO4",
                    choices=["LiFePO4", "LiCoO2"],
                    help="LiFePO4 validates 1D percolation, LiCoO2 2D")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--max_atoms", type=int, default=60)
    ap.add_argument("--max_hop", type=float, default=4.75)
    ap.add_argument("--max_hops_run", type=int, default=4)
    ap.add_argument("--n_images", type=int, default=5)
    ap.add_argument("--fmax", type=float, default=0.08)
    ap.add_argument("--max_steps", type=int, default=400)
    args = ap.parse_args()

    from mace.calculators import mace_mp
    calc = mace_mp(model=args.model, device=args.device,
                   default_dtype="float64")

    from smoke_battery_mace import MATERIALS
    host = mp_structure(args.material, MATERIALS[args.material])
    supercell, n_li = battery_enum.build_supercell(host, "Li", args.max_atoms)
    print(f"supercell: {supercell.composition.reduced_formula} "
          f"{len(supercell)} atoms, {n_li} Li sites")
    t0 = time.time()
    relaxed = relax_full(supercell, calc)
    print(f"discharged supercell relaxed [{time.time() - t0:.0f} s]")

    distinct, edges = batt_neb.enumerate_hops(relaxed, "Li",
                                              max_hop=args.max_hop)
    by_dist = sorted(distinct.items(), key=lambda kv: kv[1]["distance"])
    print(f"{len(distinct)} symmetry-distinct hop(s) <= {args.max_hop} A: "
          + ", ".join(f"{k} (d={h['distance']:.2f})" for k, h in by_dist))
    run_list = by_dist[:args.max_hops_run]
    if len(run_list) < len(by_dist):
        print(f"running the {len(run_list)} shortest (--max_hops_run)")

    results = {}
    for key, hop in run_list:
        t0 = time.time()
        initial, final, _ = batt_neb.hop_endpoints_vacancy(relaxed, hop, "Li")
        res = neb_engine.run_neb_pair(
            neb_engine.pmg_to_ase(initial), neb_engine.pmg_to_ase(final),
            calc, n_images=args.n_images, fmax=args.fmax,
            max_steps=args.max_steps, logfile=None)
        results[key] = res
        prof = " ".join(f"{e - res['e_initial']:+.3f}" for e in res["energies"])
        print(f"hop {key} (d={hop['distance']:.2f} A): "
              f"Ea_fwd={res['barrier_fwd']:.3f} eV, "
              f"Ea_rev={res['barrier_rev']:.3f} eV, "
              f"converged={res['converged']} [{time.time() - t0:.0f} s]")
        print(f"    band (eV rel initial): {prof}")

    rows, barriers = batt_neb.hop_summary(dict(run_list), results)
    th = batt_neb.percolation_thresholds(edges, barriers)
    print(f"\npercolation thresholds: 1D {th['e_m_1d']} eV, "
          f"2D {th['e_m_2d']} eV, 3D {th['e_m_3d']} eV")
    expected = {
        "LiFePO4": "1D at the in-channel barrier (~0.2-0.55 eV at GGA), "
                   "2D/3D far higher or unreachable",
        "LiCoO2": "2D at the in-plane hop barrier (1D == 2D, hexagonal Li "
                  "layer), 3D unreachable without an interlayer hop",
    }
    print(f"expected for {args.material}: {expected[args.material]}")


if __name__ == "__main__":
    main()
