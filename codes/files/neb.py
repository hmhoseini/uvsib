"""
Shared (CI-)NEB engine -- job_type "neb" payload for the MLIP runners.

ONE engine for every domain driver (battery ion migration now, catalysis
adsorbate diffusion / reaction steps later): the domain workchains construct
consistently-ordered endpoint pairs; this script only does the band physics.

Input (input_structures.json): a list of PAIR dicts
    {"initial": <pmg as_dict>, "final": <pmg as_dict>,
     "fixed": [atom indices to freeze]           (optional),
     "tag": "<free-form id>"                     (optional)}
The initial and final structures MUST have identical atom ordering (build the
final by editing the initial -- never match indices after the fact).

Per pair:
  1. optional endpoint pre-relax (positions only, fixed cell) -- endpoints
     built by editing a perfect structure are not minima until relaxed,
  2. IDPP interpolation with minimum-image convention (hops may cross the
     periodic boundary),
  3. NEB with FIRE, two-stage: loose plain band first, then climbing image
     to the target fmax,
  4. forward/backward barrier, TS image, convergence flag.

Output (output.json):
    {"results": [{tag, energies, barrier_fwd, barrier_rev, ts_index,
                  converged, error, images (endpoint/TS pmg dicts),
                  e_initial, e_final}],
     "indices": [input positions of pairs that produced a result]}
plus total.txt / failed.txt for the shared parser. A pair that raises is
recorded (error string, converged False) and counted in failed.txt -- the
band set keeps going, the caller decides what a failure means.
"""
import json
import argparse
import traceback

from pymatgen.core import Lattice, Structure
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import FIRE

try:                              # ase >= 3.23
    from ase.mep import NEB
except ImportError:               # older ase
    from ase.neb import NEB


def ase_to_pmg(atoms):
    """Convert an ASE Atoms object to a pymatgen Structure"""
    lattice = Lattice(atoms.cell.array.tolist())
    return Structure(lattice, atoms.get_chemical_symbols(),
                     atoms.get_scaled_positions().tolist(),
                     coords_are_cartesian=False)


def pmg_to_ase(pmg_structure):
    """Convert a pymatgen Structure to an ASE Atoms object"""
    return Atoms(symbols=[str(site.specie) for site in pmg_structure.sites],
                 scaled_positions=pmg_structure.frac_coords,
                 cell=pmg_structure.lattice.matrix, pbc=True)


def run_neb_pair(initial, final, calc, n_images=5, fmax=0.05, max_steps=300,
                 spring=0.1, climb=True, prerelax_endpoints=True,
                 fixed=None, logfile="neb.log"):
    """Run one (CI-)NEB band between two consistently-ordered endpoints.

    Parameters
    ----------
    initial, final : ase.Atoms
        Endpoints with IDENTICAL atom ordering. Modified in place if
        ``prerelax_endpoints``.
    calc : ASE calculator (shared across all images -- fine for MLIPs).
    n_images : int
        INTERIOR images (band size = n_images + 2).
    fmax, max_steps : NEB convergence force / total optimizer step budget.
    spring : NEB spring constant (eV/A^2).
    climb : bool
        Two-stage: plain band to max(3*fmax, 0.10), then climbing image to
        fmax. climb=False runs the plain band to fmax directly.
    prerelax_endpoints : bool
        Position-only relax of both endpoints first (cell always fixed --
        a band between cells makes no sense).
    fixed : list[int] | None
        Atom indices frozen in the endpoints AND every image.

    Returns
    -------
    dict: energies (eV, full band), barrier_fwd, barrier_rev, ts_index,
    converged, e_initial, e_final, images (list of ase.Atoms).
    """
    constraint = FixAtoms(indices=list(fixed)) if fixed else None
    for atoms in (initial, final):
        if constraint is not None:
            atoms.set_constraint(constraint)

    steps_used = 0
    if prerelax_endpoints:
        for atoms in (initial, final):
            atoms.calc = calc
            opt = FIRE(atoms, logfile=logfile)
            opt.run(fmax=fmax, steps=max_steps)
            steps_used += opt.get_number_of_steps()

    images = [initial] + [initial.copy() for _ in range(n_images)] + [final]
    for image in images:
        if constraint is not None:
            image.set_constraint(constraint)
        image.calc = calc

    neb = NEB(images, k=spring, climb=False, allow_shared_calculator=True)
    # mic=True: interpolate along the minimum-image path (hops cross the
    # periodic boundary); IDPP refines the linear guess
    neb.interpolate(method="idpp", mic=True)
    budget = max(max_steps - steps_used, 50)
    if climb:
        pre_fmax = max(3.0 * fmax, 0.10)
        opt = FIRE(neb, logfile=logfile)
        opt.run(fmax=pre_fmax, steps=budget)
        budget = max(budget - opt.get_number_of_steps(), 50)
        neb.climb = True
    opt = FIRE(neb, logfile=logfile)
    opt.run(fmax=fmax, steps=budget)
    converged = bool(opt.converged())

    energies = [float(image.get_potential_energy()) for image in images]
    e_initial, e_final = energies[0], energies[-1]
    ts_index = max(range(len(energies)), key=lambda i: energies[i])
    return {
        "energies": energies,
        "e_initial": e_initial,
        "e_final": e_final,
        "barrier_fwd": energies[ts_index] - e_initial,
        "barrier_rev": energies[ts_index] - e_final,
        "ts_index": ts_index,
        "converged": converged,
        "images": images,
    }


def run_neb_pairs(calc, n_images, fmax, max_steps, spring, climb,
                  prerelax_endpoints):
    """Run every pair in input_structures.json; write the runner outputs."""
    with open("input_structures.json", "r") as f:
        pairs = json.load(f)

    results, indices = [], []
    num_failed = 0
    for i, pair in enumerate(pairs):
        tag = pair.get("tag", f"pair_{i}")
        try:
            initial = pmg_to_ase(Structure.from_dict(pair["initial"]))
            final = pmg_to_ase(Structure.from_dict(pair["final"]))
            res = run_neb_pair(initial, final, calc,
                               n_images=n_images, fmax=fmax,
                               max_steps=max_steps, spring=spring,
                               climb=climb,
                               prerelax_endpoints=prerelax_endpoints,
                               fixed=pair.get("fixed"))
            band = res.pop("images")
            res["images"] = {
                "initial": ase_to_pmg(band[0]).as_dict(),
                "ts": ase_to_pmg(band[res["ts_index"]]).as_dict(),
                "final": ase_to_pmg(band[-1]).as_dict(),
            }
            res["tag"] = tag
            res["error"] = None
            if not res["converged"]:
                num_failed += 1
        except Exception as exc:
            res = {"tag": tag, "energies": None, "barrier_fwd": None,
                   "barrier_rev": None, "ts_index": None, "converged": False,
                   "images": None, "e_initial": None, "e_final": None,
                   "error": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc()
            num_failed += 1
        results.append(res)
        indices.append(i)

    with open("output.json", "w") as f:
        json.dump({"results": results, "indices": indices}, f)
    with open("total.txt", "w") as f:
        f.write(str(len(pairs)))
    with open("failed.txt", "w") as f:
        f.write(str(num_failed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ML_model", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--n_images", type=int, default=5)
    parser.add_argument("--spring", type=float, default=0.1)
    parser.add_argument("--climb", type=int, default=1)
    parser.add_argument("--prerelax_endpoints", type=int, default=1)
    args = parser.parse_args()

    from _calculators import make_calculator
    calc = make_calculator(args.ML_model, model=args.model,
                           model_path=args.model_path, device=args.device,
                           task_name=args.task_name)

    run_neb_pairs(calc, args.n_images, args.fmax, args.max_steps,
                  args.spring, bool(args.climb), bool(args.prerelax_endpoints))
