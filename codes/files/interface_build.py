"""Build electrode|electrolyte half-cell interfaces by lattice matching.

Stage 2 of the solid-state cell path. Run ONLY on pairs that survived the
stage-1 thermodynamic filter (`workchains/interface_stability.py`): most
electrode/electrolyte pairs decompose on contact, and a Zur-Mc-Gill supercell
plus termination enumeration plus MLIP relaxation is an expensive way to
describe a phase that never survives assembly. See docs/interfaces.md.

ALGORITHM. Zur and McGill, J. Appl. Phys. 55, 378 (1984): a topological search
for matching superlattices on two surfaces, bounded by area, area ratio, length
and angle tolerance. We do not reimplement it -- pymatgen ships it as
`ZSLGenerator`, with `CoherentInterfaceBuilder` adding termination enumeration.
Ogre (arXiv:2103.13947) extends it with surface/registry matching and is the
reference to consult if termination choice turns out to dominate the results.

HALF-CELLS, NOT A FULL CELL. This builds ONE junction. A single periodic cell
holding both `anode|electrolyte` and `electrolyte|cathode` would need two
simultaneous lattice matches whose common supercell is their least common
multiple, and would be polarised by construction -- a periodic cell cannot
sustain the potential drop between two junctions without compensating charge.
The two halves are computed separately and coupled through mu_ion, exactly as
the CHE catalysis path couples through the electrode potential.

AMORPHOUS ELECTROLYTES ARE OUT OF SCOPE. LiPON and the glassy sulfides have no
lattice to match; they need melt-quench MD and a non-epitaxial contact
construction. The runner refuses an input it cannot treat as crystalline rather
than silently matching against a spurious P1 cell.

Input (``input_structures.json``, staged via the ``file`` namespace)
    {
      "pairs": [
        {"label":            "LiCoO2|LLZO",
         "film":             {"uuid": "...", "structure": <Structure.as_dict()>},
         "substrate":        {"uuid": "...", "structure": <Structure.as_dict()>},
         "film_millers":     [[0,0,1], [1,0,4]],     # optional, default below
         "substrate_millers":[[0,0,1]]},             # optional
        ...
      ]
    }

Output (``output.json``)
    {
      "results": [
        {"label": ..., "film_uuid": ..., "substrate_uuid": ...,
         "n_built": <int>, "n_rejected": <int>,
         "interfaces": [{"structure": <Structure.as_dict()>,
                         "film_miller": [...], "substrate_miller": [...],
                         "termination": [...], "n_atoms": int,
                         "area": float, "strain_percent": float,
                         "active_mask": [bool, ...]}, ...],
         "error": <str or null>},
        ...
      ]
    }

``active_mask`` marks the atoms within ACTIVE_THICKNESS of the junction plane.
Downstream relaxation and NEB freeze the rest: the interesting physics is at
the contact, and relaxing hundreds of bulk atoms wastes the budget while adding
soft modes that make an NEB harder to converge.
"""

import json
import argparse

from monty.json import jsanitize

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder

# --- bounds ----------------------------------------------------------------
# A ZSL match is only useful if the cell is small enough to relax. This cap is
# sized for MLIP relaxation, NOT for DFT: a 96-atom electrolyte cell like LLZO
# produces ~1600-atom interfaces at 4+4 layers, which MACE handles and VASP
# does not. If a pair is ever handed to DFT, drop this by an order of magnitude
# via the per-pair "max_atoms" key rather than editing it here.
MAX_INTERFACE_ATOMS = 2500
MAX_AREA = 400.0            # A^2, ZSLGenerator cap on the matched cell
MAX_AREA_RATIO_TOL = 0.09   # Zur-McGill area-ratio tolerance
MAX_LENGTH_TOL = 0.03
MAX_ANGLE_TOL = 0.01
# Residual strain the film carries after matching. Above this the "coherent"
# interface is a fiction -- a real junction would relieve it with misfit
# dislocations, which a periodic cell of this size cannot represent.
MAX_STRAIN_PERCENT = 5.0
# Terminations grow combinatorially; keep the lowest-index ones per pair.
MAX_TERMINATIONS = 6
MAX_INTERFACES_PER_PAIR = 12
# Default Miller planes when the caller does not specify: low-index only.
DEFAULT_MILLERS = ((0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1))
# Slab thickness in layers, and the vacuum above the film.
FILM_LAYERS = 4
SUBSTRATE_LAYERS = 4
VACUUM = 15.0
GAP = 2.0
# Atoms within this distance of the junction plane relax; the rest are frozen.
ACTIVE_THICKNESS = 6.0      # A, measured either side of the contact

# An input is refused as non-crystalline if it is P1 AND large: that is the
# signature of a melt-quenched or otherwise disordered cell, for which lattice
# matching is meaningless.
AMORPHOUS_MIN_ATOMS = 40
SYMPREC = 0.1


def _is_amorphous(struct):
    """True for a cell that must not be lattice-matched.

    Deliberately conservative: only a LARGE cell with no symmetry beyond P1
    trips it, so an honest low-symmetry crystal still passes.
    """
    if len(struct) < AMORPHOUS_MIN_ATOMS:
        return False
    try:
        sg = SpacegroupAnalyzer(struct, symprec=SYMPREC).get_space_group_number()
    except Exception:
        return True
    return sg == 1


def _strain_percent(match):
    """Residual film strain of a ZSL match, in percent.

    ``ZSLMatch`` exposes the matched super-lattice vectors on both sides; the
    strain is the fractional length mismatch, taken as the larger of the two
    in-plane directions (the worse direction is what governs whether the
    coherent description holds).
    """
    try:
        fv = np.array(match.film_sl_vectors, dtype=float)
        sv = np.array(match.substrate_sl_vectors, dtype=float)
        fl = np.linalg.norm(fv, axis=1)
        sl = np.linalg.norm(sv, axis=1)
        return float(np.max(np.abs(fl - sl) / sl) * 100.0)
    except Exception:
        return float("nan")


def _active_mask(interface):
    """Atoms within ACTIVE_THICKNESS of the junction, as a bool list.

    The junction sits between the topmost substrate atom and the lowest film
    atom; ``Interface`` labels the two sides via film_indices/substrate_indices.
    """
    z = interface.cart_coords[:, 2]
    film_idx = list(interface.film_indices)
    sub_idx = list(interface.substrate_indices)
    if not film_idx or not sub_idx:
        # cannot locate the junction -- relax everything rather than freeze
        # the wrong half
        return [True] * len(interface)
    z_junction = 0.5 * (float(np.min(z[film_idx])) + float(np.max(z[sub_idx])))
    return [bool(abs(zi - z_junction) <= ACTIVE_THICKNESS) for zi in z]


def build_pair(pair):
    """All acceptable interfaces for one (film, substrate) pair."""
    label = pair.get("label", "?")
    film = Structure.from_dict(pair["film"]["structure"])
    substrate = Structure.from_dict(pair["substrate"]["structure"])

    for name, s in (("film", film), ("substrate", substrate)):
        if _is_amorphous(s):
            return {"label": label, "film_uuid": pair["film"].get("uuid"),
                    "substrate_uuid": pair["substrate"].get("uuid"),
                    "n_built": 0, "n_rejected": 0, "interfaces": [],
                    "error": f"{name} is a large P1 cell -- lattice matching "
                             f"cannot treat an amorphous electrolyte; it needs "
                             f"melt-quench MD and a non-epitaxial contact "
                             f"(docs/interfaces.md section 3)"}

    film_millers = [tuple(m) for m in pair.get("film_millers", DEFAULT_MILLERS)]
    sub_millers = [tuple(m) for m in pair.get("substrate_millers", DEFAULT_MILLERS)]
    # Per-pair thickness: a 96-atom electrolyte cell (LLZO) at the default 4
    # layers lands at ~1600 atoms, 4x over the cap. Rather than silently
    # returning nothing, let the caller trade thickness against size.
    n_film = int(pair.get("film_layers", FILM_LAYERS))
    n_sub = int(pair.get("substrate_layers", SUBSTRATE_LAYERS))
    max_atoms = int(pair.get("max_atoms", MAX_INTERFACE_ATOMS))

    zsl = ZSLGenerator(max_area_ratio_tol=MAX_AREA_RATIO_TOL, max_area=MAX_AREA,
                       max_length_tol=MAX_LENGTH_TOL, max_angle_tol=MAX_ANGLE_TOL)

    out, errors = [], []
    # rejections are counted BY REASON: "6 rejected" with no reason is the
    # silent-cap failure mode this module exists to avoid
    rej = {"too_many_atoms": 0, "too_strained": 0}
    sizes = []
    for fm in film_millers:
        for sm in sub_millers:
            try:
                cib = CoherentInterfaceBuilder(
                    substrate_structure=substrate, film_structure=film,
                    film_miller=fm, substrate_miller=sm, zslgen=zsl)
                terminations = list(cib.terminations)[:MAX_TERMINATIONS]
            except Exception as exc:
                # pymatgen #4047: CoherentInterfaceBuilder raises on some
                # inputs. Record it -- an empty interface list must never be
                # mistaken for "this pair simply has no lattice match".
                errors.append(f"{fm}/{sm}: {type(exc).__name__}: {exc}")
                continue

            strain = _strain_percent(cib.zsl_matches[0]) if cib.zsl_matches else float("nan")
            if strain == strain and strain > MAX_STRAIN_PERCENT:
                rej["too_strained"] += 1
                continue

            for term in terminations:
                try:
                    ifaces = list(cib.get_interfaces(
                        termination=term, gap=GAP, vacuum_over_film=VACUUM,
                        film_thickness=n_film,
                        substrate_thickness=n_sub, in_layers=True))
                except Exception as exc:
                    errors.append(f"{fm}/{sm} term {term}: "
                                  f"{type(exc).__name__}: {exc}")
                    continue
                for iface in ifaces:
                    sizes.append(len(iface))
                    if len(iface) > max_atoms:
                        rej["too_many_atoms"] += 1
                        continue
                    out.append({
                        # jsanitize: Interface.as_dict() carries numpy scalars
                        # and arrays that plain json refuses
                        "structure": jsanitize(iface.as_dict(), strict=True),
                        "film_miller": list(fm),
                        "substrate_miller": list(sm),
                        "termination": list(term),
                        "n_atoms": len(iface),
                        "area": float(abs(np.linalg.norm(np.cross(
                            iface.lattice.matrix[0], iface.lattice.matrix[1])))),
                        "strain_percent": strain,
                        "active_mask": _active_mask(iface),
                    })
                    if len(out) >= MAX_INTERFACES_PER_PAIR:
                        break
                if len(out) >= MAX_INTERFACES_PER_PAIR:
                    break
            if len(out) >= MAX_INTERFACES_PER_PAIR:
                break

    # Smallest first: cheapest to relax, and a small coherent cell is a better
    # description than a large one carrying the same strain.
    out.sort(key=lambda d: (d["n_atoms"], d["strain_percent"]))
    return {"label": label,
            "film_uuid": pair["film"].get("uuid"),
            "substrate_uuid": pair["substrate"].get("uuid"),
            "n_built": len(out),
            "n_rejected": sum(rej.values()),
            "rejected_by": rej,
            # atom counts of every candidate, so a pair rejected purely on size
            # says by how much rather than just "rejected"
            "sizes_seen": sorted(sizes),
            "interfaces": out,
            "error": "; ".join(errors) if errors and not out else None}


def run_interface_build(input_path="input_structures.json",
                        output_path="output.json"):
    with open(input_path) as fh:
        req = json.load(fh)
    pairs = req.get("pairs", [])
    if not pairs:
        raise ValueError("input_structures.json carries no 'pairs'")

    results = []
    for pair in pairs:
        res = build_pair(pair)
        results.append(res)
        msg = (f"{res['label']}: {res['n_built']} interface(s), "
               f"{res['n_rejected']} rejected {res['rejected_by']}")
        if res["sizes_seen"]:
            msg += (f"; candidate sizes {res['sizes_seen'][0]}-"
                    f"{res['sizes_seen'][-1]} atoms")
        if res["error"]:
            msg += f" -- {res['error']}"
        print(msg, flush=True)

    with open(output_path, "w") as fh:
        json.dump(jsanitize({"results": results}, strict=True), fh)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # accepted for compatibility with the shared generic-python CalcJob; this
    # runner is pure geometry and needs no calculator
    parser.add_argument("--ML_model", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    args = parser.parse_args()
    run_interface_build()
