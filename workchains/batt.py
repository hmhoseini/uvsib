"""
Battery (deintercalation) characteristics calculator.

Pure-python analogue of the reaction calculators (oer.py, co2rr.py, ...):
no AiiDA imports, unit-testable standalone. The BatteryWorkChain feeds it the
relaxed configurations A_n·Host (n = working ions left in a COMMON host
supercell) plus the elemental reference energy of the ion metal, and gets back
everything the DBBatteryPath row stores.

Conventions
-----------
- All energies are total energies of the SAME host supercell (eV); n is the
  integer number of working ions in that supercell. n = 0 is the charged
  (empty) host, n = N the discharged end member.
- Voltage between two hull vertices n1 < n2 (vs the A/A+ metal anode):

      V = -[E(n2) - E(n1) - (n2 - n1)·mu_A] / (z·(n2 - n1))    [V]

  with mu_A the energy per atom of elemental A on the same method and
  z = ION_Z[A] electrons per ion.
- Gravimetric capacity follows the standard cathode convention (per mass of
  the DISCHARGED formula unit): LiFePO4 -> ~170 mAh/g.
"""
from __future__ import annotations

import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

# electrons transferred per working ion
ION_Z = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1,
         "Mg": 2, "Ca": 2, "Zn": 2, "Al": 3}

# Faraday constant expressed as mAh per mol of electrons: F / 3.6
MAH_PER_MOL_E = 96485.33212 / 3.6  # = 26801.48 mAh/mol

# 1 A^3 per supercell -> cm^3 per mol of supercells: 1e-24 * N_A
A3_TO_CM3_PER_MOL = 0.602214076


def _as_structure(struct):
    """Accept a pymatgen Structure or its as_dict() form."""
    if isinstance(struct, Structure):
        return struct
    return Structure.from_dict(struct)


def n_ion(structure, working_ion):
    """Number of working-ion atoms in a structure."""
    comp = structure.composition.get_el_amt_dict()
    return int(round(comp.get(working_ion, 0.0)))


def hull_and_voltages(points, mu_ion, z):
    """Pseudo-binary hull over the ion content and the voltage profile.

    Parameters
    ----------
    points : list[(int, float)]
        (n_ion, total energy) per relaxed configuration, common supercell.
        Multiple configurations per n are fine -- the minimum is used.
    mu_ion : float
        Energy per atom of the elemental working-ion metal (same method).
    z : int
        Electrons per working ion.

    Returns
    -------
    dict with keys:
        vertices  : [{n, x, energy}] lower-hull vertices (x = n/N)
        steps     : [{x_lo, x_hi, voltage}] one entry per hull segment,
                    ascending in x (voltage is non-increasing by convexity)
        avg_voltage : float, full-range average (end members only)
        points    : [{n, x, energy, e_above_tieline}] per-n minima with their
                    height above the hull (eV per supercell)

    Raises
    ------
    ValueError
        If the charged (n=0) or discharged (n=N=max) end member is missing,
        or fewer than two distinct n are present.
    """
    emin = {}
    for n, energy in points:
        n = int(n)
        if n not in emin or energy < emin[n]:
            emin[n] = float(energy)

    if len(emin) < 2:
        raise ValueError("need at least the two end members, got "
                         f"n = {sorted(emin)}")
    n_max = max(emin)
    if 0 not in emin:
        raise ValueError("charged end member (n = 0) missing")

    ns = sorted(emin)
    # lower convex hull in (n, E) via monotone chain (2D, already x-sorted)
    hull = []
    for n in ns:
        p = (n, emin[n])
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            # pop the middle point while it lies on/above the new segment
            # (Andrew monotone chain, lower hull: pop on cross <= 0)
            if (x2 - x1) * (p[1] - y1) - (p[0] - x1) * (y2 - y1) <= 0:
                hull.pop()
            else:
                break
        hull.append(p)

    vertices = [{"n": int(n), "x": n / n_max, "energy": e} for n, e in hull]

    steps = []
    for (n1, e1), (n2, e2) in zip(hull[:-1], hull[1:]):
        voltage = -((e2 - e1) - (n2 - n1) * mu_ion) / (z * (n2 - n1))
        steps.append({"x_lo": n1 / n_max, "x_hi": n2 / n_max,
                      "voltage": voltage})

    e0, eN = emin[0], emin[n_max]
    avg_voltage = -((eN - e0) - n_max * mu_ion) / (z * n_max)

    def _hull_at(n):
        for (n1, e1), (n2, e2) in zip(hull[:-1], hull[1:]):
            if n1 <= n <= n2:
                t = (n - n1) / (n2 - n1)
                return e1 + t * (e2 - e1)
        return emin[n]  # n outside hull span cannot happen (ends are vertices)

    pts = [{"n": int(n), "x": n / n_max, "energy": emin[n],
            "e_above_tieline": emin[n] - _hull_at(n)} for n in ns]

    return {"vertices": vertices, "steps": steps,
            "avg_voltage": avg_voltage, "points": pts}


def capacities(discharged_structure, working_ion, z=None):
    """Gravimetric and volumetric capacity from the discharged structure.

    Q_grav = n·z·F / (3.6·M)  [mAh/g], M = molar mass of the (super)cell;
    Q_vol uses the discharged-state volume. Both are theoretical capacities
    for extraction of ALL working ions.
    """
    struct = _as_structure(discharged_structure)
    z = ION_Z[working_ion] if z is None else z
    n = n_ion(struct, working_ion)
    if n == 0:
        raise ValueError(f"no {working_ion} in the discharged structure")
    mass = float(struct.composition.weight)          # g/mol per supercell
    q_grav = n * z * MAH_PER_MOL_E / mass            # mAh/g
    vol_cm3_mol = struct.volume * A3_TO_CM3_PER_MOL  # cm^3/mol per supercell
    q_vol = n * z * MAH_PER_MOL_E / vol_cm3_mol      # mAh/cm^3
    return q_grav, q_vol


def volume_change_pct(discharged_structure, charged_structure):
    """(V_charged - V_discharged) / V_discharged in percent."""
    v_d = _as_structure(discharged_structure).volume
    v_c = _as_structure(charged_structure).volume
    return (v_c - v_d) / v_d * 100.0


def framework_match(discharged_structure, charged_structure, working_ion,
                    ltol=0.2, stol=0.3, angle_tol=5.0):
    """Does the empty host still match the discharged host framework?

    Strips the working ion from the discharged structure and compares with the
    charged one via StructureMatcher (primitive, volume-scaled -- so a pure
    volume change does NOT count as a mismatch; that is reported separately).
    A False here means the framework reconstructed during delithiation (layer
    gliding, collapse, amorphization onset) and the intercalation-voltage
    picture is not trustworthy.
    """
    host = _as_structure(discharged_structure).copy()
    host.remove_species([working_ion])
    charged = _as_structure(charged_structure).copy()
    if n_ion(charged, working_ion):
        charged.remove_species([working_ion])
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol,
                               primitive_cell=True, scale=True)
    return bool(matcher.fit(host, charged))


def battery_summary(configs, working_ion, mu_ion, z=None,
                    collapse_volume_pct=25.0):
    """Full tier-1 characteristics from the relaxed configuration set.

    Parameters
    ----------
    configs : list[dict]
        {"structure": Structure | as_dict, "energy": float(eV)} per relaxed
        configuration of the common host supercell. Must include the charged
        (no ion) and discharged (all sites) end members.
    working_ion : str
    mu_ion : float
        eV/atom of the elemental ion metal on the same method.
    z : int, optional
        Electrons per ion; defaults to ION_Z[working_ion].
    collapse_volume_pct : float
        |volume change| beyond which the volume_collapse flag is set.

    Returns
    -------
    dict, JSON-ready: avg_voltage, capacity_grav, capacity_vol,
    energy_density (Wh/kg), volume_change_pct, voltage_profile
    (vertices/steps/points), flags {framework_changed, volume_collapse},
    n_sites, working_ion, z.

    Raises
    ------
    KeyError   : unknown working ion and no explicit z.
    ValueError : missing end members (from hull_and_voltages / capacities).
    """
    if z is None:
        z = ION_Z[working_ion]

    parsed = []
    for cfg in configs:
        struct = _as_structure(cfg["structure"])
        parsed.append((struct, float(cfg["energy"]), n_ion(struct, working_ion)))

    profile = hull_and_voltages([(n, e) for _, e, n in parsed], mu_ion, z)
    n_max = max(n for _, _, n in parsed)

    def _best(n_target):
        best = None
        for struct, energy, n in parsed:
            if n == n_target and (best is None or energy < best[1]):
                best = (struct, energy)
        return best

    discharged, _ = _best(n_max)
    charged, _ = _best(0)

    q_grav, q_vol = capacities(discharged, working_ion, z)
    dv = volume_change_pct(discharged, charged)
    matched = framework_match(discharged, charged, working_ion)

    return {
        "working_ion": working_ion,
        "z": int(z),
        "n_sites": int(n_max),
        "avg_voltage": round(float(profile["avg_voltage"]), 4),
        "capacity_grav": round(float(q_grav), 2),      # mAh/g
        "capacity_vol": round(float(q_vol), 2),        # mAh/cm^3
        "energy_density": round(float(q_grav * profile["avg_voltage"]), 1),  # Wh/kg
        "volume_change_pct": round(float(dv), 2),
        "voltage_profile": {
            "vertices": [{k: round(float(v), 6) if k != "n" else int(v)
                          for k, v in vert.items()}
                         for vert in profile["vertices"]],
            "steps": [{k: round(float(v), 4) for k, v in s.items()}
                      for s in profile["steps"]],
            "points": [{k: round(float(v), 6) if k != "n" else int(v)
                        for k, v in p.items()}
                       for p in profile["points"]],
        },
        "flags": {
            "framework_changed": not matched,
            "volume_collapse": abs(dv) > collapse_volume_pct,
        },
    }
