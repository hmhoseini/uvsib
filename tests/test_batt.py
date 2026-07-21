"""Unit tests for the pure battery calculator (workchains/batt.py).

Run from the repo root:  python -m pytest tests/ -q
No AiiDA, no database -- pymatgen + numpy only.
"""
import os
import sys

import numpy as np
import pytest
from pymatgen.core import Composition, Lattice, Structure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "workchains"))
import batt  # noqa: E402

MU = -1.90  # fake Li metal reference, eV/atom
Z = 1


def _energies_from_voltages(n_max, voltages, mu=MU, e0=-100.0):
    """Build E(n) that reproduces the given per-step voltages exactly.

    voltages[i] is the plateau between n=i and n=i+1:
        E(n+1) = E(n) + mu - z*V
    """
    energies = {0: e0}
    for n in range(n_max):
        energies[n + 1] = energies[n] + mu - Z * voltages[n]
    return energies


def test_single_plateau_voltage_exact():
    energies = _energies_from_voltages(4, [3.0, 3.0, 3.0, 3.0])
    res = batt.hull_and_voltages(list(energies.items()), MU, Z)
    assert res["avg_voltage"] == pytest.approx(3.0)
    # constant voltage -> intermediate points are ON the tie line, one segment
    assert len(res["steps"]) == 1
    assert res["steps"][0]["voltage"] == pytest.approx(3.0)
    for p in res["points"]:
        assert p["e_above_tieline"] == pytest.approx(0.0, abs=1e-12)


def test_two_plateaus_and_hull_vertex():
    # 3.5 V plateau for the first two ions inserted, 3.0 V for the last two
    # (voltage non-increasing with x <=> E(x) convex) -> hull vertex at n=2
    energies = _energies_from_voltages(4, [3.5, 3.5, 3.0, 3.0])
    res = batt.hull_and_voltages(list(energies.items()), MU, Z)
    assert res["avg_voltage"] == pytest.approx(3.25)
    assert len(res["steps"]) == 2
    # ascending x; voltage must be non-increasing along x (convexity)
    v = [s["voltage"] for s in res["steps"]]
    assert v[0] == pytest.approx(3.5) and v[1] == pytest.approx(3.0)
    assert any(vert["n"] == 2 for vert in res["vertices"])


def test_unstable_intermediate_dropped_from_hull():
    energies = _energies_from_voltages(2, [3.0, 3.0])
    energies[1] += 0.25  # push n=1 above the end-member tie line
    res = batt.hull_and_voltages(list(energies.items()), MU, Z)
    assert [vert["n"] for vert in res["vertices"]] == [0, 2]
    assert len(res["steps"]) == 1
    mid = [p for p in res["points"] if p["n"] == 1][0]
    assert mid["e_above_tieline"] == pytest.approx(0.25)


def test_multiple_configs_per_n_min_wins():
    energies = _energies_from_voltages(2, [3.0, 3.0])
    points = list(energies.items()) + [(1, energies[1] + 1.0)]
    res = batt.hull_and_voltages(points, MU, Z)
    assert res["avg_voltage"] == pytest.approx(3.0)


def test_missing_end_member_raises():
    with pytest.raises(ValueError):
        batt.hull_and_voltages([(1, -5.0), (2, -7.0)], MU, Z)  # no n=0
    with pytest.raises(ValueError):
        batt.hull_and_voltages([(0, -5.0)], MU, Z)


def _dummy_structure(species, volume_per_atom=10.0):
    n = len(species)
    a = (n * volume_per_atom) ** (1 / 3)
    coords = [[i / n, i / n, i / n] for i in range(n)]
    return Structure(Lattice.cubic(a), species, coords)


def test_capacity_lifepo4_convention():
    # theoretical capacity of LiFePO4 is 169.9 mAh/g -- the standard sanity
    struct = _dummy_structure(["Li", "Fe", "P", "O", "O", "O", "O"])
    q_grav, q_vol = batt.capacities(struct, "Li")
    expected = batt.MAH_PER_MOL_E / Composition("LiFePO4").weight
    assert q_grav == pytest.approx(expected)
    assert q_grav == pytest.approx(169.9, abs=0.2)
    # volumetric = gravimetric * density
    density = struct.composition.weight / (struct.volume * batt.A3_TO_CM3_PER_MOL)
    assert q_vol == pytest.approx(q_grav * density)


def test_capacity_multivalent_z():
    struct = _dummy_structure(["Mg", "Mn", "O", "O", "O", "O"])
    q1, _ = batt.capacities(struct, "Mg")            # z = 2 from ION_Z
    q2, _ = batt.capacities(struct, "Mg", z=1)
    assert q1 == pytest.approx(2 * q2)


def test_framework_match_and_volume_change():
    # rocksalt LiCoO2-like toy: host survives -> match
    lat = Lattice.cubic(4.0)
    discharged = Structure(lat, ["Li", "Co", "O", "O"],
                           [[0, 0, 0], [0.5, 0.5, 0.5],
                            [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    charged = discharged.copy()
    charged.remove_species(["Li"])
    charged.scale_lattice(charged.volume * 0.95)
    assert batt.framework_match(discharged, charged, "Li")
    assert batt.volume_change_pct(discharged, charged) == pytest.approx(-5.0)

    # reconstructed host (swap sublattice geometry) -> mismatch
    reconstructed = Structure(Lattice.orthorhombic(2.0, 8.0, 4.1),
                              ["Co", "O", "O"],
                              [[0, 0, 0], [0.5, 0.1, 0.3], [0.2, 0.6, 0.8]])
    assert not batt.framework_match(discharged, reconstructed, "Li")


def test_battery_summary_end_to_end():
    """Synthetic 2-site host with one exact plateau; checks every field."""
    lat = Lattice.cubic(5.0)
    discharged = Structure(lat, ["Li", "Li", "Ti", "Ti", "O", "O", "O", "O"],
                           [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5],
                            [0, 0.5, 0.5], [0.25, 0.25, 0.25],
                            [0.75, 0.75, 0.25], [0.75, 0.25, 0.75],
                            [0.25, 0.75, 0.75]])
    half = discharged.copy(); half.remove_sites([1])
    charged = discharged.copy(); charged.remove_sites([0, 1])
    charged.scale_lattice(charged.volume * 0.94)

    energies = _energies_from_voltages(2, [2.0, 2.0], e0=-80.0)
    configs = [
        {"structure": charged, "energy": energies[0]},
        {"structure": half, "energy": energies[1]},
        {"structure": discharged, "energy": energies[2]},
        {"structure": half, "energy": energies[1] + 0.5},  # worse ordering
    ]
    res = batt.battery_summary(configs, "Li", MU)

    assert res["avg_voltage"] == pytest.approx(2.0)
    assert res["n_sites"] == 2 and res["z"] == 1
    q_expected = 2 * batt.MAH_PER_MOL_E / discharged.composition.weight
    assert res["capacity_grav"] == pytest.approx(q_expected, abs=0.01)
    assert res["energy_density"] == pytest.approx(2.0 * q_expected, abs=0.5)
    assert res["volume_change_pct"] == pytest.approx(-6.0)
    assert res["flags"] == {"framework_changed": False,
                            "volume_collapse": False}
    assert len(res["voltage_profile"]["steps"]) == 1
    # JSON-ready: no numpy scalars anywhere
    import json
    json.dumps(res)
