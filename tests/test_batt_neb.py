"""Unit tests for the battery NEB driver (workchains/batt_neb.py).

Run from the repo root:  python -m pytest tests/ -q
No AiiDA, no MLIP -- pymatgen + numpy only.
"""
import os
import sys

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "workchains"))
import batt_neb  # noqa: E402


def layered_licoo2_sc():
    """2x2x1 supercell of O3 LiCoO2 (R-3m) -- 12 Li in flat layers."""
    lat = Lattice.hexagonal(2.82, 14.05)
    prim = Structure.from_spacegroup(
        "R-3m", lat, ["Li", "Co", "O"],
        [[0, 0, 0], [0, 0, 0.5], [0, 0, 0.2396]])
    prim.make_supercell([2, 2, 1])
    return prim


def test_ion_site_classes_single_orbit():
    struct = layered_licoo2_sc()
    classes = batt_neb.ion_site_classes(struct, "Li")
    li_idx = [i for i, s in enumerate(struct) if s.specie.symbol == "Li"]
    assert sorted(classes) == li_idx
    assert len(set(classes.values())) == 1  # all Li equivalent in O3


def test_enumerate_hops_licoo2_in_plane():
    struct = layered_licoo2_sc()
    distinct, edges = batt_neb.enumerate_hops(struct, "Li", max_hop=3.0)
    # one symmetry class: the in-plane nearest-neighbor hop at a = 2.82 A
    assert len(distinct) == 1
    hop = next(iter(distinct.values()))
    assert hop["distance"] == pytest.approx(2.82, abs=0.01)
    # full graph: every Li has 6 in-plane neighbors, directed edges
    n_li = sum(1 for s in struct if s.specie.symbol == "Li")
    assert len(edges) == 6 * n_li
    # every edge carries the class key of the single distinct hop
    assert {e["class_key"] for e in edges} == set(distinct)


def test_enumerate_hops_needs_ion():
    struct = layered_licoo2_sc()
    with pytest.raises(ValueError):
        batt_neb.enumerate_hops(struct, "Na")
    with pytest.raises(ValueError):
        batt_neb.enumerate_hops(struct, "Li", max_hop=1.0)  # nothing in range


def test_vacancy_endpoints_ordering_and_geometry():
    struct = layered_licoo2_sc()
    distinct, _ = batt_neb.enumerate_hops(struct, "Li", max_hop=3.0)
    hop = next(iter(distinct.values()))
    initial, final, moving = batt_neb.hop_endpoints_vacancy(struct, hop, "Li")

    assert len(initial) == len(struct) - 1 == len(final)
    assert initial[moving].specie.symbol == "Li"
    # identical ordering: every non-moving atom is at the same position
    di = initial.cart_coords - final.cart_coords
    moved = np.linalg.norm(di, axis=1)
    assert np.count_nonzero(moved > 1e-8) == 1
    assert moved[moving] == pytest.approx(hop["distance"], abs=1e-6)
    # the moving ion landed on the vacancy site (B, minimum-image)
    target = struct.lattice.get_cartesian_coords(
        struct[hop["b"]].frac_coords + np.array(hop["jimage"]))
    assert np.linalg.norm(final[moving].coords - target) < 1e-8


def test_dilute_endpoints_ordering():
    struct = layered_licoo2_sc()
    host = struct.copy()
    host.remove_species(["Li"])
    initial, final, moving = batt_neb.hop_endpoints_dilute(
        host, [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], (0, 0, 0), "Li")
    assert moving == len(host)  # appended last
    assert len(initial) == len(host) + 1 == len(final)
    assert initial[moving].specie.symbol == "Li"
    assert final[moving].specie.symbol == "Li"
    # host untouched in both
    assert np.allclose(initial.cart_coords[:-1], host.cart_coords)
    assert np.allclose(final.cart_coords[:-1], host.cart_coords)


def _edge(a, b, jimage, key):
    return {"a": a, "b": b, "jimage": jimage, "class_key": key}


def test_percolation_cubic_anisotropic():
    """1 site/cell, direct image hops: x opens at 0.3, y 0.5, z 0.9."""
    edges = [_edge(0, 0, (1, 0, 0), "x"),
             _edge(0, 0, (0, 1, 0), "y"),
             _edge(0, 0, (0, 0, 1), "z")]
    th = batt_neb.percolation_thresholds(
        edges, {"x": 0.3, "y": 0.5, "z": 0.9})
    assert th == {"e_m_1d": 0.3, "e_m_2d": 0.5, "e_m_3d": 0.9}


def test_percolation_1d_only():
    """A single channel direction never percolates in 2D/3D."""
    edges = [_edge(0, 0, (0, 1, 0), "b")]
    th = batt_neb.percolation_thresholds(edges, {"b": 0.27})
    assert th["e_m_1d"] == 0.27
    assert th["e_m_2d"] is None and th["e_m_3d"] is None


def test_percolation_two_sites_no_wrap_is_not_percolation():
    """A bond INSIDE the cell (no image crossing) must not count."""
    edges = [_edge(0, 1, (0, 0, 0), "ab")]
    th = batt_neb.percolation_thresholds(edges, {"ab": 0.1})
    assert th["e_m_1d"] is None
    # add the wrap-around bond -> percolates through both barriers (max=0.4)
    edges.append(_edge(1, 0, (1, 0, 0), "ba"))
    th = batt_neb.percolation_thresholds(edges, {"ab": 0.1, "ba": 0.4})
    assert th["e_m_1d"] == 0.4


def test_percolation_missing_class_never_opens():
    edges = [_edge(0, 0, (1, 0, 0), "x"), _edge(0, 0, (0, 1, 0), "y")]
    th = batt_neb.percolation_thresholds(edges, {"x": 0.2})  # y failed NEB
    assert th["e_m_1d"] == 0.2 and th["e_m_2d"] is None


def test_hop_summary_merge_and_sort():
    distinct = {"k1": {"a": 0, "b": 1, "distance": 2.8},
                "k2": {"a": 0, "b": 2, "distance": 3.1}}
    results = {"k1": {"barrier_fwd": 0.5, "barrier_rev": 0.45,
                      "converged": True, "error": None},
               "k2": {"barrier_fwd": 0.3, "barrier_rev": 0.35,
                      "converged": True, "error": None}}
    rows, barriers = batt_neb.hop_summary(distinct, results)
    assert [r["class_key"] for r in rows] == ["k2", "k1"]  # by barrier
    assert barriers == {"k1": 0.5, "k2": 0.35}  # max(fwd, rev)

    # unconverged hop: reported, but excluded from the percolation barriers
    results["k2"]["converged"] = False
    rows, barriers = batt_neb.hop_summary(distinct, results)
    assert "k2" not in barriers and len(rows) == 2
