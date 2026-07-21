"""Unit tests for the deintercalation enumerator (workchains/battery_enum.py).

Run from the repo root:  python -m pytest tests/ -q
No AiiDA, no database -- pymatgen + numpy only.
"""
import os
import sys

import pytest
from pymatgen.core import Lattice, Structure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "workchains"))
import battery_enum  # noqa: E402


def layered_licoo2():
    """O3 LiCoO2 (R-3m), the classic layered cathode."""
    lat = Lattice.hexagonal(2.82, 14.05)
    return Structure.from_spacegroup(
        "R-3m", lat, ["Li", "Co", "O"],
        [[0, 0, 0], [0, 0, 0.5], [0, 0, 0.2396]])


def metallic_lial():
    """B32 LiAl -- no charge-balanced oxidation guess (Al anion?) is common;
    exercises whichever branch the guesser picks without failing."""
    lat = Lattice.cubic(6.37)
    return Structure.from_spacegroup(
        "Fd-3m", lat, ["Li", "Al"], [[0.625, 0.625, 0.625], [0, 0, 0]])


def test_build_supercell_caps_atoms_and_counts_ions():
    sc, n = battery_enum.build_supercell(layered_licoo2(), "Li", max_atoms=64)
    assert sc.num_sites <= 64
    assert n == sum(1 for s in sc if s.specie.symbol == "Li")
    assert n >= 8  # a real supercell, not the primitive cell
    # supercell not degenerate: shortest lattice vector grew
    assert min(sc.lattice.abc) > 2.9


def test_missing_ion_raises():
    with pytest.raises(ValueError):
        battery_enum.build_supercell(layered_licoo2(), "Na")


def test_x_counts_grid():
    assert battery_enum.x_counts(12, 4) == [0, 2, 5, 7, 10, 12]
    assert battery_enum.x_counts(2, 4) == [0, 1, 2]  # collapses, keeps ends


def test_enumerate_licoo2_ewald_ranked():
    plan = battery_enum.enumerate_deintercalation(
        layered_licoo2(), "Li", n_x_steps=2, max_configs_per_x=3,
        supercell_max_atoms=48, seed=1)
    assert plan["ewald_ranked"] is True
    n = plan["n_sites"]
    assert plan["counts"][0] == 0 and plan["counts"][-1] == n
    # end members: exactly one config each; correct ion counts everywhere
    for k, structs in plan["configs"].items():
        assert len(structs) == 1 if k in (0, n) else len(structs) >= 1
        for s in structs:
            n_li = sum(1 for site in s if site.specie.symbol == "Li")
            assert n_li == k
            # host sublattice untouched
            n_co = sum(1 for site in s if site.specie.symbol == "Co")
            assert n_co == sum(1 for site in plan["supercell"]
                               if site.specie.symbol == "Co")
    # intermediates capped
    for k in plan["counts"][1:-1]:
        assert len(plan["configs"][k]) <= 3


def test_enumerate_deterministic_with_seed():
    kwargs = dict(n_x_steps=2, max_configs_per_x=2,
                  supercell_max_atoms=48, seed=7)
    p1 = battery_enum.enumerate_deintercalation(layered_licoo2(), "Li", **kwargs)
    p2 = battery_enum.enumerate_deintercalation(layered_licoo2(), "Li", **kwargs)
    for k in p1["counts"]:
        assert [s.frac_coords.tolist() for s in p1["configs"][k]] == \
               [s.frac_coords.tolist() for s in p2["configs"][k]]


def test_enumerate_metallic_fallback_runs():
    """A host without a clean ionic picture must still enumerate (random
    branch) and produce valid configs -- never crash on missing oxi states."""
    plan = battery_enum.enumerate_deintercalation(
        metallic_lial(), "Li", n_x_steps=1, max_configs_per_x=2,
        supercell_max_atoms=32, seed=3)
    n = plan["n_sites"]
    for k, structs in plan["configs"].items():
        assert structs, f"no configs at k={k}"
        for s in structs:
            assert sum(1 for site in s if site.specie.symbol == "Li") == k


def test_configs_are_distinct():
    plan = battery_enum.enumerate_deintercalation(
        layered_licoo2(), "Li", n_x_steps=1, max_configs_per_x=4,
        supercell_max_atoms=48, seed=5)
    from pymatgen.analysis.structure_matcher import StructureMatcher
    matcher = StructureMatcher(primitive_cell=False, scale=False)
    for k in plan["counts"][1:-1]:
        structs = plan["configs"][k]
        for i in range(len(structs)):
            for j in range(i + 1, len(structs)):
                assert not matcher.fit(structs[i], structs[j])
