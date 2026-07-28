"""Unit tests for the experimental-injection helpers (workchains/exp_include.py).

Run from the repo root:  python -m pytest tests/ -q
No AiiDA, no network -- pymatgen only.
"""
import os
import sys

import pytest
from pymatgen.core import Lattice, Structure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "workchains"))
import exp_include  # noqa: E402


def rocksalt(a=4.2, species=("Li", "O")):
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(a), species, [[0, 0, 0], [0.5, 0.5, 0.5]])


def zincblende(a=4.2, species=("Li", "O")):
    return Structure.from_spacegroup(
        "F-43m", Lattice.cubic(a), species, [[0, 0, 0], [0.25, 0.25, 0.25]])


def _out(structures, energies, indices):
    return {"structures": [s.as_dict() for s in structures],
            "energies": energies, "indices": indices}


def test_split_output_slices_three_way():
    s = rocksalt()
    out = _out([s] * 6, [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], [0, 1, 2, 3, 4, 5])
    gen, exp, ref = exp_include.split_output_slices(out, [3, 2, 1])
    assert [len(gen), len(exp), len(ref)] == [3, 2, 1]
    assert [e.energy for e in exp] == [-4.0, -5.0]


def test_split_output_slices_with_dropped_structures():
    """Non-converged structures are dropped by the runner; the surviving
    indices must still land in the right groups.
    Groups (sizes [3, 2, 2]): gen = idx 0-2, exp = idx 3-4, ref = idx 5-6.
    Inputs 1 (gen), 3 (exp) and 6 (ref) did not converge."""
    s = rocksalt()
    out = _out([s] * 4, [-1.0, -3.0, -5.0, -6.0], [0, 2, 4, 5])
    gen, exp, ref = exp_include.split_output_slices(out, [3, 2, 2])
    assert [e.energy for e in gen] == [-1.0, -3.0]
    assert [e.energy for e in exp] == [-5.0]
    assert [e.energy for e in ref] == [-6.0]


def test_split_output_slices_legacy_no_indices():
    s = rocksalt()
    out = {"structures": [s.as_dict()] * 3, "energies": [-1.0, -2.0, -3.0]}
    gen, ref = exp_include.split_output_slices(out, [2, 1])
    assert len(gen) == 2 and len(ref) == 1


def test_split_output_slices_index_overflow_raises():
    s = rocksalt()
    out = _out([s], [-1.0], [5])
    with pytest.raises(ValueError):
        exp_include.split_output_slices(out, [2, 2])


def test_dedup_forced_drops_equivalent_of_selection():
    selected = [rocksalt()]
    # same rocksalt (slightly scaled -> matcher normalizes volume), plus a
    # genuinely different polymorph
    dup = rocksalt(a=4.25)
    new = zincblende()
    kept = exp_include.dedup_forced(selected, [("u1", dup), ("u2", new)])
    assert [u for u, _ in kept] == ["u2"]


def test_dedup_forced_dedups_candidates_against_each_other():
    """csp and gen may both have stored the same experimental structure --
    only one survives even when the ML selection has nothing."""
    kept = exp_include.dedup_forced(
        [], [("u1", zincblende()), ("u2", zincblende(a=4.3)),
             ("u3", rocksalt())])
    assert [u for u, _ in kept] == ["u1", "u3"]


def test_dedup_forced_accepts_dicts():
    kept = exp_include.dedup_forced([rocksalt().as_dict()],
                                    [("u1", zincblende().as_dict())])
    assert len(kept) == 1
    from pymatgen.core import Structure as S
    assert isinstance(kept[0][1], S)
