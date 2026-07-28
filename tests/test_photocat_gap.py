"""Unit tests for the photocat gap-ensemble math (codes/files/photocat_gap.py).

Run from the repo root:  python -m pytest tests/ -q
No AiiDA, no ML packages -- the pure functions only.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "codes", "files"))
import photocat_gap as pg  # noqa: E402

FID = pg.FIDELITY  # alignn_mbj: mbj, alignn_opt/megnet_pbe: pbe, modnet_expt: expt


def test_ensemble_stats_expt_target_members_only():
    gaps = {"alignn_mbj": 2.0, "modnet_expt": 2.4, "megnet_pbe": 1.2}
    mean, spread, primary = pg.ensemble_stats(gaps, FID)
    assert mean == pytest.approx(2.2)
    assert spread == pytest.approx(0.4)
    assert primary == ["alignn_mbj", "modnet_expt"]   # pbe member excluded


def test_ensemble_stats_single_and_empty():
    mean, spread, primary = pg.ensemble_stats({"alignn_mbj": 2.0}, FID)
    assert mean == 2.0 and spread is None and primary == ["alignn_mbj"]
    mean, spread, primary = pg.ensemble_stats({"megnet_pbe": 1.0}, FID)
    assert mean is None and primary == []
    mean, _, _ = pg.ensemble_stats({"alignn_mbj": None, "modnet_expt": 2.2}, FID)
    assert mean == 2.2                                 # None gaps skipped


def test_failure_probability_window():
    # gap right at the lower edge -> 50% failure, no upper edge
    p, sigma = pg.failure_probability(1.8, None, 1.8, None, 0.5, 0)
    assert p == pytest.approx(0.5)
    assert sigma == pytest.approx(0.5)
    # comfortably inside a window -> small
    p, _ = pg.failure_probability(2.5, 0.1, 1.5, 3.5, 0.5, 0)
    assert p < 0.05
    # metal vs gap_min 1.5 -> essentially certain failure
    p, _ = pg.failure_probability(0.0, None, 1.5, None, 0.5, 0)
    assert p > 0.99
    # disagreement widens sigma and pulls p toward 0.5
    p_tight, _ = pg.failure_probability(2.3, 0.0, 1.8, None, 0.5, 0)
    p_wide, s_wide = pg.failure_probability(2.3, 1.0, 1.8, None, 0.5, 0)
    assert p_wide > p_tight
    assert s_wide == pytest.approx(math.sqrt(0.5**2 + 0.5**2))
    # suspicion flags widen sigma too
    _, s_flag = pg.failure_probability(2.3, None, 1.8, None, 0.5, 2)
    assert s_flag == pytest.approx(math.sqrt(0.5**2 + 0.5**2))


def test_assess_clean_candidate():
    gaps = {"alignn_mbj": 2.6, "modnet_expt": 2.5, "megnet_pbe": 1.9}
    rec = pg.assess(gaps, FID, {"Ti", "O"}, 1.8, 3.5, 0.5)
    assert rec["gap_mean"] == pytest.approx(2.55)
    assert rec["flags"] == []
    assert rec["p_fail"] < 0.1


def test_assess_d10_flag_and_sigma_bump():
    gaps = {"alignn_mbj": 2.0, "modnet_expt": 2.0}
    clean = pg.assess(gaps, FID, {"Ti", "O"}, 1.8, None, 0.5)
    cu = pg.assess(gaps, FID, {"Cu", "O"}, 1.8, None, 0.5)
    assert "d10_cu_ag_mbj_unreliable" in cu["flags"]
    assert cu["sigma_eff"] > clean["sigma_eff"]
    assert cu["p_fail"] > clean["p_fail"]     # closer to coin-flip near the edge


def test_assess_fidelity_inversion():
    # PBE above the expt-target mean is physically backwards
    gaps = {"alignn_mbj": 1.6, "megnet_pbe": 2.4}
    rec = pg.assess(gaps, FID, {"Ti", "O"}, 1.5, None, 0.5)
    assert "fidelity_inversion" in rec["flags"]
    assert "single_expt_target_model" in rec["flags"]


def test_assess_metal_and_no_model():
    rec = pg.assess({"alignn_mbj": 0.0}, FID, {"Fe"}, 1.5, None, 0.5)
    assert "predicted_metal" in rec["flags"]
    assert rec["p_fail"] > 0.99

    rec = pg.assess({"megnet_pbe": 1.0}, FID, {"Ti", "O"}, 1.5, None, 0.5)
    assert rec["gap_mean"] is None
    assert rec["p_fail"] == 1.0
    assert "no_expt_target_model" in rec["flags"]


def test_runner_end_to_end_with_mock_backends(tmp_path, monkeypatch):
    """Full run() pass on the runner I/O contract -- fake predictors stand in
    for the ML packages; per-model failure is recorded, not fatal."""
    import json
    from pymatgen.core import Lattice, Structure

    rutile = Structure(Lattice.tetragonal(4.6, 2.95), ["Ti", "Ti", "O", "O", "O", "O"],
                       [[0, 0, 0], [0.5, 0.5, 0.5], [0.3, 0.3, 0], [0.7, 0.7, 0],
                        [0.8, 0.2, 0.5], [0.2, 0.8, 0.5]])
    cu2o = Structure(Lattice.cubic(4.27), ["Cu", "Cu", "Cu", "Cu", "O", "O"],
                     [[0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75],
                      [0.25, 0.75, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]])

    def _boom():
        raise ImportError("no tensorflow")

    monkeypatch.setitem(pg.BACKEND_FACTORIES, "alignn_mbj",
                        lambda: (lambda s: 3.0 if "Ti" in s.formula else 1.0))
    monkeypatch.setitem(pg.BACKEND_FACTORIES, "modnet_expt",
                        lambda: (lambda s: 3.2 if "Ti" in s.formula else 2.0))
    monkeypatch.setitem(pg.BACKEND_FACTORIES, "megnet_pbe", _boom)

    monkeypatch.chdir(tmp_path)
    with open("input_structures.json", "w") as f:
        json.dump([{"structure": rutile.as_dict(), "tag": "uuid-tio2"},
                   {"structure": cu2o.as_dict(), "tag": "uuid-cu2o"}], f)

    pg.run(["alignn_mbj", "modnet_expt", "megnet_pbe"], 1.8, 3.5, 0.5)

    out = json.load(open("output.json"))
    assert int(open("total.txt").read()) == 2
    assert int(open("failed.txt").read()) == 0
    by_tag = {r["tag"]: r for r in out["results"]}
    tio2, cu = by_tag["uuid-tio2"], by_tag["uuid-cu2o"]
    assert tio2["gap_mean"] == pytest.approx(3.1)
    assert tio2["p_fail"] < 0.35
    assert "ImportError" in tio2["errors"]["megnet_pbe"]
    # Cu2O: big disagreement + d10 flag -> high failure probability
    assert "d10_cu_ag_mbj_unreliable" in cu["flags"]
    assert cu["spread"] == pytest.approx(1.0)
    assert cu["p_fail"] > 0.5


def test_assess_is_json_ready():
    import json
    rec = pg.assess({"alignn_mbj": 2.0, "megnet_pbe": None}, FID,
                    {"Cu", "O"}, 1.8, 3.5, 0.5,
                    errors={"megnet_pbe": "ImportError: no tensorflow"})
    json.dumps(rec)
    assert rec["errors"]["megnet_pbe"].startswith("ImportError")
