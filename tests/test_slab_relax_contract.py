"""Contract test for the slab-relax runner (codes/files/slab_relax.py).

The global batching in SurfaceBuilderWorkChain mixes slabs from different
bulks in one chunk, so the bulk uuid + epa MUST ride with each slab and be
echoed back -- downstream stages (adsorbates -> photocat -> manual HSE)
identify slabs by the bulk uuid they came from. This test runs the real
runner with ASE's EMT calculator (no MLIP needed).

Run from the repo root:  python -m pytest tests/ -q
"""
import json
import os
import sys

import pytest
from ase.build import bulk as ase_bulk, fcc111
from ase.calculators.emt import EMT
from ase.io import jsonio
from pymatgen.io.ase import AseAtomsAdaptor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "codes", "files"))
import slab_relax  # noqa: E402


def _encoded_cu_slab(vacuum=5.0, rattle=None):
    atoms = fcc111("Cu", size=(1, 1, 3), vacuum=vacuum)
    if rattle:
        atoms.rattle(stdev=rattle, seed=1)
    import numpy as np
    atoms.info["miller_index"] = (1, 1, 1)
    atoms.info["oriented_unit_cell"] = AseAtomsAdaptor.get_structure(
        ase_bulk("Cu", "fcc", a=3.61, cubic=True)).as_dict()
    atoms.info["shift"] = 0.0
    atoms.info["scale_factor"] = np.eye(3, dtype=int)  # as slab_generate ships it
    atoms.info["energy"] = 0.0     # placeholder; runner overwrites post-relax
    return jsonio.encode(atoms)


def _epa_cu():
    atoms = ase_bulk("Cu", "fcc", a=3.61)
    atoms.calc = EMT()
    return atoms.get_potential_energy() / len(atoms)


def test_mixed_bulk_chunk_echoes_uuid_epa_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    epa = _epa_cu()
    payload = [
        {"slab": _encoded_cu_slab(), "uuid": "bulk-A", "epa": epa, "index": 0},
        {"slab": "THIS IS NOT A SLAB", "uuid": "bulk-B", "epa": epa, "index": 1},
        {"slab": _encoded_cu_slab(rattle=0.01), "uuid": "bulk-B",
         "epa": epa + 0.5, "index": 2},   # different epa -> different gamma
    ]
    with open("input_structures.json", "w") as f:
        json.dump(payload, f)

    slab_relax.run_slab_relax(EMT(), epa=None, fmax=0.1, max_steps=200)

    out = json.load(open("output.json"))
    assert out["n_total"] == 3
    assert out["n_failed"] == 1
    assert out["failed_slabs"][0]["uuid"] == "bulk-B"
    assert out["failed_slabs"][0]["index"] == 1

    recs = {r["index"]: r for r in out["slabs"]}
    assert recs[0]["uuid"] == "bulk-A" and recs[2]["uuid"] == "bulk-B"
    # per-slab epa is actually used: same relaxed slab, epa shifted by
    # +0.5 eV/atom -> gamma differs by n_slab*0.5/(2A), i.e. records 0 and 2
    # must NOT share the reference (they'd be near-equal otherwise)
    assert recs[0]["surface_formation_energy"] > recs[2]["surface_formation_energy"]
    # every record reconstructs into a pymatgen Slab with the miller index
    from pymatgen.core.surface import Slab
    slab = Slab.from_dict(recs[0]["slab"])
    assert tuple(slab.miller_index) == (1, 1, 1)


def test_legacy_bare_list_with_cli_epa(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    epa = _epa_cu()
    with open("input_structures.json", "w") as f:
        json.dump([_encoded_cu_slab()], f)

    slab_relax.run_slab_relax(EMT(), epa=epa, fmax=0.1, max_steps=200)

    out = json.load(open("output.json"))
    assert out["n_total"] == 1 and out["n_failed"] == 0
    rec = out["slabs"][0]
    assert rec["uuid"] is None and rec["index"] == 0
    assert rec["surface_formation_energy"] > 0


def test_missing_epa_fails_loudly_per_slab(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    payload = [{"slab": _encoded_cu_slab(), "uuid": "bulk-A", "index": 0}]
    with open("input_structures.json", "w") as f:
        json.dump(payload, f)

    slab_relax.run_slab_relax(EMT(), epa=None, fmax=0.1, max_steps=200)

    out = json.load(open("output.json"))
    assert out["n_failed"] == 1
    assert "no epa" in out["failed_slabs"][0]["reason"]
