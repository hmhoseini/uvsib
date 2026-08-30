# Building `venv_electronic` on rosi4

Remote Python environment for the AiiDA **`electronic`** code
([`code_electronic_rosi_gpua100.yaml`](../../aiid_computers_and_codes/code_electronic_rosi_gpua100.yaml)).
That code's `prepend_text` does nothing but:

```
source /bigdata/casus/fwuk/mirho50/venv_electronic/bin/activate
```

and then `/bigdata/casus/fwuk/mirho50/bin/run_aiida_python` runs `python -u aiida.py`,
where `aiida.py` is [`uvsib/codes/files/electronic.py`](../codes/files/electronic.py)
staged verbatim. So this venv only has to satisfy that one script:

* `matgl` — MEGNet multi-fidelity band-gap model (`megnet_mfi`, the workhorse)
* `alignn` + `jarvis-tools` — optional ALIGNN cross-check (`alignn_pbe`, `alignn_mbj`)
* `pymatgen` — structure I/O and Mulliken electronegativity

**Host:** `rosi4.fz-rossendorf.de`
**Path:** `/bigdata/casus/fwuk/mirho50/venv_electronic`
**Builder:** `uv` (`~/.local/bin/uv`, ≥ 0.8)

---

## 1. Why these versions (do not "just upgrade")

`aiida.py` is written against the **DGL-era matgl 1.x** and the **pre-ALIGNN2**
`alignn.pretrained.get_prediction(model_name=...)` API. Newer releases break it:

| package | pin | reason |
|---|---|---|
| `matgl` | `==1.1.3` | Last DGL-backend line. 2.x moved to a PyG backend, renamed/relocated the pretrained-model repo, and dropped `MEGNet-MP-2019.4.1-BandGap-mfi` from the old download URL. |
| `dgl` | `==2.1.0` | Required by matgl 1.x MEGNet. Linux/x86-64 **CPU** wheel from `https://data.dgl.ai/wheels/repo.html`. Built against torch 2.2. |
| `torch` | `==2.2.0+cpu` | Matches dgl 2.1.0. **CPU on purpose** — the `gpu-a100` node driver is CUDA 12.2, too old for the cu13 torch that used to be here; ML inference on a handful of structures is sub-second on CPU. |
| `torchdata` | `==0.7.1` | matgl 1.1.3 caps it `<0.8`; 0.7.1 pairs with torch 2.2. |
| `numpy` | `<2` | dgl 2.1.0 is built against the NumPy 1.x ABI. |
| `alignn` | `==2024.5.27` | Still has `mp_gappbe_alignn` / `jv_mbj_bandgap_alignn` and the classic `get_prediction`. 2025+/2026 releases are ALIGNN2 (`ALIGNN2_MODELS`, new API) and a broken legacy shim. |
| `lightning` | `==2.2.5` | Era-appropriate for matgl 1.1.3 (`import lightning.pytorch`). |
| `pymatgen` | latest ok | Only `Structure` + `Element.ionization_energy/electron_affinity` are used. |

Python: **3.10** (matches the DGL / torch 2.2 wheels).

---

## 2. Build

`uv` venvs are **not relocatable** — the absolute path is baked into `bin/activate*`
and every `bin/` console-script shebang. **Create it at the final path.** If you ever
build elsewhere and move it:
`grep -rlI OLDPATH venv_electronic/ | xargs sed -i 's#OLDPATH#NEWPATH#g'`.

```bash
ssh rosi4.fz-rossendorf.de
cd /bigdata/casus/fwuk/mirho50

# keep the current one until the new one is verified
mv venv_electronic venv_electronic.old.$(date +%Y%m%d)

export UV_HTTP_TIMEOUT=180
~/.local/bin/uv venv --python 3.10 /bigdata/casus/fwuk/mirho50/venv_electronic

~/.local/bin/uv pip install \
  --python /bigdata/casus/fwuk/mirho50/venv_electronic/bin/python \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --find-links https://data.dgl.ai/wheels/repo.html \
  "torch==2.2.0+cpu" "torchdata==0.7.1" "numpy<2" \
  "dgl==2.1.0" \
  "matgl==1.1.3" "alignn==2024.5.27" \
  jarvis-tools pymatgen ase "lightning==2.2.5" pydantic pydantic-settings "pyparsing<3"
```

The install copies ~3 GB into `/bigdata` (uv warns it can't hardlink across
filesystems — expected, ignore). A verified lockfile lives beside the venv as
`venv_electronic.freeze.pinned_YYYYMMDD.txt`; regenerate with
`uv pip freeze --python .../venv_electronic/bin/python`.

---

## 3. Pre-stage the pretrained models (mandatory)

Compute nodes cannot download these at run time:

* matgl 1.1.3's built-in URL (`github.com/materialsvirtuallab/matgl/raw/main/pretrained_models/`)
  now **404s** — the repo was renamed and `pretrained_models/` deleted from `main`.
  The files still exist on the **`v1.1.3` tag**.
* ALIGNN's baked figshare URL uses `figshare.com/ndownloader/...`, which returns
  **HTTP 202 with an empty body**. The `ndownloader.figshare.com` host works.

matgl reads `~/.cache/matgl/<model>/` first, so a populated cache means no download:

```bash
M=MEGNet-MP-2019.4.1-BandGap-mfi
D=$HOME/.cache/matgl/$M
mkdir -p "$D"
for f in model.json model.pt state.pt; do
  curl -sSL --fail -o "$D/$f" \
    "https://github.com/materialsvirtuallab/matgl/raw/v1.1.3/pretrained_models/$M/$f"
done
```

ALIGNN caches its zip next to the package. Only needed if `models` ever includes an
`alignn_*` entry (the workchain default is `megnet_mfi` only):

```bash
AP=/bigdata/casus/fwuk/mirho50/venv_electronic/lib/python3.10/site-packages/alignn
curl -sSL --fail -o "$AP/mp_gappbe_alignn.zip"     https://ndownloader.figshare.com/files/31458814
curl -sSL --fail -o "$AP/jv_mbj_bandgap_alignn.zip" https://ndownloader.figshare.com/files/31458694
```

If a **partial/empty** `*.zip` is already there (a failed earlier run), delete it
first — the download branch only fires when the file is absent.

---

## 4. Verify

```bash
RUN=/bigdata/casus/fwuk/mirho50/aiida_calculations/uvsib/2e/d3/a6a7-49e0-4557-b2dd-d799ed3f6fb4
T=/tmp/electronic_itest; rm -rf $T; mkdir -p $T; cd $T
cp "$RUN/aiida.py" "$RUN/input_structures.json" .   # or any real staged inputs
source /bigdata/casus/fwuk/mirho50/venv_electronic/bin/activate
/bigdata/casus/fwuk/mirho50/bin/run_aiida_python \
  --models=megnet_mfi --megnet_fidelity=2 --gap_min=0.4 --gap_max=3.1 --pH=0.0
python -c "import json; d=json.load(open('output.json')); \
print(d['status'], d['config']['models_used']); \
[print(r['uuid'][:8], r['band_info']['gap_eV']) for r in d['results']]"
```

Expect `ok  ['megnet_mfi']` and a **non-null** `gap_eV` per structure.

Also check activation itself (catches the non-relocatable-venv trap):

```bash
bash -c "source /bigdata/casus/fwuk/mirho50/venv_electronic/bin/activate; command -v python && python --version"
```

Then swap: `mv venv_electronic.old.* /somewhere` or delete once happy. No AiiDA
code-node change is needed — the YAML `source`s the same path.

---

## 5. Known-harmless noise

* `matgl … Incompatible model version detected!` on model load — the model still
  loads and predicts.
* `DGL backend not selected … Setting the default backend to "pytorch"` — first-run
  only; writes `~/.dgl/config.json`.
* `pytorch-lightning` may appear in the freeze next to `lightning`; unused, harmless.
* No CUDA / `torch.cuda.is_available() == False` — intentional (see §1).

---

## 6. If you must upgrade later

* Bumping `matgl` past 1.x means rewriting `megnet_gap()` in `electronic.py` for the
  PyG backend **and** sorting out model hosting (matgl ≥ 2 fetches from elsewhere).
* Bumping `alignn` into ALIGNN2 territory means rewriting `alignn_gap()` against
  `alignn.pretrained.ALIGNN2_MODELS` and the new predict entrypoint; the gap model
  keys also change (e.g. `optb88vdw_bandgap_radius`, `mbj_bandgap_radius`,
  `snumat_Band_gap_HSE_radius`).
* `dgl` publishes no new Linux wheels beyond the 2.x line on `data.dgl.ai`; staying
  on the DGL backend effectively pins torch to ≤ 2.4.
* `alignn_gap()` already tolerates both scalar and 1-element-list returns from
  `get_prediction` (see [`electronic.py`](../codes/files/electronic.py)).
