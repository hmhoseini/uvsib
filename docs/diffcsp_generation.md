---
title: "DiffCSP — a third parallel CSP generator alongside MatterGen and GNoME"
author: "uvsib"
---

# Motivation

[DiffCSP](https://github.com/jiaor17/DiffCSP) (Jiao *et al.*) is a denoising
diffusion model for crystal structure generation. Its `scripts/sample.py`
entry point is genuinely **composition-conditioned**: it fixes the atom types
to a requested formula and diffuses only the lattice and fractional
coordinates. That is exactly the role `CSPWorkChain` needs -- a third,
independently toggleable candidate source for crystal structure prediction,
parallel to MatterGen's CSP mode and GNoME's SAPS-csp mode:

```
csp:  CSPWorkChain
        ├── mattergen.csp ─┐
        ├── gnome.csp ─────┤→ extend csp_structures → ML relax → minima-hopping → store "csp"
        └── diffcsp.csp ───┘
```

`mattergen.csp_enabled`, `gnome.enabled`, and `diffcsp.enabled` are
independently toggled in `input.yaml`; **at least one must be on** for the CSP
path. Whichever are enabled run in parallel per formula and their candidates
are merged. Each is best-effort except MatterGen's own majority-failure check
(disabled by design once GNoME or DiffCSP is also enabled): a failed or empty
branch is logged and the run continues on the other generators' structures.

**DiffCSP is not wired into the `gen` path** (`GeneratorWorkChain`, the
de-novo generator toggled by `mattergen.gen_enabled`/`gnome.enabled` there).
An earlier iteration of this integration tried fitting DiffCSP's *other* script,
`scripts/generation.py`, into that path -- it is an *unconditioned* ab-initio
generator (no `--formula`; it samples atom counts from a fixed per-dataset
training histogram and, for non-carbon datasets, also diffuses atom types),
so making it respect a target chemical system required generating in bulk and
filtering after the fact. `scripts/sample.py`'s real conditioning on a target
formula is a much better fit for `CSPWorkChain`, where a formula is already
the whole point -- so that's where DiffCSP lives now.

# Two DiffCSP scripts, and which one this integration uses

| | `scripts/sample.py` (used here) | `scripts/generation.py` (not used) |
|---|---|---|
| Conditioning | fixes atom types to `--formula`, diffuses lattice + coords only | unconditioned; samples atom count from a fixed per-dataset histogram, may also diffuse atom types |
| Output | one `.cif` per `--num_evals` attempt, under `save_path/formula/` | one `eval_gen.pt` tensor file under `model_path/` |
| Fits | `CSPWorkChain` (a formula is already given) | would need post-hoc filtering to approximate a `gen`-path target chemical system |

# Checkpoints: `diffcsp/mp_csp` and `diffcsp/mp_gen`, not `diffcsp/prop_models`

`sample.py`/`generation.py` load `--model_path` via `eval_utils.load_model()`,
which expects a directory containing `hparams.yaml` and a `*.ckpt` -- e.g. a
DiffCSP checkout's `diffcsp/mp_csp/` (CSP-trained checkpoint) or
`diffcsp/mp_gen/` (gen-trained checkpoint). **`diffcsp/prop_models/<name>/`
is a different thing** -- it's the path `eval_utils.get_model_path()` builds
for *property-predictor* checkpoints (used by the `opt` / property-guided
task), not the structure-generating diffusion model. Pointing `model_path` at
`prop_models` would load the wrong kind of model entirely.

Because checkpoint folder names aren't a derivable convention (nothing
guarantees a `perov_5` CSP checkpoint would be named `perov_5_csp`),
`model_path` is configured as an explicit absolute path in `config.yaml`
rather than constructed from a dataset name.

# Components

```
codes/diffcsp/
  calculation.py   DiffCSPCalculation  — stages the runner + sidecar, retrieves output.json
  parser.py        DiffCSPParser       — output.json -> Dict(structures=[...])  (same as MatterGen/GNoME)
  workchain.py     DiffCSPCSPWorkChain (csp)  — BaseRestart, builds the runner's cmdline
codes/files/
  diffcsp_csp.py   runner (-> aiida.py): shells out to DiffCSP's scripts/sample.py --formula ...,
                   reads back the resulting CIFs, charge/dedup (refine.py) -> output.json
```

Entry points: `diffcsp` (calc), `diffcsp_parser`, `diffcsp.csp`.

## Formula scaling: targeting a non-trivial cell size (`workchain.py`)

`sample.py` diffuses exactly the atom counts parsed from `--formula`
(`chemparse.parse_formula`) -- there is no search over cell size, so left
alone it always targets the reduced (Z=1) formula unit (e.g. 2 atoms for
ZnO). `get_cmdline_csp` in `codes/diffcsp/workchain.py` scales the formula it
passes to `sample.py` by the largest integer coefficient that keeps the cell
at or below 20 atoms -- the same scaling `get_cmdline_csp` in
`codes/mattergen/workchain.py` applies to MatterGen's CSP
`target_compositions`. For any composition with up to 20 atoms this lands in
`[11, 20]` atoms/cell (e.g. ZnO -> `Zn10O10`, coefficient 10); a formula whose
reduced unit already exceeds 20 atoms raises `ValueError` (mirroring
MatterGen's own CSP limit). This only changes what's sent to `sample.py` --
the identity `chemical_formula` used elsewhere in `CSPWorkChain` (reference
entries, dedup, storage) is untouched, and `unique_low_energy_comp` matches
structures by their own `reduced_formula`, so it's agnostic to what cell size
any generator (MatterGen, GNoME, or DiffCSP) actually produced.

## Runner pipeline (`diffcsp_csp.py`)

1. Shell out to `<repo_path>/scripts/sample.py --model_path ... --formula ... --num_evals ... --batch_size ... --step_lr ... --save_path <this job's own working directory>`, with `cwd=repo_path` -- `eval_utils.py` does `sys.path.append('.')` to resolve the `diffcsp` package, so the subprocess must run from the repo root; `--save_path` is passed explicitly (the CalcJob's own working directory, captured before changing `cwd`) so CIFs land there instead of inside the shared repo checkout.
2. `sample.py` writes `<save_path>/<formula>/<formula>_<i>.cif` for each of the `num_evals` attempts (some may be silently skipped by `sample.py` itself if it failed to build a valid structure for that attempt).
3. Read every CIF back into a pymatgen `Structure`.
4. Charge-neutrality filter + primitive-cell de-duplication, reusing `refine.py` (shared with MatterGen/GNoME).

No MLIP energy screen runs here, same reasoning as the earlier gen-path
design: `CSPWorkChain.predict_ml_energies` already relaxes and ML-scores the
merged MatterGen/GNoME/DiffCSP CSP candidates in one shared job downstream,
followed by MinimaHopping -- a second energy pass in the runner would be
redundant.

The output contract is identical to MatterGen's/GNoME's: `output.json` is a
list of pymatgen structure dicts, so nothing downstream changes.

# Configuration

Opt-in for MatterGen/GNoME (off by default); **DiffCSP defaults on** for the
CSP path.

`input.yaml`:

```yaml
diffcsp:
  enabled: true          # flip off if you don't want DiffCSP in the CSP path
DiffCSP_CSP:
  num_evals: 20          # candidate structures sampled per formula
  batch_size: 20         # sample.py's internal batch size (<= num_evals is typical)
  step_lr: 0.00001       # diffusion sampler step size; DiffCSP recommends 1e-5 for an mp_20-trained CSP checkpoint
  num_runs: 1            # number of independent DiffCSPCSPWorkChain submissions per formula (mirrors GNoME_CSP.num_runs)
```

`config.yaml`:

```yaml
codes:
  DiffCSP_CSP:
    code_string: DiffCSP@v100        # an env with torch/torch_geometric/hydra/pymatgen/pyxtal/chemparse/p_tqdm/smact; run_aiida_python wrapper
    repo_path: /data/hossein/DiffCSP # checkout of https://github.com/jiaor17/DiffCSP providing scripts/sample.py
    model_path: /data/hossein/DiffCSP/diffcsp/mp_csp  # CSP-trained checkpoint dir (hparams.yaml + *.ckpt) -- NOT diffcsp/prop_models
    job_script: { nodes: 1, ntasks: 1, cpus: 12, time: 43200, exclusive: False }
```

No `models:` block is needed for DiffCSP -- `repo_path` and `model_path` are
both explicit, machine-specific paths under `codes.DiffCSP_CSP`, not resolved
through a shared `path_to_pretrained_models` tree.

## Parameter reference

| Parameter | Where | What it controls | Default |
|---|---|---|---|
| `diffcsp.enabled` | `input.yaml` | run DiffCSP CSP at all (paired with `mattergen.csp_enabled`/`gnome.enabled`; ≥1 required for the CSP path) | `true` |
| `codes.DiffCSP_CSP.repo_path` | `config.yaml` | path to the DiffCSP checkout providing `scripts/sample.py` | — |
| `codes.DiffCSP_CSP.model_path` | `config.yaml` | CSP-trained checkpoint directory (`hparams.yaml` + `*.ckpt`) | — |
| `DiffCSP_CSP.num_evals` | `input.yaml` | candidate structures sampled per formula | `20` |
| `DiffCSP_CSP.batch_size` | `input.yaml` | `sample.py`'s internal batch size | `20` |
| `DiffCSP_CSP.step_lr` | `input.yaml` | diffusion sampler step size (DiffCSP hyperparameter) | `1e-5` |
| `DiffCSP_CSP.num_runs` | `input.yaml` | number of independent `DiffCSPCSPWorkChain` submissions per formula | `1` |

## Turning it on

1. Register an AiiDA code `DiffCSP@v100` whose executable is the
   `run_aiida_python` wrapper (`python -u aiida.py "$@"`) in an environment
   with torch, torch_geometric, hydra, pytorch_lightning, pymatgen, pyxtal,
   chemparse, p_tqdm, and smact (`sample.py`'s own dependencies).
2. Set `repo_path` and `model_path` in `config.yaml` as above.
3. `diffcsp.enabled` already defaults to `true`; set `mattergen.csp_enabled`
   and/or `gnome.enabled` too if you want them to run alongside it.

Once `DiffCSPCSPWorkChain` finishes successfully for a formula (on its own,
or alongside MatterGen/GNoME), `CSPWorkChain.final_step` stores the resulting
low-energy structures with source `csp` exactly as it does for the
MatterGen/GNoME-only paths -- no changes were needed in
`PhaseDiagramMLWorkChain` itself.

# gen path reminder

DiffCSP doesn't participate in the `gen` path at all (see above). That path
is covered by `mattergen.gen_enabled`, which defaults `True` specifically so
`GeneratorWorkChain` has a default generator; if you turn it off, set
`gnome.enabled: true` or that path fails `ERROR_GENERATIVE_FAILED`. This is
independent from `mattergen.csp_enabled` (default `False`), which governs
whether MatterGen also runs in the CSP path alongside DiffCSP.
