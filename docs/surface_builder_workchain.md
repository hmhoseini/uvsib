# SurfaceBuilderWorkChain maintainer notes

This document explains how `SurfaceBuilderWorkChain` works, which parameters
control it, and how it stays consistent with `PhaseDiagramMLWorkChain`. The
implementation lives in `workchains/surface_builder.py` and is registered as
the AiiDA workflow entry point `surfacebuilder`.

The short version: this work chain reads the selected bulk structures for a
composition, enumerates orthogonal slabs for each bulk, relaxes all generated
slabs in balanced chunks, selects the lowest-surface-energy slabs per bulk, and
stores them in `DBSurface` for downstream adsorbate and reaction workflows.

## Source map

| Code | Role |
|---|---|
| `workchains/surface_builder.py` | Orchestrates bulk lookup, slab generation, relaxation chunking, selection, and database storage. |
| `codes/files/slab_generate.py` | CalcJob runner that computes each bulk energy per atom and generates orthogonal pymatgen slabs. |
| `codes/files/slab_relax.py` | CalcJob runner that relaxes one chunk of slabs and returns surface formation energies. |
| `codes/files/_calculators.py` | Creates the ASE calculator for the configured ML model. |
| `workchains/utils.py` | Provides `get_code()` and `get_model_device()` for code/model lookup. |
| `db/utils.py` | Provides `query_structure()`, `query_by_columns()`, and `add_slab()`. |
| `workflows/settings.py` | Exposes global limits and reads `input.yaml` / `config.yaml`. |

## Inputs

`SurfaceBuilderWorkChain` has one direct AiiDA input:

| Input | AiiDA type | Meaning |
|---|---|---|
| `chemical_formula` | `Str` | Target formula whose selected bulk structures should be converted to slabs. |

`MainWorkChain._construct_surface_builder()` passes only this formula. All
model, scheduler, and slab parameters are read from `settings`.

## Outline

The work chain outline is:

```text
setup
run_slabgen
inspect_slabgen
run_relax
inspect_relax
store_results
final_report
```

The two submitted CalcJob stages use the calculation entry point selected by:

```python
CalculationFactory(str(settings.inputs["face_build"]["model"]).lower())
```

For each job, `get_code(settings.inputs["face_build"]["model"])` selects the
AiiDA code from `config.yaml`.

## Stage behavior

### Setup

`setup()` stores `chemical_formula` in `self.ctx`, then calls
`get_struct_uuid(chemical_formula)`.

The bulk source depends on `settings._PD_VERIFICATION`:

| Mode | Bulk source |
|---|---|
| `_PD_VERIFICATION = False` | Reads `DBComposition.stable_struct["ml_uuid_list"]`, then loads each matching `DBStructureVersion` row using `method = settings.inputs["bulk_relax"]["model"]`. |
| `_PD_VERIFICATION = True` | Queries `DBStructureVersion` rows for the formula using `method = "r2SCAN"`, sorted by energy, excluding `source = "MPDB_ref"`. |

The returned structures are truncated to `settings.MAX_NUM_BULK`. If no
structures are found, the work chain exits with `ERROR_NO_STRUCTURES_FOUND`.

### Slab Generation

`run_slabgen()` submits one slab-generation CalcJob per selected bulk
structure. Each job receives an `input_structures.json` file containing one
record:

```json
[
  {
    "uuid": "<bulk structure uuid>",
    "structure": "<pymatgen Structure.as_dict()>",
    "from_manifest": false
  }
]
```

The staged runner is `slab_generate.py`. It is called with:

| CLI parameter | Value |
|---|---|
| `--ML_model` | `settings.inputs["face_build"]["model"]` |
| `--model` | model name from `get_model_device()` |
| `--model_path` | model path from `get_model_device()` |
| `--task_name` | `settings.inputs["face_build"]["head"]` |
| `--device` | device from `get_model_device()` |
| `--max_miller_idx` | `settings.inputs["face_build"]["max_miller_idx"]` |

Each slab-generation job retrieves `output.json`, parsed through the generic
`sqs_parser`. For each bulk, the runner returns:

```json
{
  "uuid": "<bulk structure uuid>",
  "epa": "<bulk energy per atom, eV>",
  "n_total": "<number of slabs before orthogonality filtering>",
  "n_orth": "<number of orthogonal slabs kept>",
  "slabs": ["<ase.io.jsonio encoded slab>", "..."]
}
```

Important details in `slab_generate.py`:

| Guard / behavior | Meaning |
|---|---|
| `SYMPREC_LADDER = (0.1, 0.05, 0.01)` | Standardizes noisy ML-relaxed cells using loose-to-tight symmetry tolerances. |
| `max_normal_search = 1` | Avoids expensive normal searches; orthogonality is checked afterwards. |
| `MAX_CONV_ATOMS` and `LOWSYM_SG_MAX` | Skip accidental large, low-symmetry cells that can explode slab enumeration. |
| `GEN_TIMEOUT_S` | Per-bulk walltime guard around `generate_all_slabs()`. |
| `process_slab()` | Keeps only slabs whose alpha and beta are within 1 degree of 90 degrees, then sets the c-axis vacuum. |

### Slab-Generation Inspection

`inspect_slabgen()` drops failed generation jobs and bulks with no orthogonal
slabs. For successful bulks it stores only small bookkeeping in
`self.ctx.relax_plan`:

```python
{
    "bulks": {
        "<uuid>": {"epa": 0.0, "n_slabs": 0}
    },
    "chunk_uuids": [
        {"<uuid>": 50}
    ]
}
```

The slabs themselves are not kept in the work-chain checkpoint. They are
re-read from each slab-generation node by `_bulk_slabs()`.

The work chain builds a deterministic global slab list in sorted UUID order and
then chunks it with `MAX_SLABS_PER_CHUNK = 50`. Chunks can contain slabs from
multiple bulks, so every slab item carries its own `uuid`, `epa`, and global
`index`.

If no bulk produced orthogonal slabs, the work chain exits with
`ERROR_NO_SURFACE`.

### Slab Relaxation

`run_relax()` submits one relaxation CalcJob per global chunk. The staged runner
is `slab_relax.py`, and each input item has this shape:

```json
{
  "slab": "<ase.io.jsonio encoded slab>",
  "uuid": "<bulk structure uuid>",
  "epa": "<bulk energy per atom, eV>",
  "index": "<global slab index>"
}
```

The relaxation runner receives:

| CLI parameter | Value |
|---|---|
| `--ML_model` | `settings.inputs["face_build"]["model"]` |
| `--model` | model name from `get_model_device()` |
| `--model_path` | model path from `get_model_device()` |
| `--task_name` | `settings.inputs["face_build"]["head"]` |
| `--device` | device from `get_model_device()` |
| `--fmax` | `settings.inputs["face_build"]["fmax"]` |
| `--max_steps` | `settings.inputs["face_build"]["max_steps"]` |

The runner relaxes each slab with `BFGSLineSearch` and computes:

```text
surface_formation_energy = (E_slab - n_slab * epa) / (2 * area)
```

The per-slab UUID is echoed in each output record. This is load-bearing because
non-converged slabs are dropped, so downstream attribution must not depend on
position or chunk topology.

### Relaxation Inspection

`inspect_relax()` merges all chunk outputs by echoed bulk UUID. Failed chunks
are logged and skipped; failed slabs inside successful chunks are also reported
and skipped.

For each bulk UUID, converged slabs are sorted by
`surface_formation_energy`. The work chain keeps at most
`settings.MAX_NUM_SURF` slabs per bulk. If no slab converges for any bulk, the
work chain exits with `ERROR_NO_SURFACE`.

### Database Storage

`inspect_relax()` keeps `(slab, surface_formation_energy)` pairs per bulk, and
`store_results()` writes each selected pair using:

```python
add_slab(uuid_str, chemical_formula, slab,
         head=settings.inputs["face_build"].get("head"),
         formation_energy=surface_formation_energy)
```

`add_slab()` creates a `DBSurface` row with:

| Field | Value |
|---|---|
| `structure_uuid` | UUID of the parent bulk structure. |
| `composition` | Current `chemical_formula`. |
| `slab` | Selected pymatgen slab dictionary, including energy metadata. |
| `formation_energy` | The slab's `surface_formation_energy` computed during relaxation, stored as its own column (used by `AdsorbatesWorkChain` to rank surfaces). |
| `attributes` | `{"model_head": head}` when `settings.inputs["face_build"]["head"]` is set, else `None`. |

`SurfaceBuilderWorkChain` itself does not update
`DBComposition.stable_struct`. The stable bulk manifest is owned by
`PhaseDiagramMLWorkChain` or, in verification mode, by `PDVerificationWorkChain`
and the r2SCAN structure query.

## Important parameters

Scientific and selection parameters:

| Parameter | Default / source | Meaning |
|---|---|---|
| `settings.MAX_NUM_BULK` | `workflows/settings.py`, currently `10` | Maximum number of bulk structures converted to slabs. |
| `settings.MAX_NUM_SURF` | `workflows/settings.py`, currently `10` | Maximum number of lowest-energy surfaces stored per bulk. |
| `settings._PD_VERIFICATION` | `workflows/settings.py`, currently `False` | Chooses between ML selected bulk UUIDs and r2SCAN-verified structures. |
| `settings.inputs["face_build"]["model"]` | `input.yaml` | ML model used for slab generation energy and slab relaxation. |
| `settings.inputs["face_build"]["head"]` | `input.yaml` | Model head/task passed to the calculator and stored with surfaces. |
| `settings.inputs["face_build"]["max_miller_idx"]` | `input.yaml` | Maximum Miller index for slab enumeration. Higher values increase cost rapidly. |
| `settings.inputs["face_build"]["fmax"]` | `input.yaml` | Force convergence threshold for slab relaxation. |
| `settings.inputs["face_build"]["max_steps"]` | `input.yaml` | Maximum optimizer steps for each slab. |
| `MAX_SLABS_PER_CHUNK` | `workchains/surface_builder.py`, currently `50` | Maximum slabs sent to one relaxation CalcJob. |

Runner-level slab-generation safeguards:

| Parameter | Source | Meaning |
|---|---|---|
| `SYMPREC_LADDER` | `codes/files/slab_generate.py` | Symmetry tolerances used to standardize noisy ML-relaxed bulks. |
| `MAX_CONV_ATOMS` | `codes/files/slab_generate.py` | Atom-count half of the large-low-symmetry skip. |
| `LOWSYM_SG_MAX` | `codes/files/slab_generate.py` | Spacegroup-number half of the large-low-symmetry skip. |
| `GEN_TIMEOUT_S` | `codes/files/slab_generate.py` | Per-bulk timeout for `generate_all_slabs()`. |

Infrastructure parameters:

| Parameter | Used in | Meaning |
|---|---|---|
| `settings.configs["codes"][face_model]["code_string"]` | `get_code()` | AiiDA code used for slab generation and relaxation. |
| `settings.configs["codes"][face_model]["job_script"]` | `_facebuild_options()` | Scheduler resources, wallclock, parser, and optional exclusive node request. |
| `settings.configs["models"]["path_to_pretrained_models"]` | `get_model_device()` | Base directory for model checkpoints. |
| `settings.configs["models"][face_model]` | `get_model_device()` | Model/checkpoint name. |
| `settings.configs["codes"][face_model]["job_script"]["device"]` | `get_model_device()` | Device string passed to the calculator. |

## Consistency With PhaseDiagramMLWorkChain

`PhaseDiagramMLWorkChain` and `SurfaceBuilderWorkChain` are consecutive parts
of the same bulk-to-surface contract:

```text
PhaseDiagramMLWorkChain
  -> DBComposition.stable_struct["ml_uuid_list"]
  -> SurfaceBuilderWorkChain
  -> DBSurface rows keyed by parent bulk structure_uuid
```

The key consistency points are:

| Contract | PhaseDiagramMLWorkChain side | SurfaceBuilderWorkChain side |
|---|---|---|
| Bulk identity | Stores selected structure UUIDs in `stable_struct["ml_uuid_list"]`. | Reads exactly those UUIDs when `_PD_VERIFICATION = False`. |
| Energy method | Stores structures whose `method` equals `ml_bulk_model`, normally `settings.inputs["bulk_relax"]["model"]`. | Loads each selected UUID with `method = settings.inputs["bulk_relax"]["model"]`. |
| Selection limit | `unique_low_energy_comp(..., EHULL_ML, min_n_return=1)` chooses low-energy bulk candidates. | Uses up to `MAX_NUM_BULK` selected bulks and does not re-run hull filtering. |
| Provenance | Each stored UUID points to a `DBStructureVersion` row. | Every generated and relaxed slab carries that bulk UUID through chunking and into `DBSurface.structure_uuid`. |
| Downstream handoff | Produces the bulk manifest for later stages. | Produces the clean-slab rows consumed by adsorbate workflows. |

This means method naming must remain aligned. If
`PhaseDiagramMLWorkChain.store_stable_structs()` stores UUIDs relaxed with one
`ml_bulk_model`, but `SurfaceBuilderWorkChain.get_struct_uuid()` queries the
same UUIDs using a different `settings.inputs["bulk_relax"]["model"]`, the
surface builder may find no rows or may build slabs from an inconsistent energy
set.

`settings.inputs["face_build"]["model"]` can technically differ from
`settings.inputs["bulk_relax"]["model"]`: the former controls slab energy and
slab relaxation, while the latter controls which bulk rows are retrieved from
the phase-diagram manifest. For strict energetic consistency, use the same
model/head family for bulk phase-diagram selection and face building, or record
and justify the mixed-model choice.

When `_PD_VERIFICATION = True`, the surface builder intentionally bypasses
`stable_struct["ml_uuid_list"]` and uses r2SCAN structures instead. That mode is
consistent with `PDVerificationWorkChain`, not with the ML-only bulk manifest.

## MainWorkChain Integration

`MainWorkChain.should_run_surface_builder()` skips this stage when:

| Condition | Reason |
|---|---|
| `settings.SOFT_STOP_BEFORE_SURFACE` is true | The run is configured to stop after generation / phase-diagram / synthesizability stages. |
| `nanoparticles` mode is active | Nanoparticle submissions use a different path. |
| `DBComposition.step_status["surface_builder"] == "Done"` | A shared surface-builder stage already completed for this composition. |

If another workflow has `surface_builder` marked `Running`, `MainWorkChain`
waits and re-checks the database before submitting a duplicate builder.

When this stage starts, `MainWorkChain` writes:

```text
DBComposition.status = "Running"
DBComposition.step_status["surface_builder"] = "Running"
```

On success it writes:

```text
DBComposition.step_status["surface_builder"] = "Done"
```

On failure it writes:

```text
DBComposition.status = "Failed"
DBComposition.step_status["surface_builder"] = "Failed"
```

## Exit codes

| Code | Name | Typical cause |
|---|---|---|
| 300 | `ERROR_CALCULATION_FAILED` | Reserved for failed calculations, although current failed slab-generation and relaxation chunks are usually skipped unless all surfaces are lost. |
| 301 | `ERROR_NO_STRUCTURES_FOUND` | No selected bulk structures were found for the formula. |
| 302 | `ERROR_NO_SURFACE` | No orthogonal slabs were generated or no relaxed slab converged. |

## Practical change notes

- Keep the UUID with each slab item. Relax chunks can mix bulks and can drop
  failed slabs, so UUID attribution cannot be reconstructed from list position.
- Keep `epa` per slab item. Mixed-bulk chunks mean a single chunk-level `--epa`
  is only valid for legacy single-bulk payloads.
- Avoid storing full slab lists in `self.ctx`. The work chain intentionally
  stores counts and re-reads slabs from CalcJob outputs to keep checkpoints
  small.
- If `MAX_SLABS_PER_CHUNK` changes, update this document and check scheduler
  memory/runtime behavior. The current code uses `50`.
- Changes to `PhaseDiagramMLWorkChain.stable_struct` must preserve
  `ml_uuid_list` or `SurfaceBuilderWorkChain` needs a matching reader update.
- Changes to the bulk model key in `input.yaml` affect both phase-diagram
  storage and surface-builder lookup. Treat the string value as a database
  method contract, not only a calculator choice.
- The older `codes/files/slab_relax.py` module docstring mentions chunks of
  `<= 250` slabs; the current work-chain constant is `MAX_SLABS_PER_CHUNK = 50`.
  Prefer the work-chain constant when checking runtime behavior.
