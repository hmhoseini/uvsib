# AdsorbatesWorkChain maintainer notes

This document explains how `AdsorbatesWorkChain` works, which parameters
control it, what it stores, and how it depends on `SurfaceBuilderWorkChain`.
The registered AiiDA workflow entry point is `adsorbates`, implemented in
`workchains/adsorbates.py`.

The short version: this work chain reads the clean slab rows produced by
`SurfaceBuilderWorkChain`, selects the lowest-energy stored surfaces, generates
and relaxes reaction-specific adsorbate structures with the configured ML
model, computes CHE overpotentials for the requested reaction pathway, and
stores the results in `DBSurfaceMLAdsorbate`.

## Source map

| Code | Role |
|---|---|
| `workchains/adsorbates.py` | Orchestrates slab lookup, ML adsorbate jobs, overpotential calculation, and database storage. |
| `codes/files/adsorbates.py` | CalcJob runner that builds adsorption sites/intermediates, relaxes gas references, clean slabs, and adsorbate slabs, validates relaxed adsorbates, and writes `output.json`. |
| `workchains/oer.py`, `co2rr.py`, `noxrr.py`, `cer.py`, `her.py`, `nrr.py`, `orr.py` | Reaction-specific CHE overpotential calculators. |
| `docs/reactions_and_pathways.md` | Human-readable registry of implemented reactions and pathway names. |
| `docs/sanity_checks.md` | Details of post-relaxation adsorbate validation in the runner. |
| `db/utils.py` | Provides `query_by_columns()` and `add_surface_ml_adsorbate()`. (`get_structure_uuid_surface_id()` used to live here; it is now defined directly in `workchains/adsorbates.py` and the old join-based version in `db/utils.py` is commented out.) |
| `db/tables.py` | Defines `DBSurface` and `DBSurfaceMLAdsorbate`. |
| `workchains/utils.py` | Provides `get_code()` and `get_model_device()` for code/model lookup. |
| `workflows/settings.py` | Exposes global constants and reads `input.yaml` / `config.yaml`. |

`workchains/adsorbates_dft.py` is a separate DFT-verification variant with a
different input signature and extra VASP stages. It is not the registered
`adsorbates` entry point in `setup.json`.

## Inputs

`AdsorbatesWorkChain` has three direct AiiDA inputs:

| Input | AiiDA type | Meaning |
|---|---|---|
| `chemical_formula` | `Str` | Target formula whose stored surfaces should be screened. |
| `reaction` | `Str` | Reaction dispatch key, for example `OER`, `CO2RR`, `NOXRR`, `CER`, `HER`, `NRR`, or `ORR`. |
| `reaction_path` | `Str` | Pathway name passed to the reaction-specific adsorbate generator and overpotential calculator. OER ignores the pathway. |

`MainWorkChain._construct_adsorbates_builder()` passes the formula, reaction,
and reaction path. The ML model, scheduler code, force threshold, and maximum
relaxation steps are read from `settings.inputs["adsorbates"]`.

## Outline

The work-chain outline is:

```text
setup
run_adsorbs
inspect_adsorbs
store_results_ml
final_report
```

Each submitted adsorbate sub-workflow is selected by:

```python
WorkflowFactory(settings.inputs["adsorbates"]["model"].lower())
```

That means the adsorbate stage uses the same generic ML workflow family as
other ML jobs in the project, but with `job_info["job_type"] = "adsorbates"`.

## Stage behavior

### Setup

`setup()` stores the input values in `self.ctx`, reports that the work chain
is running, and calls the module-level helper defined in
`workchains/adsorbates.py` itself:

```python
def get_structure_uuid_surface_id(chemical_formula):
    rows = query_by_columns(DBSurface, {"composition": chemical_formula})
    return [(row.structure_uuid, row.id) for row in rows]
```

This queries `DBSurface` directly by `composition` (no join to `DBStructure`)
and returns `(structure_uuid, surface_id)` tuples for every stored surface of
that formula.

If no rows are found, the work chain reports an error and exits with
`ERROR_NO_SURFACE_FOUND` (renamed from `ERROR_NO_STRUCTURES_FOUND`; the check
now happens after the initial "Running Adsorbates WorkChain..." report line,
not before it). In practice, this usually means the surface stage has not
produced any `DBSurface` rows for the formula yet.

### Surface Selection

`run_adsorbs()` groups the `(structure_uuid, surface_id)` rows **by parent
bulk uuid** rather than treating them as one flat pool:

```python
by_uuid.setdefault(str(structure_uuid), []).append((sfe, slab_row, surface_id))
```

where `sfe` is `DBSurface.formation_energy` (falling back to `float("inf")`
if it is `None`, so unranked surfaces sort last instead of crashing the
comparison). Within each bulk's group, rows are sorted by `sfe` and truncated
to the first `MAX_NUM_ADS` (`settings.MAX_NUM_ADS`, currently `10`). The work
chain submits adsorbate jobs for every surface kept after that per-bulk cut
and records their `(structure_uuid, surface_id)` pairs in
`self.ctx.selected_surfaces`, reporting how many of each bulk's surfaces were
considered.

Important consequence: `SurfaceBuilderWorkChain` may store up to
`settings.MAX_NUM_BULK * settings.MAX_NUM_SURF` surfaces, and
`AdsorbatesWorkChain` now screens up to `MAX_NUM_ADS` lowest-formation-energy
surfaces **for each bulk uuid** (not 10 total across the whole composition),
so multiple bulk structures no longer compete for the same global slot.

### Adsorbate ML Job Construction

`_construct_adsorbate_builder()` creates a builder for the configured ML model.
It passes one clean slab dictionary as `input_structures`:

```python
builder.input_structures = List([slab])
```

The sub-workflow receives this `job_info` payload:

| Key | Value |
|---|---|
| `job_type` | `"adsorbates"` |
| `ML_model` | `settings.inputs["adsorbates"]["model"]` |
| `model_name` | Model name from `get_model_device()` |
| `model_path` | Model checkpoint path from `get_model_device()` |
| `model_head` | `settings.inputs["adsorbates"]["head"]` |
| `device` | Device from `get_model_device()` |
| `slab_energy` | `slab["energy"]` from the selected `DBSurface` row |
| `fmax` | `settings.inputs["adsorbates"]["fmax"]` |
| `max_steps` | `settings.inputs["adsorbates"]["max_steps"]` |
| `reaction` | Requested reaction key |
| `pathway` | Requested reaction path |

The generic ML workflow converts `model_head` to the runner CLI argument
`--task_name` through `uvsib.codes.utils.get_cmdline()`.

The runner CLI in `codes/files/adsorbates.py` also supports validation flags
such as `--no-validate`, `--check-slab-integrity`, and
`--check-energy-outliers`. In the current registered path,
`uvsib.codes.utils.get_cmdline()` always appends `--no-validate` for
`job_type == "adsorbates"`, so the default runner validation layers are
disabled unless that command-line construction is changed.

### Runner Behavior

For each selected clean slab, `codes/files/adsorbates.py`:

1. Reads `input_structures.json` and reconstructs the pymatgen `Slab`.
2. Dispatches to the requested reaction/pathway generator.
3. Builds the required gas-phase reference set for the pathway.
4. Finds adsorption sites with `AdsorbateSiteFinder`.
5. Tests several surface repeats from `get_multipliers()`.
6. Places all required intermediates on ontop, bridge, and hollow sites.
7. Drops initial placements with unreasonable short distances.
8. Relaxes gas references once, storing per-molecule reference energies.
9. Relaxes the clean slab for each repeat.
10. Relaxes every adsorbate in a complete site/intermediate set.
11. Applies post-relaxation sanity checks only if validation is enabled. The
    runner default is enabled, but the registered workflow currently passes
    `--no-validate`.
12. Writes `output.json`, `total.txt`, `failed.txt`, and `rejected.json`.

Only complete relaxed sets are returned. If one intermediate in a site set
fails relaxation or validation, that set is dropped and other sites continue.

The retrieved `output.json` has this shape:

```json
{
  "structures": [
    {
      "site_type": "ontop",
      "ads_coord": [0.0, 0.0, 0.0],
      "repeat": [1, 1, 1],
      "structures": [
        "<ase.io.jsonio encoded adsorbate slab>",
        "<ase.io.jsonio encoded gas reference>",
        "<ase.io.jsonio encoded clean slab>"
      ]
    }
  ]
}
```

Each encoded ASE structure carries `atoms.info["adsorbate"]`. Energies are
stored under the lower-case model key, for example `mace_energy`,
`mattergen_energy`, `mattersim_energy`, or `uma_energy`.

### Adsorbate Job Inspection

`inspect_adsorbs()` checks each selected sub-workflow. Failed jobs are reported
and skipped. If every selected sub-workflow fails, the work chain exits with
`ERROR_CALCULATION_FAILED`.

For successful jobs, the work chain stores:

```python
self.ctx.ml_results["<structure_uuid>_<surface_id>"] = output_dict["structures"]
```

### ML Result Storage

`store_results_ml()` maps the reaction key to an overpotential calculator:

| Reaction | Calculator | Nominal eta threshold in code |
|---|---|---|
| `OER` | `calculate_oer_overpotential` | `2.0` |
| `CO2RR` | `calculate_co2rr_overpotential` | `2.0` |
| `CER` | `calculate_cer_overpotential` | `2.0` |
| `NRR` | `calculate_nrr_overpotential` | `2.0` |
| `NOXRR` | `calculate_noxrr_overpotential` | `2.0` |
| `HER` | `calculate_her_overpotential` | `2.0` |
| `ORR` | `calculate_orr_overpotential` | `2.0` |

If the reaction key is unknown, the work chain exits with
`ERROR_CALCULATION_FAILED`.

For each relaxed adsorption set, the work chain decodes every returned ASE
structure and builds:

```python
energy_set[adsorbed.info["adsorbate"]] = adsorbed.info["<model>_energy"]
```

The energy set includes surface intermediates, gas references, and the clean
slab marker `*`. The reaction calculator consumes this dictionary plus
`reaction_path` and returns:

```text
eta, dG_steps, dG_cumulative
```

If a pathway-required intermediate is missing, that adsorption set is skipped
with a report message. This can happen when the runner drops a fragile relaxed
intermediate during validation.

The eta-threshold filter is active: candidates with `eta > eta_threshold`
(currently `2.0` for every reaction) are skipped before storage. Only
candidates within the threshold increment `self.ctx.candidates` and get
stored, so the final-report phrase "below eta threshold" is accurate for the
current ML path.

### Database Storage

Each stored result is inserted with:

```python
add_surface_ml_adsorbate(
    existing_uuid=structure_uuid,
    surf_id=surface_id,
    surface_miller_index=miller_index,
    comp=chemical_formula,
    react=reaction,
    react_path=reaction_path,
    site_type=site_type,
    ads_coord=ads_coord,
    repeat=repeat,
    e=eta,
    dG_steps=dG_steps,
    dG_cumulative=dG_cumulative,
    ad_set=adsorb_set,
)
```

This creates a `DBSurfaceMLAdsorbate` row with:

| Field | Value |
|---|---|
| `structure_uuid` | Parent bulk UUID inherited from `DBSurface`. |
| `surface_id` | Primary key of the clean slab row in `DBSurface`. |
| `surface_miller_index` | `slab["miller_index"]` from the clean slab row. |
| `composition` | Current `chemical_formula`. |
| `reaction` | Reaction key. |
| `reaction_path` | Pathway name. |
| `site_type` | Adsorption site class, usually `ontop`, `bridge`, or `hollow`. |
| `ads_coord` | Adsorption coordinate returned by pymatgen. |
| `repeat` | Surface repeat used for this adsorption set. |
| `eta` | Computed overpotential. |
| `dG_steps` | Step-wise free energies from the reaction calculator. |
| `dG_cumulative` | Cumulative pathway free energies. |
| `adsorb_set` | JSON payload containing relaxed encoded structures and metadata. |

### Final Report

If no eta values were stored, `final_report()` returns
`NO_CANDIDATES_WITHIN_ETA_LIMIT`. Otherwise it reports the number of eta values
stored for the requested reaction path.

## Important parameters

Direct scientific and selection parameters:

| Parameter | Source | Meaning |
|---|---|---|
| `chemical_formula` | Work-chain input | Composition whose stored surfaces should be screened. |
| `reaction` | Work-chain input | Reaction calculator and adsorbate generator dispatch key. |
| `reaction_path` | Work-chain input | Pathway name used by the generator and CHE calculator. |
| `settings.MAX_NUM_ADS` | `workflows/settings.py`, currently `10` | Maximum number of lowest-formation-energy `DBSurface` rows screened **per bulk uuid** (not globally). |

Adsorbate ML relaxation parameters:

| Parameter | Source | Meaning |
|---|---|---|
| `settings.inputs["adsorbates"]["model"]` | `input.yaml` | ML model/workflow used for adsorbate relaxation. |
| `settings.inputs["adsorbates"]["head"]` | `input.yaml` | Model head/task passed as `model_head` in `job_info`. |
| `settings.inputs["adsorbates"]["fmax"]` | `input.yaml` | Force convergence threshold for gas references, clean slabs, and adsorbate slabs. |
| `settings.inputs["adsorbates"]["max_steps"]` | `input.yaml` | Maximum optimizer steps for each relaxation. |
| `settings.configs["codes"][ads_model]["code_string"]` | `get_code()` | AiiDA code used by the selected ML workflow. |
| `settings.configs["models"][ads_model]` | `get_model_device()` | Model/checkpoint name. |
| `settings.configs["models"]["path_to_pretrained_models"]` | `get_model_device()` | Base directory for model checkpoints. |
| `settings.configs["codes"][ads_model]["job_script"]["device"]` | `get_model_device()` | Device string passed to the calculator. |

Runner-level generation and validation parameters:

| Parameter | Source | Meaning |
|---|---|---|
| Reaction/pathway registry | `codes/files/adsorbates.py` and `docs/reactions_and_pathways.md` | Defines the intermediates, gas references, and pathway steps. |
| Site types | `codes/files/adsorbates.py`, `ontop`, `bridge`, `hollow` | Adsorption site classes sampled for each slab. |
| `get_multipliers()` | `codes/files/adsorbates.py` | Surface repeat candidates used before adsorbate placement. |
| `has_reasonable_distances()` | `codes/files/adsorbates.py` | Pre-relaxation filter for obviously bad atom overlaps. |
| `validate_adsorbates` | Runner default `True`; registered workflow currently passes `--no-validate` | Enables post-relaxation layers 0-2 unless disabled. |
| `bind_tol` | Runner default `1.25` | Surface-binding distance tolerance. |
| `graph_tol` | Runner default `_ADSORBATE_BOND_TOL = 1.25` | Molecular-identity bond graph tolerance. |
| `check_slab_integrity` | Runner default `False` | Optional slab reconstruction filter. |
| `check_energy_outliers` | Runner default `False` | Optional MAD outlier filter across adsorption-site energies. |

Global pipeline parameters:

| Parameter | Source | Meaning |
|---|---|---|
| `settings.SOFT_STOP_BEFORE_SURFACE` | `input.yaml` via `settings.py` | If true, `MainWorkChain` stops before both surface and adsorbate stages. |
| `settings.AKMC_ENABLED` | `input.yaml` via `settings.py` | Controls whether `MainWorkChain` proceeds to AKMC after adsorbates. |

## Connection to SurfaceBuilderWorkChain

`SurfaceBuilderWorkChain` and `AdsorbatesWorkChain` are consecutive stages of
the surface-screening contract:

```text
SurfaceBuilderWorkChain
  -> DBSurface rows
  -> AdsorbatesWorkChain
  -> DBSurfaceMLAdsorbate rows
```

The connection is through the database, not through a direct AiiDA output link.

| Contract | SurfaceBuilderWorkChain side | AdsorbatesWorkChain side |
|---|---|---|
| Clean slab creation | Enumerates, relaxes, selects, and stores slabs with `add_slab()`. | Reads those rows through the local `get_structure_uuid_surface_id()` and `query_by_columns(DBSurface, {"id": surface_id})`. |
| Parent bulk identity | Stores `DBSurface.structure_uuid` as the UUID of the bulk that produced the slab. | Carries that UUID into `DBSurfaceMLAdsorbate.structure_uuid`, and also groups candidate surfaces by it before ranking. |
| Composition | Stores `DBSurface.composition = chemical_formula`. | Uses the formula to find all surfaces for the composition. |
| Surface identity | Each clean slab row has a database `id`. | Stores the same `surface_id` in every adsorbate result. |
| Surface ranking | Stores `DBSurface.formation_energy` (the relaxed `surface_formation_energy`). | Sorts candidate surfaces **within each bulk uuid** by `formation_energy` (missing values sort last) and keeps up to `MAX_NUM_ADS` per bulk. |
| Surface metadata | Stores the slab dictionary, including energy and `miller_index`. | Reads `slab["miller_index"]` for storage and passes the full slab to the adsorbate runner. |
| Model provenance | Stores optional `DBSurface.attributes["model_head"]` from face building. | Uses separate `settings.inputs["adsorbates"]` model/head settings for adsorbate relaxation. |

In `MainWorkChain`, the surface stage is run and inspected before the adsorbate
stage:

```text
surface_builder
inspect_surface_builder
adsorbates
inspect_adsorbates
```

On success, `MainWorkChain` marks:

```text
DBComposition.step_status["surface_builder"] = "Done"
```

For adsorbates, status is tracked per `(reaction, reaction_path)`:

```text
DBComposition.step_status["adsorbates"][reaction][reaction_path] = "Running|Done|Failed"
```

`should_run_adsorbates()` skips a completed reaction/pathway only when the
status is `Done` and at least one matching `DBSurfaceMLAdsorbate` row exists.

## Practical implications of the handoff

- `AdsorbatesWorkChain` requires `DBSurface` rows. Running it before the
  surface builder completes will usually produce `ERROR_NO_SURFACE_FOUND`.
- The clean slab row is the parent object for all adsorbate results. Deleting
  or replacing `DBSurface` rows after adsorbates have run can orphan the
  scientific meaning of stored `DBSurfaceMLAdsorbate` rows even if the database
  foreign keys remain valid.
- The adsorbate stage now ranks candidates by `DBSurface.formation_energy`
  (the stored `surface_formation_energy`), grouped per bulk uuid, instead of
  the flat `slab["energy"]` sort used previously. If `formation_energy` stops
  being populated by `SurfaceBuilderWorkChain.store_results()` (or `slab_relax.py`
  stops computing it), rows silently sort last (`float("inf")` fallback)
  rather than raising — update this document if that fallback behavior changes.
- Surface-builder and adsorbate models are configured independently:
  `settings.inputs["face_build"]` controls clean slab generation/relaxation,
  while `settings.inputs["adsorbates"]` controls adsorbate relaxation. Mixed
  models are possible, but the result should be treated as a mixed-model
  screening choice.
- `DBSurface.structure_uuid` attribution is load-bearing. It connects bulk
  selection, clean slabs, adsorbate results, and downstream stages such as AKMC.
  It is now also the grouping key for per-bulk surface selection in
  `run_adsorbs()`.
- `SurfaceBuilderWorkChain` can store many surfaces per bulk, and
  `AdsorbatesWorkChain` screens up to `settings.MAX_NUM_ADS` per bulk uuid (not
  a single global cap). If higher coverage is needed, raise `MAX_NUM_ADS` in
  `workflows/settings.py`.
- The eta-threshold filter (`eta > eta_threshold` -> skip) is active again; it
  had previously been disabled. Candidates above threshold are not stored and
  do not count toward `self.ctx.candidates`.

## DFT-verification variant

`workchains/adsorbates_dft.py` defines another class also named
`AdsorbatesWorkChain`, but it is not the entry point registered as
`adsorbates`. Its main differences are:

| Area | Registered ML work chain | `adsorbates_dft.py` variant |
|---|---|---|
| Inputs | `chemical_formula`, `reaction`, `reaction_path` | Adds direct `ML_model` input. |
| Surface count | Groups surfaces by bulk uuid, sorts each group by `formation_energy`, and keeps up to `MAX_NUM_ADS` per bulk. | Iterates over all surface rows. |
| Supported reactions in storage map | `OER`, `CO2RR`, `CER`, `NRR`, `NOXRR`, `HER`, `ORR` | `OER`, `CO2RR`, `NOXRR`. |
| Eta filtering | Threshold filter is active (`eta > eta_threshold` -> skip). | Filters candidates above reaction threshold. |
| Extra stages | ML only. | Adds `run_dft()` and `inspect_store_dft()` for VASP/r2SCAN verification candidates. |
| Status on no candidates | Returns `NO_CANDIDATES_WITHIN_ETA_LIMIT`. | Reports a warning and does not return a no-candidate exit code. |

Keep these files mentally separate when making changes. If the DFT variant is
intended to become the registered workflow, update `setup.json`,
`uvsib.egg-info/entry_points.txt`, and this document together.

## Exit codes

| Code | Name | Typical cause |
|---|---|---|
| 300 | `ERROR_CALCULATION_FAILED` | Unknown reaction key, every adsorbate sub-workflow failed, or another unrecoverable calculation failure. |
| 301 | `ERROR_NO_SURFACE_FOUND` | No `DBSurface` rows were found for the formula (renamed from `ERROR_NO_STRUCTURES_FOUND`). |
| 302 | `NO_CANDIDATES_WITHIN_ETA_LIMIT` | No eta values were stored; every computed candidate was above `eta_threshold` (or every set was missing a required intermediate). |

## Practical change notes

- When adding a new reaction, wire it in three places: the adsorbate generator
  in `codes/files/adsorbates.py`, the overpotential calculator import, and the
  `reaction_map` in `workchains/adsorbates.py`.
- Keep the model energy key aligned with `settings.inputs["adsorbates"]["model"]`.
  `store_results_ml()` expects decoded structures to contain
  `"<model_lower>_energy"` in `atoms.info`.
- If the runner starts retrieving `rejected.json` or validation flags through
  the workflow layer, document the new outputs and `job_info` keys here.
- If `SurfaceBuilderWorkChain` changes the slab dictionary schema or stops
  populating `DBSurface.formation_energy`, check the selection logic in
  `run_adsorbs()` and `slab["miller_index"]` usage in `store_results_ml()`.
- If `settings.MAX_NUM_ADS` changes, update the surface-selection and
  parameter sections above.
