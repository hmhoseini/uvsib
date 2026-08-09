# MainWorkChain to PhaseDiagramMLWorkChain flow

This document explains the part of the UvSiB workflow that goes:

```text
MainWorkChain
  -> PhaseDiagramMLWorkChain
       -> MPDBMLWorkChain
       -> CSPWorkChain
       -> GeneratorWorkChain
```

The goal of this part of the workflow is simple:

1. Start from a requested chemical formula, for example `Y2Ru2O7`.
2. Collect or create possible bulk crystal structures for that formula and its chemical subsystems.
3. Relax those structures with the chosen machine-learning interatomic potential.
4. Build an ML-based phase diagram.
5. Store the unique low-energy structures that should be used by later stages, such as surface building and adsorbate screening.

In plain language, this stage answers: "Which bulk structures are worth carrying forward?"

## Source code map

| Workflow | Main file | AiiDA entry point |
|---|---|---|
| `MainWorkChain` | `workchains/main.py` | submitted directly by `MainSubmissionController` |
| `PhaseDiagramMLWorkChain` | `workchains/phase_diagram.py` | `phasediagram` |
| `MPDBMLWorkChain` | `workchains/mpdb_ml.py` | `mpdbml` |
| `CSPWorkChain` | `workchains/csp.py` | `csp` |
| `GeneratorWorkChain` | `workchains/gen.py` | `gen` |
| MatterGen generation/CSP wrappers | `codes/mattergen/workchain.py` | `mattergen.base`, `mattergen.csp` |
| GNoME generation/CSP wrappers | `codes/gnome/workchain.py` | `gnome.base`, `gnome.csp` |
| Minima Hopping wrapper | `codes/minimahopping/workchain.py` | `minimahopping` |
| ML relaxation wrappers | `codes/mace/workchain.py`, `codes/mattersim/workchain.py`, `codes/upet/workchain.py`, `codes/uma/workchain.py` | `mace`, `mattersim`, `upet`, `uma` |

## Big picture

```text
frontend / submitter
  |
  v
MainWorkChain
  |
  | checks DBComposition.step_status["pd_ml"]
  | waits if another identical/shared phase-diagram stage is already running
  |
  v
PhaseDiagramMLWorkChain
  |
  +-- MPDBMLWorkChain
  |     Materials Project / MPDB structures
  |     -> ML relaxation
  |     -> store ML energies as new DBStructureVersion rows
  |
  +-- CSPWorkChain
  |     target formula structures from MatterGen and/or GNoME
  |     -> ML relaxation
  |     -> keep low-energy unique candidates
  |     -> minima hopping from selected candidates
  |     -> store final CSP structures
  |
  +-- GeneratorWorkChain
  |     new structures for missing chemical systems
  |     -> MatterGen and/or GNoME per chemical system
  |     -> ML relaxation
  |     -> keep low-energy unique candidates
  |     -> mark DBChemsys.gen_structures = "Ready"
  |
  v
wait until all needed chemical systems are ready
  |
  v
construct ML phase diagram for requested formula
  |
  v
store DBComposition.stable_struct["ml_uuid_list"]
```

## What MainWorkChain does

`MainWorkChain` is the top-level workflow for a submitted composition/reaction pair. Its inputs are:

| Input | Meaning |
|---|---|
| `chemical_formula` | The reduced composition to process, for example `Y2Ru2O7`. |
| `chemical_systems` | The new chemical systems that still need generated structures, for example `["O", "Ru", "Y", "O-Ru", "O-Y", "Ru-Y", "O-Ru-Y"]` minus systems already present in `DBChemsys`. |
| `reaction` | Reaction family for later surface/adsorbate stages, for example `OER` or `CO2RR`. |
| `reaction_path` | Specific reaction pathway. |
| `nanoparticles` | If it looks like a range such as `2-5`, the workflow enters nanoparticle mode and skips this bulk phase-diagram path. |
| `similarities` | Similarity-analysis options for other stages. Not used by the phase-diagram branch described here. |
| `sqs` | Non-empty dict means this is an SQS request, so normal phase-diagram ML generation is skipped and the SQS branch runs instead. |

During `setup`, `MainWorkChain` reads:

```python
settings.inputs["bulk_relax"]["model"]
```

and stores it as `ml_bulk_model`. This model is the ML potential used for bulk relaxation and phase-diagram energies. Typical registered backend workflow names include `mace`, `mattersim`, `upet`, and `uma`.

### Phase-diagram gate

`MainWorkChain.should_run_pd_ml()` decides whether to run `PhaseDiagramMLWorkChain`.

It skips the phase-diagram ML stage when:

| Condition | Reason |
|---|---|
| `sqs` input is non-empty | SQS requests use the SQS path instead of normal MatterGen/GNoME generation. |
| `nanoparticles` looks like a range | Nanoparticle submissions use the nanoparticle path. |
| `DBComposition.step_status["pd_ml"] == "Done"` | The shared phase-diagram stage already completed for this composition. |

If `DBComposition.step_status["pd_ml"] == "Running"`, the workflow waits by submitting a small `PythonJob` that sleeps for about 360 seconds, then checks the database again. This avoids duplicate work when two reaction pathways for the same composition are submitted at the same time.

When the phase-diagram branch starts, `MainWorkChain` writes:

```text
DBComposition.status = "Running"
DBComposition.step_status["pd_ml"] = "Running"
```

If the child workflow succeeds, it changes the step to `"Done"`. If it fails, it changes the composition status to `"Failed"` and exits with `ERROR_CALCULATION_FAILED`.

## What PhaseDiagramMLWorkChain does

`PhaseDiagramMLWorkChain` owns the bulk-structure discovery and filtering step.
For a more implementation-focused maintainer guide, see
`docs/phase_diagram_ml_workchain.md`.

Inputs:

| Input | Meaning |
|---|---|
| `chemical_formula` | The exact target formula whose stable structures are needed. |
| `chemical_systems` | New chemical systems that need generated structures. These are usually subsystems of the target chemical space that are missing from `DBChemsys`. |
| `ml_bulk_model` | The ML relaxation backend selected by `settings.inputs["bulk_relax"]["model"]`. |

Its stages run in this order:

1. `MPDBMLWorkChain`, if needed.
2. `CSPWorkChain`, unless disabled or already done.
3. `GeneratorWorkChain`, unless disabled or no new chemical systems are needed.
4. Wait until all chemical systems needed for the phase diagram are marked ready.
5. Build the phase diagram and store the low-energy structures for the target formula.

### 1. MPDBMLWorkChain

`MPDBMLWorkChain` uses known structures from the Materials Project / MPDB side of the database.

It is skipped if the database already contains DFT structures for the target formula with source `MPDB_stb` or `MPDB_exp`.

When it runs, it does this:

1. Calls `add_from_mpdb(chemical_formula)`.
   This adds missing MPDB stable structures, MPDB experimental structures, and elemental reference structures.
2. Finds MPDB structures for the requested formula.
3. Finds elemental reference structures that still need an ML relaxation for this `ml_bulk_model`.
4. Sends all of those structures to the selected ML relaxation backend.
5. Converts the output structures and energies into `ComputedStructureEntry` objects.
6. Adds a new `DBStructureVersion` for each relaxed structure, using the ML model name as the method.

The important output is not an AiiDA output port. It is the database update: MPDB-derived structures now have ML-relaxed energies and can be used in the ML phase diagram.

Important parameters:

| Parameter | Where it comes from | Why it matters |
|---|---|---|
| `ml_bulk_model` | `MainWorkChain -> PhaseDiagramMLWorkChain -> MPDBMLWorkChain` | Chooses the ML potential used to relax known structures. All hull energies should ideally use the same method. |
| `bulk_relax.head` | `input.yaml`, read as `settings.inputs["bulk_relax"]["head"]` | Selects the model head/task, for models that support heads. |
| `bulk_relax.fmax` | `input.yaml` | Force convergence threshold. Smaller values are stricter but more expensive. |
| `bulk_relax.max_steps` | `input.yaml` | Maximum geometry-optimization steps. Too small can leave structures poorly relaxed. |
| `configs["models"][ml_bulk_model]` | `config.yaml` | Gives the model file/checkpoint name. |
| `configs["codes"][ml_bulk_model]["code_string"]` | `config.yaml` | AiiDA `Code` used to run the relaxation. |
| `configs["codes"][ml_bulk_model]["job_script"]["device"]` | `config.yaml` | CPU/GPU device passed to the ML code. |
| `EHULL_SCAN` | `workflows/settings.py`, currently `0.1` eV/atom | Used when fetching MPDB structures near the DFT hull. |
| `DFT_FUNC` | `workflows/settings.py`, currently `GGA` | Selects fallback elemental reference energies when ML-relaxed references are missing. |

`MPDBMLWorkChain.final_report()` runs at the end of the current outline. It
reports any DFT-fallback elemental references returned by `get_ref_entries()`,
which is a sign that hull energies may include per-element offset risk.

### 2. CSPWorkChain

`CSPWorkChain` searches for structures with the exact target formula. This is the "crystal structure prediction" path.

It is skipped if:

| Condition | Reason |
|---|---|
| `_SKIP_CSP` is `True` in `workflows/settings.py` | Developer switch to disable CSP. |
| The database already has structures for this composition with source `csp` | CSP results already exist. |

When it runs, it does this:

1. Launches one or more MatterGen CSP jobs if `mattergen.enabled` is true.
2. Launches one or more GNoME CSP jobs if `gnome.enabled` is true.
3. Merges all structures from successful generator jobs.
4. Fails if no structure was produced, or if more than half of the MatterGen CSP jobs failed.
5. Relaxes all CSP structures with `ml_bulk_model`.
6. Uses the ML phase diagram to keep unique low-energy structures.
7. Randomly selects up to `n_mh` low-energy candidates and runs Minima Hopping from them.
8. Filters Minima Hopping results again by ML energy above hull.
9. Stores the combined low-energy CSP and Minima Hopping structures in the database with source `csp` and method `ml_bulk_model`.

Important parameters:

| Parameter | Where it comes from | Why it matters |
|---|---|---|
| `MatterGen_CSP.num_runs` | `input.yaml` | Number of independent MatterGen CSP jobs. More runs increase candidate diversity and cost. |
| `MatterGen_CSP.batch_size` | `input.yaml` | Number of structures sampled per MatterGen batch. |
| `MatterGen_CSP.num_batches` | `input.yaml` | Number of MatterGen batches. Approximate MatterGen candidate count is `batch_size * num_batches * num_runs`. |
| `MatterGen_CSP` model path/code | `config.yaml` | Selects the MatterGen executable and checkpoint for CSP. |
| `mattergen.enabled` | `input.yaml`, default `True` | Enables/disables MatterGen in both CSP and Generator paths. |
| `gnome.enabled` | `input.yaml`, default `False` | Enables/disables GNoME in both CSP and Generator paths. |
| `GNoME_CSP.num_runs` | `input.yaml`, default read as `1` if key is missing | Number of GNoME CSP jobs. |
| `GNoME_CSP.n_max` | `input.yaml` | Maximum number of SAPS candidates before screening. |
| `GNoME_CSP.max_per_template` | `input.yaml` | Candidate cap per seed template. |
| `GNoME_CSP.threshold` | `input.yaml` | Minimum substitution probability for SAPS substitutions. |
| `GNoME_CSP.partial` | `input.yaml` | Maximum number of symmetry-distinct template orbits swapped at once. |
| `GNoME_CSP.keep` | `input.yaml` | Number of candidates kept after optional GNoME screening. |
| `GNoME_CSP.screen` | `input.yaml` | Screening ML model, or `none` to skip screening. |
| `GNoME_CSP.head` | `input.yaml` | Screening model head/task. |
| `GNoME_CSP.k_donors` | `input.yaml`, default `3` inside builder | Number of donor elements used for analog template search. |
| `GNoME_CSP.seed_ehull` | `input.yaml`, default `0.10` | Energy-above-hull cutoff for MP template seeds. |
| `GNoME_CSP.seed_cap` | `input.yaml`, default `60` | Maximum number of seed templates staged to the job. |
| `GNoME_CSP.icet_seeds` | `input.yaml`, default `True` | Whether to add icet-enumerated alloy seeds. |
| `GNoME_CSP.icet_max_size` | `input.yaml`, default `4` | Maximum atom count for icet seed enumeration. |
| `MinimaHopping.num_runs` | `input.yaml` | Maximum number of CSP candidates that will be refined by Minima Hopping. |
| `MinimaHopping.model` | `input.yaml` | ML potential used inside Minima Hopping. It can differ from `bulk_relax.model`. |
| `MinimaHopping.head` | `input.yaml` | Model head/task for Minima Hopping. |
| `MinimaHopping.mh_steps` | `input.yaml` | Length of the Minima Hopping search. Larger values explore more but cost more. |
| `EHULL_ML` | `workflows/settings.py`, currently `0.1` eV/atom | Structures with energy above hull less than or equal to this threshold are treated as low-energy. |

The most scientifically important CSP parameters are `ml_bulk_model`, `EHULL_ML`, `MatterGen_CSP.num_runs`, `MatterGen_CSP.batch_size`, `MatterGen_CSP.num_batches`, `GNoME_CSP.keep`, and `MinimaHopping.num_runs`/`mh_steps`.

### 3. GeneratorWorkChain

`GeneratorWorkChain` creates structures for chemical systems that are needed by the phase diagram but are not already marked ready in the database.

Example: for `Y2Ru2O7`, the full chemical space includes elements, binaries, and the ternary `O-Ru-Y`. If some of those systems are missing from `DBChemsys`, they are passed as `chemical_systems` and generated here.

It is skipped if:

| Condition | Reason |
|---|---|
| `_SKIP_GEN` is `True` | Developer switch to disable generation. The code marks each requested `DBChemsys.gen_structures` as `"Ready"` instead. |
| `chemical_systems` is empty | There are no new chemical systems to generate. |

When it runs, it does this for each chemical system:

1. Launches MatterGen generation if `mattergen.enabled` is true.
2. Launches GNoME generation if `gnome.enabled` is true.
3. Treats each branch as best-effort. A MatterGen failure or GNoME failure is only a warning if the other branch produced structures.
4. Fails the chemical system only if no enabled generator produced structures.
5. Relaxes the merged generated structures with `ml_bulk_model`.
6. Builds a phase diagram for that chemical system and keeps unique low-energy structures.
7. Stores them in the database with source `generated` and method `ml_bulk_model`.
8. Sets `DBChemsys.gen_structures = "Ready"` for that chemical system.

Important parameters:

| Parameter | Where it comes from | Why it matters |
|---|---|---|
| `MatterGen_generate.energy_above_hull` | `input.yaml` | Conditioning value sent to MatterGen. It guides generated structures toward a target stability range. |
| `MatterGen_generate.batch_size` | `input.yaml` | Number of generated structures per batch. |
| `MatterGen_generate.num_batches` | `input.yaml` | Number of batches. Approximate MatterGen candidate count is `batch_size * num_batches` per chemical system. |
| `mattergen.enabled` | `input.yaml`, default `True` | Enables/disables MatterGen generation. |
| `gnome.enabled` | `input.yaml`, default `False` | Enables/disables GNoME generation. |
| `GNoME_generate.n_max` | `input.yaml` | Maximum number of SAPS candidates before screening. |
| `GNoME_generate.max_per_template` | `input.yaml` | Candidate cap per template. |
| `GNoME_generate.threshold` | `input.yaml` | Minimum substitution probability. |
| `GNoME_generate.partial` | `input.yaml` | Number of template orbits that may be partially substituted. |
| `GNoME_generate.keep` | `input.yaml` | Number of candidates kept after screening. |
| `GNoME_generate.screen` | `input.yaml` | Screening ML model, or `none`. |
| `GNoME_generate.head` | `input.yaml` | Screening model head/task. |
| `GNoME_generate.k_donors` | `input.yaml`, default `3` inside builder | Number of donor elements used for analog template search. |
| `GNoME_generate.seed_ehull` | `input.yaml`, default `0.10` | Energy-above-hull cutoff for MP template seeds. |
| `GNoME_generate.seed_cap` | `input.yaml`, default `60` | Maximum number of seed templates. |
| `GNoME_generate.icet_seeds` | `input.yaml`, default `True` | Whether to add icet-enumerated alloy seeds. |
| `GNoME_generate.icet_max_size` | `input.yaml`, default `4` | Maximum atom count for icet seed enumeration. |
| `bulk_relax.model/head/fmax/max_steps` | `input.yaml` | Controls relaxation and energy evaluation of generated structures. |
| `EHULL_ML` | `workflows/settings.py`, currently `0.1` eV/atom | Final low-energy cutoff for generated structures. |

The most important Generator parameters are the generator toggles (`mattergen.enabled`, `gnome.enabled`), the number of generated candidates, the ML relaxation settings, and `EHULL_ML`.

## How the final stable structures are selected

After MPDB, CSP, and Generator stages finish, `PhaseDiagramMLWorkChain` waits for all chemical systems returned by `get_chemical_systems(chemical_formula)` to have:

```text
DBChemsys.gen_structures = "Ready"
```

This wait is performed by a `PythonJob` running `is_data_available(...)`, which checks the database every 60 seconds and times out after 36000 seconds, or 10 hours.

Then `store_stable_structs()` builds the final ML phase diagram:

1. Reads all `DBStructureVersion` rows whose `chemsys` is in the chemical space and whose `method` equals `ml_bulk_model`.
2. Ignores rows with source `MPDB_ref`, because those are elemental references and are handled separately.
3. Keeps only:
   - elemental entries,
   - entries with the exact requested formula.
4. Adds elemental reference entries from `get_ref_entries(...)`.
5. Builds a pymatgen `PhaseDiagram`.
6. Selects unique structures with reduced formula equal to `chemical_formula`.
7. Removes obviously invalid structures with lattice vector components larger than 100 Angstrom.
8. Reduces structures to primitive cells when possible.
9. Removes duplicates using `StructureMatcher`.
10. Keeps structures with `energy_above_hull <= EHULL_ML`.
11. If no structure passes, returns `ERROR_NO_STRUCTURES_FOUND`.
12. Stores selected structure UUIDs in:

```text
DBComposition.stable_struct["ml_uuid_list"]
```

By default, `EHULL_ML = 0.1`, so the workflow stores unique structures within 0.1 eV/atom of the ML convex hull.

## Status and restart behavior

The workflow uses database status fields to avoid repeated work:

| Table/field | Values used here | Meaning |
|---|---|---|
| `DBComposition.status` | `Created`, `Running`, `Failed`, `Done` | Overall composition status. |
| `DBComposition.step_status["pd_ml"]` | `Running`, `Done`, `Failed` | Phase-diagram ML status for the composition. |
| `DBChemsys.gen_structures` | `Ready` or empty | Whether generated structures are available for a chemical system. |

Practical consequences:

- If a phase-diagram ML stage is already `Running`, another `MainWorkChain` for the same composition waits instead of launching duplicate MPDB/CSP/gen jobs.
- If `pd_ml` is already `Done`, later reaction pathways reuse the existing stable bulk structures.
- If CSP or Generator fails for a chemical system and no generated data were stored, `PhaseDiagramMLWorkChain` removes the corresponding `DBChemsys` rows during failure handling. That allows a later retry to recreate them.

## Parameters that most affect scientific results

These parameters change the structures and energies that enter the hull:

| Parameter | Effect |
|---|---|
| `bulk_relax.model` | The main ML potential. Changing it changes all relaxed geometries and energies. |
| `bulk_relax.head` | The model task/head. Important for models trained with multiple heads. |
| `bulk_relax.fmax` | Relaxation convergence. Loose values can leave strained structures and noisy energies. |
| `bulk_relax.max_steps` | Relaxation budget. Too low can cause non-converged structures. |
| `EHULL_ML` | Stability cutoff. Smaller means stricter filtering; larger means more metastable candidates survive. |
| `MatterGen_generate.energy_above_hull` | Directly conditions MatterGen toward structures with the requested stability. |
| MatterGen candidate counts | More candidates improve coverage but cost more GPU time and relaxation time. |
| GNoME candidate/screening settings | Control how many substitution-derived candidates are created and kept. |
| `MinimaHopping.mh_steps` and `MinimaHopping.num_runs` | Control how much local structure search happens after CSP. |
| Elemental references | Missing ML-relaxed elemental references fall back to DFT references, which can introduce per-element energy offsets. |

## Parameters that mostly affect cost and throughput

These mainly change runtime, queue load, or memory use:

| Parameter | Effect |
|---|---|
| `MatterGen_CSP.num_runs` | More independent CSP jobs. |
| `MatterGen_CSP.batch_size` / `num_batches` | More MatterGen samples. |
| `MatterGen_generate.batch_size` / `num_batches` | More generated samples per chemical system. |
| `GNoME_* .n_max` / `keep` / `seed_cap` | More GNoME candidates and larger staged template pools. |
| `configs["codes"][...]["job_script"]` | Scheduler resources: nodes, tasks, CPUs, wall time, exclusive nodes. |
| `is_data_available` timeout | Maximum time to wait for all chemical systems to become ready. |

## Common outcomes

### Fast reuse path

If `DBComposition.step_status["pd_ml"]` is already `"Done"`, `MainWorkChain` does not run the phase-diagram branch. It continues to the later stages using the existing `stable_struct["ml_uuid_list"]`.

### Known MPDB structure exists

If MPDB stable or experimental structures are already present for the target
formula with `method == ml_bulk_model`, `PhaseDiagramMLWorkChain` skips
`MPDBMLWorkChain`. CSP and Generator may still run if needed.

### No new chemical systems

If `chemical_systems` is empty, `GeneratorWorkChain` is skipped. The final phase diagram is built from existing database structures plus any MPDB/CSP data produced in this run.

### Generator branch fails but another branch succeeds

In `GeneratorWorkChain`, MatterGen and GNoME are best-effort. If MatterGen fails but GNoME produces structures, the workflow continues. If both enabled generators produce no structures for a chemical system, the workflow fails.

### No stable structure found

If the final phase diagram contains no unique structure for the requested formula within `EHULL_ML`, the workflow exits with `ERROR_NO_STRUCTURES_FOUND`. This usually means candidate generation was insufficient, the ML model/reference energies are inconsistent, or the target formula is not near the predicted hull.

## Minimal mental model

Think of the workflow as a funnel:

```text
many possible structures
  from MPDB + CSP + generated subsystems
        |
        v
ML relaxation with one chosen bulk model
        |
        v
deduplicate structures
        |
        v
keep structures near the ML convex hull
        |
        v
store UUIDs for later surface and reaction workflows
```

The two most important things to check before trusting results are:

1. All structures and elemental references were evaluated with the same `bulk_relax.model`.
2. The candidate pool was large and diverse enough for the chemistry, especially through MatterGen/GNoME counts and Minima Hopping settings.
