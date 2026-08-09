# Changes: `surface_builder.py` and `adsorbates.py`

Comparing commit `9359486` → current working tree (`adsorbates.py` changes are
still uncommitted).

## `workchains/surface_builder.py`

- **Bulk selection now respects the phase-diagram verification flag.**
  `get_struct_uuid()` was reworked: when `_PD_VERIFICATION` is on it queries
  r2SCAN structures as before (minus `MPDB_ref` entries); when it's off it
  instead reads the ML-selected `stable_struct.ml_uuid_list` from
  `DBComposition` and looks each one up by uuid. Also now caps results to
  `MAX_NUM_BULK` and returns a `(structures, from_manifest)` tuple. Replaces
  the old `_SKIP_PD_VERIFICATION` flag/branch.

- **Slab generation is now one CalcJob per bulk structure**, instead of a
  single job batching every bulk together — a pathological bulk fails alone
  rather than taking the whole generation step down.

- **Slab relaxation batching changed from per-bulk to global.** Previously
  each bulk's slabs were chunked and relaxed independently
  (`MAX_SLABS_PER_CHUNK` per bulk). Now all bulks' slabs are pooled into one
  global list (`_global_items`) and packed into chunks of
  `MAX_SLABS_PER_CHUNK` (default 50, was 250) that can mix multiple bulks.
  Each slab carries its own uuid/epa so it can be attributed back to its bulk
  after relaxation (`inspect_relax` regroups by uuid instead of by chunk
  owner). `--epa` is no longer passed as a chunk-level command-line arg since
  a chunk can span bulks.

- **`inspect_slabgen`/`inspect_relax` report per-bulk provenance** (`source`,
  which chunk lost how many slabs on failure) and no longer store raw slab
  lists in the workchain checkpoint — only counts/uuids, with slabs re-read
  from the calc nodes when needed.

- **`add_slab` calls now pass `formation_energy` and `head`** (from
  `settings.inputs['face_build']`) so the stored slab carries its surface
  formation energy, not just the geometry.

- Minor: `MAX_NUM_BULK`, `MAX_NUM_SURF`, `_PD_VERIFICATION` are now read from
  `settings` at module load instead of being local/renamed constants;
  `MAX_SLABS_PER_CHUNK` dropped from 250 to 50.

## `workchains/adsorbates.py`

- **`get_structure_uuid_surface_id` moved into this module** (previously
  imported from `db.utils`) and now queries `DBSurface` directly by
  composition.

- **Per-bulk cap on surfaces considered for adsorbate screening.** `run_adsorbs`
  used to globally sort all surfaces by slab energy and hard-stop after the
  10th overall. It now groups surfaces by bulk `structure_uuid`, sorts each
  group by `formation_energy` (falling back to `inf` if missing), and keeps
  up to `MAX_NUM_ADS` (10) surfaces *per bulk* — so multiple bulk structures
  no longer starve each other of adsorbate candidates.

- **Missing-surface handling moved and reworded.** The `ERROR_NO_STRUCTURES_FOUND`
  early return was removed from `setup()`'s critical path and now happens
  after the "Running Adsorbates WorkChain..." report line, renamed to
  `ERROR_NO_SURFACE_FOUND` ("No stable surface was found...", was "No
  structures were found...") with an added error-level report call.

- **Eta-threshold filtering re-enabled.** A previously commented-out
  `if eta > eta_threshold: continue` guard is now active, so candidates above
  the eta threshold are actually skipped again (this had been disabled with a
  `# TODO: remove?` note).
