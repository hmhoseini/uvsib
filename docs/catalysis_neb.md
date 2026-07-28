# Catalysis NEB with explicit solvation — design, fine-tune recipe, machinery

Status 2026-07-22: design settled, **fine-tune frame machinery implemented and
smoke-tested** (gen-0). The production catalysis NEB driver (workchain stage
after adsorbates) comes once the fine-tuned model exists.

## 1. What is being computed — and what is not

The target reaction step is surface hydrogenation from the solvent, e.g.

    *O + H2O(solv) -> *OH + OH(solv)      (an H is deliberately taken from a
                                           solvation water and walked onto *O)

with explicit water above the slab/NP and the shared (CI-)NEB engine
(`codes/files/neb.py`) producing the barrier.

**Framing (important for the numbers):** this is a *chemical* step on a
neutral PES — the MLIP-accessible proxy at the potential of zero charge. The
real electrochemical steps are proton-coupled electron transfers whose
barriers depend on electrode potential and solvent reorganization; MLIPs have
no electrons and no potential. The NEB barriers therefore *complement* the
CHE thermodynamic etas (they never replace potential-dependent kinetics) and
must be reported as such.

Two method-level rules, inherited from the battery NEB driver:

1. **Endpoint discipline.** The final state is built by editing ONE parent
   (move only the transferred H), never by preparing two solvated states
   independently. Otherwise the "barrier" is dominated by spurious solvent
   reorganization (hysteresis), not chemistry.
2. **One snapshot is an anecdote.** Barriers depend on the instantaneous
   H-bond network; sample M solvent snapshots along MD and report the
   barrier *distribution*.

Additionally: H-transfer barriers carry 0.1–0.2 eV ZPE corrections (and real
tunneling below ~300 K). Harmonic ZPE at endpoints + TS from MLIP finite
differences is cheap — wire it into the production driver from the start.

## 2. THE FINE-TUNE RECIPE (crucial — this is where the numbers are won)

> **Foundation heads cannot do this chemistry as-is.** MPtrj/MatPES know bulk
> crystals, oc20 knows adsorbates on dry surfaces, omol knows gas-phase
> molecules. Metal/water interfaces with dissociating O–H bonds are a
> composite domain gap, and the quantities at stake (0.1–0.5 eV barriers,
> ~0.2 eV H-bond energetics) sit at the noise floor of out-of-domain
> foundation models.

1. **Base + labels.** Fine-tune from MACE-MP (medium), multihead or naive
   with replay. Label at **BEEF-vdW** — on-method with the existing CuAuHCO
   round-3 corpus (clusters + fluid boxes), which is the seed pool. What that
   pool lacks and gen-0 adds: slab + water films, co-adsorbed *O/*OH/*OOH +
   water, and H-transfer paths.
2. **Active learning on the NEB paths themselves.** Run the H-transfer NEBs
   with the current model; DFT-single-point the band images (the TS region is
   exactly where the model extrapolates); add, retrain. 2–3 generations
   normally converge the barriers. Non-converged bands are still valid
   training geometries — harvest them too.
3. **Committee UQ.** Train a 3-seed committee; committee spread on TS images
   is the extrapolation alarm (MACE has no native UQ).
4. **Anchors before trust.** Reproduce H2O dissociation on Cu(111)
   (~1.1–1.3 eV, RPBE-class) and proton transfer along a water wire
   (~0.1–0.3 eV) before believing any CuAu number.
5. **Convergence metric.** Barrier drift between generations on the FROZEN
   substrate set (below), plus the gamma-rank guard.

## 3. Active-learning loop WITHOUT the full chain (design decision)

Question: with a gen-1 model in hand, does generating gen-2 DFT inputs need
the complete catalysis chain (bulks, face ordering could change), or just an
adsorbates-level stage?

**Decision: the AL loop runs standalone on a substrate set FROZEN from the
gen-0 production run.** Training data needs *diverse, representative*
coverage of the chemistry, not the argmin faces — MLIP accuracy transfers
across faces of the same chemical system. Re-deriving bulks/faces each
generation churns the substrate under the loop and destroys the convergence
signal (barrier drift would mix model improvement with substrate movement).

Guard for the "face ordering might change" concern: between generations,
re-relax the frozen slab set with the new model (one relax bundle, minutes)
and check the surface-energy rank correlation. Stable → proceed; churning →
catastrophic-forgetting alarm (replay failing), which must be caught anyway.

The **full chain runs exactly once more at the end**, with the converged
model, for self-consistent production numbers.

## 4. Machinery (implemented)

```
codes/files/_solvate.py          pure solvation geometry (seeded, tested)
codes/files/solvation_frames.py  standalone harvester (no AiiDA needed)
db/tables.py::DBFinetuneFrame    frame store (batch/generation/status)
db/ingest_frames.py              harvest output -> DB rows
run_dir/export_all.py            --finetune-frames single-batch DFT export
tests/test_solvate.py            8 geometry + contract tests (EMT)
tests/smoke_solvation_mace.py    GPU physics smoke (MACE-MP, CuAu(111)+*O)
```

Per task (one adsorbed slab; *O located from `ads_coord`):
water film packed at liquid density (seeded rejection sampling, fail-loudly
when the vacuum cannot hold it) → loose pre-relax → Langevin NVT → snapshots
(kind `md_snapshot`) → per snapshot, k nearest intact-water donor H's →
endpoint pair by single-H edit → reactive region free, rest frozen
(`free_radius`, slab anchor) → shared CI-NEB → every band image harvested
(kinds `neb_endpoint`, `neb_image`, with barriers + convergence flags in
`meta`). Every frame echoes the task attribution (surface_id, bulk_uuid,
composition, miller, reaction) — same end-to-end attribution rule as the
slab-relax contract.

**Endpoint-collapse guard** (found live in the first GPU smoke): on the
foundation PES the constructed *OH + OH(solv) product may not be a local
minimum — during endpoint pre-relax the H falls back to its donor water,
both endpoints land in the reactant basin, and the "barrier" reads ~0. The
harvester detects this (transferred-H displacement between the relaxed
endpoints < `collapse_threshold`, default 0.7 A), flags every frame of the
pair with `collapsed: true`, and re-runs the band between the CONSTRUCTED
endpoints (no pre-relax). Those path-scan energies are NOT barriers — but
the TS-region geometries are exactly the frames DFT must label, and whether
the product basin is real is precisely what the fine-tune will decide.

### Workflow gen-0 → DFT

```bash
# 1. harvest (GPU, standalone; substrate from a --with-structures export)
cd <workdir>
python codes/files/solvation_frames.py \
    --from-export CuAu_export.json --composition CuAu --n_surfaces 5 \
    --ML_model MACE --model <path-or-tag> --device cuda
#    (or hand-write input_structures.json: {"params": {...}, "tasks": [...]})

# 2. ingest into the DB
python -m uvsib.db.ingest_frames output.json --batch cuau-gen0 --generation 0

# 3. export the DFT batch (single JSON, structures always embedded)
python run_dir/export_all.py --finetune-frames cuau-gen0 --mark-exported
```

`--mark-exported` flips status new → exported so the same frames cannot be
handed to DFT twice; the labeling stage later sets `labeled`. Generation N+1
repeats 1–3 with the fine-tuned model on the SAME tasks (frozen substrate)
under a new batch name (`cuau-gen1`, `--generation 1`).

### Knobs (defaults in `solvation_frames.DEFAULT_PARAMS`)

film `thickness` 6 A at liquid density, `gap` 2.3 A; MD 300 K, 0.5 fs,
1500 equil + 400 stride, 3 snapshots; 2 pairs/snapshot within 3.5 A;
NEB 5 interior images, fmax 0.08, freeze beyond 6 A of the reactive triplet
plus everything deeper than 4 A below the slab top; cap 80 frames/task
(md_snapshots dropped first, loudly).

## 5. Deferred (production driver, after the fine-tune converges)

- CatalysisNEBWorkChain stage after adsorbates (mirrors BatteryNEBWorkChain):
  bundled pairs over sites x snapshots via the `neb` job_type, results into a
  `db_catalysis_neb` table keyed to the same surface records as the etas.
- ZPE corrections at endpoints/TS; barrier distributions in the export.
- *OOH/*OOH-type steps beyond *O + H -> *OH (same endpoint discipline).
- NP support (same machinery; vacuum box instead of slab cell).
