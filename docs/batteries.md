# Battery pathway (deintercalation electrode characteristics)

A bulk pathway that runs after the gen/csp + phase-diagram stages, as an
alternative to the catalysis (surface builder + adsorbates) branch. It takes
the stable host structure(s) containing a working ion A (Li, Na, K, Mg, Zn),
removes A stepwise, and computes electrode characteristics from the energies
E(x) of A_x·Host on a pseudo-binary convex hull.

Submission reuses the existing schema with **no DB migration**:

```python
{"composition": "LiFePO4", "reaction": "battery", "reaction_path": "Li"}
```

`reaction_path` carries the working ion, and the per-ion step status nests in
`DBComposition.step_status["battery"][<ion>]` exactly like the per-(reaction,
pathway) adsorbates status. Surface builder, adsorbates and the catalysis
calculators are skipped for battery submissions (bulk pathway, no slabs).

## What tier 1 computes (MLIP-only, no new calculation plugins)

For each host and working ion:

- **Voltage profile** from the pseudo-binary convex hull over x. Between
  adjacent hull vertices x1 -> x2:

      V = -[E(x2) - E(x1) - (x2 - x1)·mu_A] / (z·(x2 - x1)·e)

  with mu_A = energy/atom of the elemental ion metal (same MLIP, via the
  existing elemental-reference machinery) and z = 1 (Li/Na/K) or 2 (Mg/Zn).
- **Average voltage** over the full range (depends only on the end members).
- **Gravimetric capacity** Q = n·z·F / (3.6·M) in mAh/g, M = molar mass of the
  discharged formula unit (standard cathode convention: LiFePO4 -> 170 mAh/g).
- **Volumetric capacity** (mAh/cm3) from the discharged-state volume.
- **Specific energy** (Wh/kg) = Q_grav x V_avg.
- **Volume change** between charged and discharged end members (cycling proxy).
- **Charged-endpoint stability**: e_above_hull of the empty host against the
  chemsys hull the gen path already built. Far above the hull = conversion-like
  chemistry -> flagged, not silently reported as intercalation.
- **Framework integrity**: StructureMatcher on the host sublattice (ion
  removed) charged vs discharged, plus a volume-collapse guard. Detects layer
  gliding / reconstruction that invalidates the intercalation picture.

All of tier 1 rides on the existing MLIP relax workchains (MACE / UMA /
MatterSim, `job_type: relax`) — zero new AiiDA calculation or parser plugins.

## Method choices locked in

1. **v1 is deintercalation-only.** The submitted formula must contain the
   working ion (LiCoO2, NaFePO4, ...). Insertion into an empty host (TiO2 + Li)
   needs interstitial-site search and is a v2 feature.
2. **One common supercell for all x**, built once from the host primitive cell
   (capped at `supercell_max_atoms`), so every grid point is an exact fraction
   k/N of the N ion sites — no incommensurate occupancies.
3. **Enumeration is pymatgen-only** (no enum.x binary, no icet requirement):
   partial occupancy x on the ion sublattice, orderings ranked by Ewald
   electrostatics (oxidation states from BVAnalyzer / composition guess); if no
   oxidation states can be assigned (metallic hosts), fall back to
   symmetry-distinct sampling with a fixed seed. icet enumeration can replace
   this later without touching the workchain.
4. **Fail loudly.** Framework collapse and hull-unstable charged endpoints are
   recorded as flags on the DB row; a voltage from a structure that fell apart
   is never reported as a clean result.
5. **PBE-head caveat.** MACE omat_pbe is plain PBE: absolute voltages of
   TM-oxide redox couples sit systematically low (roughly 0.5–1 V). Fine for
   *ranking*; publishable absolute voltages need the tier-2 DFT (+U) stage.

## Architecture / file map

| piece | file | notes |
|---|---|---|
| pure calculator | `workchains/batt.py` | hull, voltages, capacities, volume change; **no AiiDA imports**, unit-tested standalone (same family as `oer.py`/`co2rr.py`) |
| enumeration helper | `workchains/battery_enum.py` | supercell choice + per-x orderings; pure pymatgen, unit-tested standalone |
| workchain | `workchains/battery.py` | `BatteryWorkChain`: setup -> enumerate -> bundled MLIP relax fan-out -> analyze -> store |
| DB table | `db/tables.py::DBBatteryPath` | one row per (composition, working_ion, host); results in Postgres, **not** AiiDA storage (ephemeral) |
| entry point | `setup.json` | `battery = uvsib.workchains.battery:BatteryWorkChain` |
| main gating | `workchains/main.py` | `should_run_battery` on `reaction == "battery"`; surface builder + adsorbates skip battery submissions |
| tests | `tests/test_batt.py`, `tests/test_battery_enum.py` | run in any env with pymatgen, no AiiDA / no DB |
| local smoke | `tests/smoke_battery_mace.py` | standalone LFP/LCO end-to-end with a local MACE (not collected by pytest) |

The relax fan-out bundles the working-ion elemental reference through the same
`missing_element_references` / `element_reference_entries` machinery the gen
and csp paths use, so the anode reference (Li bcc, Na bcc, ...) is relaxed on
the same MLIP — methodologically consistent voltages for free.

### input.yaml block (all keys optional; module runs without the block)

```yaml
battery:
  model: MACE            # MLIP for the config relaxations (defaults to bulk_relax)
  head: Default
  fmax: 0.05
  max_steps: 200
  n_x_steps: 4           # intermediate compositions between charged and discharged
  max_configs_per_x: 8   # orderings kept per grid point (Ewald-ranked)
  supercell_max_atoms: 128
  max_hosts: 3           # stable_struct candidates to sweep (<= MAX_NUM_BULK)
```

## Status

- [x] plan agreed (2026-07-21)
- [x] `workchains/batt.py` pure calculator + unit tests
- [x] `workchains/battery_enum.py` enumeration helper + unit tests
- [x] `BatteryWorkChain` + DB table + entry point + main.py gating
- [x] local end-to-end smoke (2026-07-21, MACE-MP-0 medium, RX 6900 XT;
      enumerate -> cell+position relax -> batt.py, ~50 s per material):
      - LiFePO4 (mp-19017): V_avg 3.438 V (exp 3.45 -- MPtrj carries Fe +U),
        Q 169.9 mAh/g (theoretical, exact), 584 Wh/kg, dV -5.2% (exp ~ -7%),
        framework intact; the 3.25-3.60 V staircase around the experimental
        flat plateau is the finite-supercell ordering artifact, as expected
      - LiCoO2 (mp-22526): V_avg 3.277 V (low vs ~3.9 exp -- the documented
        MLIP voltage caveat for Co redox), Q 273.8 mAh/g (theoretical, exact),
        898 Wh/kg, dV +4.6% -- correctly POSITIVE (layered c-axis expansion on
        delithiation, opposite sign to olivine), O3 framework intact
      - determinism confirmed: repeated runs bit-identical energies
- [ ] deploy: `python -m uvsib.db.tables` to create `db_battery_path`, reinstall
      for the new entry point, submit a battery entry
- [ ] tier 2: DFT (+U) verification stage on hull-vertex configurations only
      (analog of pd_verification; battery_fw recipe)
- [ ] tier 2: MLIP-NEB migration barriers (dilute vacancy / ion hop) — **shared
      infrastructure with the planned catalysis NEB**, design them together
- [ ] v2: insertion mode for ion-free hosts (interstitial site search)

## Tier-2 notes (for when we get there)

- **NEB**: one generic MLIP-NEB workchain (images from IDPP interpolation,
  climbing image, existing relax calcjobs extended with a `job_type: neb`)
  serves both battery migration barriers and catalysis reaction barriers —
  do not build two.
- **DFT verification**: statics/short relaxes on the hull vertices only
  (typically 3–6 structures per host), PBE+U per the battery_fw settings;
  MP2020-style corrections only matter if we mix with MP entries — internal
  voltages need consistent U, nothing else.
- **Electrolyte window / interface reactivity** (grand-potential hulls): out of
  scope until someone asks.
