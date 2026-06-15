---
title: "Synthesizability classification of generated structures"
author: "uvsib"
---

# What it is

A `MainWorkChain` stage that classifies **every generated structure** of a
composition by synthesizability, using three independent views (the three the
literature splits into). It runs after the phase diagram is built and before the
surface/adsorbate stages, so the 0 K hull it needs already exists.

It is post-processing only — no remote jobs. It reads the generated structures
and their MLIP energies from the database, builds the hull, scores each
structure, and writes the scores back.

# Where it sits

```
MainWorkChain
  pd_ml (generates + relaxes structures, builds hull)
  pd_verification
  --> synthesizability   <-- this stage   (workchains/main.py)
  sqs
  surface_builder
  adsorbates
  nano_generator
```

Gated like the other stages: skipped for the `sqs` and `nano` paths, skipped if
`step_status["synthesizability"] == "Done"`, and globally toggled by
`settings.SYNTH_ENABLED` (`input.yaml: synthesizability.enabled`).

# The three classifiers

All three live in `workchains/_synth_classifiers.py` (pure pymatgen/numpy, unit
tested); the AiiDA wrapper is `workchains/synthesizability.py`.

| View | Question | Signal |
|------|----------|--------|
| **thermo** | Is it thermodynamically accessible? | energy above the 0 K hull vs a metastability window |
| **reaction** | Will it form rather than its competitors? | formation energy from elemental precursors + the decomposition products it would form instead + polymorph selectivity at its composition |
| **pu** | Does it look like things that get made? | a positive-unlabeled learned score (loaded model, or a transparent surrogate) |

## 1. thermo — thermodynamic accessibility

`e_above_hull` from `PhaseDiagram.get_e_above_hull`. Labels: `product`
(on the hull, `<= ehull_tol`), `metastable` (within `meta_window` above), or
`inaccessible`. Score `= exp(-e_above_hull / thermo_scale)`.

The window matters: metals/alloys sit at the tight end of the Sun/Ceder
metastability scale, so `meta_window` defaults to 0.05 eV/atom — and the same
caveat the literature flags applies: polymorph gaps can be below both MLIP and
DFT-functional noise, so near-hull calls are exactly where a DFT check is worth
its cost.

## 2. reaction — precursor → product selectivity

`PhaseDiagram.get_form_energy_per_atom` (reaction energy from the elemental
precursors) and `get_decomp_and_e_above_hull` (the competing phases the
composition decomposes into — the "products" that form instead). Plus a
**polymorph gap**: the energy distance to the best generated structure at the
same composition (0 ⇒ this is the preferred polymorph). Labels:
`selective_product` / `competes` / `outcompeted`.

E.g. an off-hull `Cu2Au` is reported as decomposing into 0.667 Cu3Au + 0.333
CuAu — the actual products, not just a number.

## 3. pu — positive-unlabeled synthesizability

Featurizes the composition (fraction-weighted mean/std of element properties:
Z, electronegativity, group, row, mass, radius, Mendeleev number, + n_elements;
matminer-free). If a pickled model with `predict_proba` is configured
(`pu_model_path`), its probability is used. Otherwise a **transparent surrogate**
— `sigmoid(2 - 18·e_above_hull - 0.4·(n_elements-2))`, flagged
`model: surrogate(ehull+complexity)` — so the stage runs end-to-end without a
trained model. Drop a real PU classifier (SynthNN / CSPML / SynCoTrain-style,
fit on the same featurizer over ICSD-positives) at `pu_model_path` to replace it.

## Combined verdict

Weighted average of the three scores (`weights`, default equal), with a hard
rule: an on-hull, selectively-preferred phase is `synthesizable` regardless.
Labels by `synth_cut` / `maybe_cut`: `synthesizable` / `maybe` / `unlikely`.

# Output

Per structure, the full record (`thermo`, `reaction`, `pu`, `combined`) is
merged into that generated `DBStructureVersion`'s `attributes` JSONB under
`"synthesizability"`. A run-level summary (counts per label + the ranked
synthesizable candidates) is stored on the `DBComposition` row's `attributes`
and emitted as the workchain's `synthesizability` Dict output.

# Configuration (`input.yaml`)

```yaml
synthesizability:
  enabled: true
  ehull_tol: 0.005       # eV/atom; on-hull tolerance
  meta_window: 0.05      # eV/atom; metastable window above the hull
  thermo_scale: 0.04     # eV/atom; accessibility score decay
  complexity_penalty: 0.4  # PU surrogate penalty per element beyond binary
  synth_cut: 0.66        # combined -> "synthesizable"
  maybe_cut: 0.33        # combined -> "maybe"
  # weights: [1.0, 1.0, 1.0]          # thermo, reaction, pu
  # pu_model_path: /path/to/pu.pkl    # trained PU model; omit -> surrogate
```

# Scope note

Per the project scoping decision, the target space is metallic systems
(alloys/intermetallics), not oxides/carbides — which is also where the hull,
the metastability window, and the MLIP energies feeding this stage are most
trustworthy. The honest limit is that all three views are **0 K energetics**:
true synthesis-pathway/selectivity needs finite-T free energies (entropy,
temperature) the potential can supply via phonons/MD but this stage does not yet
compute.

# Finite-temperature extension (phonons → Gibbs hull)

By default the three views score on the **0 K** hull. When
`synthesizability.finite_T.enabled` is set, the stage first computes phonon
free energies and scores on the **Gibbs hull at the synthesis temperature** —
so vibrational entropy can flip which competing phase is the product.

Sub-step (per the user's spec):

```
near-hull candidates + hull vertices
  -> PhononWorkChain  (codes/phonon + codes/files/phonon.py)
       tight relax -> phonopy finite displacements -> MLIP forces
       -> force constants -> harmonic F_vib(T) -> QHA over a few volumes -> G(T)
  -> per-atom free-energy correction  G(T) - E0  at the target T
  -> finite-T (Gibbs) convex hull
  -> classify on the finite-T hull
```

Only structures within `ehull_window` of the 0 K hull (plus the hull vertices)
get phonons — far-above-hull phases won't be rescued by entropy, and phonons are
expensive. Corrections are mapped back by structure UUID; the finite-T hull is
rebuilt from all entries shifted by their `F_vib(T)`.

**Hard requirement — conservative model.** Phonons are second derivatives of the
PES, so forces must be the analytic gradient of one energy. Use **MACE or
MatterSim**; direct-force heads (eqV2/UMA, ORB) give asymmetric force constants
and spurious imaginary modes and must not be used (`finite_T.model`, default
`MACE`). Tight pre-relaxation (`fmax 1e-3`) avoids spurious imaginary modes; QHA
captures the dominant thermal-expansion/Gibbs term without full anharmonic
(TDEP/SCPH) treatment for most metals.

```yaml
synthesizability:
  finite_T:
    enabled: false
    temperature: 1000.0      # K, the synthesis-temperature hull
    model: MACE              # conservative MLIP ONLY
    ehull_window: 0.10       # eV/atom; phonons only for near-hull structures
    displacement: 0.01       # A; do a convergence check per family
    volume_scales: [0.97, 0.99, 1.0, 1.01, 1.03]
```

Two honest limits (literature, 2024–26): (1) **absolute** vibrational free
energies from foundation MLIPs carry a PES-softening bias (~meV/atom) that
fine-tuning roughly halves — trust the stage for *ranking* (where error largely
cancels), DFT-spot-check before quoting absolute entropies; (2) for metals the
**electronic** free energy (Fermi smearing, electron–phonon) is non-negligible
at synthesis T and is **not** in `F_vib(T)` — a future term, not captured here.

# Status

- Classifiers (`_synth_classifiers.py`): implemented, unit-tested on a metallic
  Cu–Au hull (on-hull preferred polymorph → synthesizable; +25 meV/atom worse
  polymorph → maybe; +75 meV/atom → unlikely; off-hull `Cu2Au` → correct
  decomposition products). Finite-T hull tested to flip a ranking
  (a higher-entropy CuAu polymorph overtakes the 0 K winner at 900 K).
- Workchain + DB write-back + `MainWorkChain` wiring + entry points: implemented.
- Phonon runner (`codes/files/phonon.py`) + `codes/phonon/` plugin: implemented;
  the runner end-to-end tested with EMT on fcc Cu (ZPE +0.033 eV/atom at 0 K →
  −0.51 eV/atom at 1200 K, thermal expansion, no imaginary modes).
- Open seams: trained PU model (`pu_model_path`); MLIP phonon fine-tuning for
  quantitative absolute free energies; an electronic free-energy term for metals.
