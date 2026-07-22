# MP-experimental injection (ending the MLIP-energy lottery)

The gen/csp stages used to feed downstream physics (catalysis, battery)
whatever the generators produced and the ML hull kept — for known materials
that is a lottery: the experimentally relevant polymorph can lose to a
generated artifact by MLIP energy error, or never appear at all (olivine vs
maricite NaFePO4 is the canonical case; the paper's own taxonomy flags it).

Three coordinated mechanisms close every gap between "MP knows the structure"
and "the pipeline actually uses it". All are on by default and degrade to a
logged no-op when MP is unreachable.

## 1. Experimental GNoME seeds (`codes/gnome/templates.py`)

`build_template_pool` gains pool #0: `theoretical=False` (ICSD-matched) MP
entries for the target itself — the exact formula in csp mode, the system and
all subsystems in gen mode — with **no ehull cutoff** (the relevant polymorph
is often metastable) and a cap (`exp_cap`, default 10). Highest priority, so
the pool dedup keeps these over analog seeds. Knobs: `exp_seeds` / `exp_cap`
in the `GNoME_generate` / `GNoME_CSP` input.yaml blocks.

This makes SAPS build on the right frameworks — but SAPS *transforms* seeds
(orbit mapping, substitution scoring, per-template caps), so seeding alone
guarantees nothing verbatim. Hence:

## 2. Verbatim injection into the relax bundles (`csp.py` / `gen.py`)

`get_mp_experimental_structures` (codes/utils.py) pulls `theoretical=False`
entries sorted by ehull, cap 10 (`mp_experimental.cap`), and **falls back to
the lowest-ehull theoretical entries when a composition has zero experimental
hits** — disordered experimental phases (cubic LLZO, LTO, P2-Na layers) are
represented in MP by ordered models sometimes flagged theoretical, so
experimental-*only* would lose exactly the solid-state-ionics staples.

The pulled structures ride the existing bundled MLIP relax (bundle order:
generated | injected | references, split back by input index via
`exp_include.split_output_slices` — robust to non-converged drops) and are
stored with DB source `"mp_experimental"`. They are relaxed **on-method**, so
their energies are directly comparable to everything else on the hull; in gen
mode they double as real hull competitors for every chemical subsystem.

## 3. Force-include into `stable_struct` (`phase_diagram.py`)

In-window injected structures compete on merit like any entry. The force
step rescues the rest: stored `mp_experimental` rows are appended to the
manifest even when the ML hull window (`EHULL_ML` = 0.05 eV/atom) would drop
them —

- **deduplicated** against the ML selection *and each other* with
  `exp_include.dedup_forced`, so the manifest never carries the same host
  twice. The dedup matcher is deliberately TIGHTER (pymatgen defaults) than
  the loose selection matcher: a false merge silently re-loses the
  experimental polymorph (verified: the loose tolerances merge rocksalt with
  zincblende; both keep maricite/olivine apart), while a missed merge only
  costs one redundant host;
- **prepended**, because capped consumers (battery `max_hosts`) must see the
  trusted hosts first;
- with their ML `e_above_hull` computed and reported — an
  "OUTSIDE the ML window, MLIP disagrees with experiment" line in the report
  is the visible symptom of a model problem, instead of a silently wrong host;
- recorded in the manifest as `stable_struct["forced_experimental"]`.

## input.yaml

```yaml
mp_experimental:       # optional block, defaults shown
  inject: true
  cap: 10
  force_include: true
GNoME_generate:        # (and GNoME_CSP)
  exp_seeds: true
  exp_cap: 10
```

## Guarantees and non-guarantees

With all three mechanisms, every experimentally-known MP polymorph of a
submitted composition (cap 10) is in the host set downstream stages see,
relaxed on-method, exactly once. What this does NOT fix: structures absent
from MP entirely, disordered phases beyond their ordered MP representatives,
and MLIP *energies* being wrong — the force-include makes the disagreement
visible and survivable, not correct.

Tests: `tests/test_exp_include.py` (pure helpers). Verified live against MP:
NaFePO4 returns maricite + olivine (both experimental, both kept as distinct
hosts), LLZO returns the ordered tetragonal garnet.
