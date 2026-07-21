"""
Deintercalation configuration enumerator for the battery pathway.

Pure pymatgen -- no AiiDA imports, no enum.x binary, no icet requirement --
so it runs in-daemon inside the BatteryWorkChain and is unit-testable
standalone (icet-based enumeration can replace it later without touching the
workchain).

Strategy
--------
1. Build ONE common supercell of the host primitive cell (capped at
   ``max_atoms``), so every composition on the grid is an exact ion count k of
   the N ion sites -- no incommensurate partial occupancies.
2. Grid: ~n_x_steps intermediate counts between the charged (k=0) and
   discharged (k=N) end members.
3. Per intermediate k: candidate vacancy orderings are subsets of the ion
   sites. When oxidation states can be guessed for the host composition the
   candidates are ranked by Ewald electrostatics (standard MP practice; the
   uniform-background term of the charged cell is constant at fixed k, so the
   RANKING is well defined). Metallic hosts, where no charge assignment
   exists, fall back to seeded random sampling. Either way the ranked list is
   deduplicated with StructureMatcher and capped at ``max_configs_per_x``.

Everything is deterministic for a fixed ``seed``.
"""
from __future__ import annotations

import math
import random
import warnings
from itertools import combinations, islice

import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# candidates examined per grid point = OVERSAMPLE * max_configs_per_x
OVERSAMPLE = 6
# hard cap on enumerated/sampled subsets per grid point (before ranking)
SAMPLE_CAP = 2000
# hard cap on Ewald evaluations per grid point (each is a full summation on
# the ~100-atom supercell; ranking a seeded subsample is plenty for a screen)
EWALD_CAP = 400


def _as_structure(struct):
    if isinstance(struct, Structure):
        return struct
    return Structure.from_dict(struct)


def primitive_host(structure, symprec=0.1):
    """Primitive cell of the host (falls back gracefully on low symmetry)."""
    struct = _as_structure(structure)
    try:
        prim = SpacegroupAnalyzer(struct, symprec=symprec) \
            .get_primitive_standard_structure()
        if prim.num_sites <= struct.num_sites:
            return prim
    except Exception:
        pass
    try:
        return struct.get_primitive_structure()
    except Exception:
        return struct


def build_supercell(host, working_ion, max_atoms=128):
    """Common supercell for the whole x grid.

    Starts from the primitive host and greedily doubles the shortest lattice
    direction while the atom count stays within ``max_atoms`` -- keeps the cell
    as cubic-ish as possible (better for Ewald ranking and later NEB reuse).

    Returns (supercell Structure, N ion sites). Raises ValueError if the host
    does not contain the working ion.
    """
    prim = primitive_host(host)
    if not any(site.specie.symbol == working_ion for site in prim):
        raise ValueError(f"host {prim.composition.reduced_formula} contains "
                         f"no {working_ion}")

    mult = [1, 1, 1]
    while True:
        lengths = [prim.lattice.abc[i] * mult[i] for i in range(3)]
        i = int(np.argmin(lengths))
        trial = list(mult)
        trial[i] += 1
        if prim.num_sites * trial[0] * trial[1] * trial[2] > max_atoms:
            break
        mult = trial

    supercell = prim.copy()
    supercell.make_supercell(mult)
    n_sites = sum(1 for site in supercell if site.specie.symbol == working_ion)
    return supercell, n_sites


def x_counts(n_sites, n_x_steps=4):
    """Ion counts to compute: end members + ~n_x_steps intermediates."""
    targets = np.linspace(0, n_sites, n_x_steps + 2)
    return sorted({int(round(t)) for t in targets})


def _guess_site_charges(supercell, working_ion):
    """Per-element oxidation states from the DISCHARGED composition, or None.

    Uses the first (most probable) oxi_state_guesses() solution of the reduced
    composition. Returns a {element_symbol: charge} map; None when no charge-
    balanced assignment exists (metallic hosts -> caller falls back to random
    sampling).
    """
    comp = supercell.composition.reduced_composition
    try:
        guesses = comp.oxi_state_guesses()
    except Exception:
        return None
    if not guesses:
        return None
    charges = dict(guesses[0])
    if charges.get(working_ion, 0) <= 0:
        return None  # a "working ion" that is not a cation makes no sense here
    return charges


def _ewald_energy(structure, charges):
    """Ewald energy with the guessed charges (ranking only).

    The partially deintercalated cell is not charge neutral; EwaldSummation
    then works against a uniform background. That constant shift is identical
    for all orderings at the same ion count, so the ranking is unaffected.
    """
    decorated = structure.copy()
    decorated.add_oxidation_state_by_element(
        {el: charges.get(el, 0.0) for el in
         (sp.symbol for sp in decorated.composition.elements)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EwaldSummation(decorated).total_energy


def _keep_subsets(ion_indices, k, rng):
    """Subsets of ion sites to KEEP (size k): exhaustive when small, else a
    seeded random sample of distinct subsets (both capped at SAMPLE_CAP)."""
    n = len(ion_indices)
    total = math.comb(n, k)
    if total <= SAMPLE_CAP:
        return [frozenset(c) for c in combinations(ion_indices, k)]
    subsets = set()
    while len(subsets) < SAMPLE_CAP:
        subsets.add(frozenset(rng.sample(ion_indices, k)))
    return list(subsets)


def _config_from_subset(supercell, ion_indices, keep):
    """Supercell with the working ion only on the ``keep`` subset of sites."""
    remove = [i for i in ion_indices if i not in keep]
    struct = supercell.copy()
    struct.remove_sites(remove)
    return struct


def enumerate_configs(supercell, working_ion, keep_counts,
                      max_configs_per_x=8, seed=42):
    """Ordered vacancy configurations for every ion count in ``keep_counts``.

    Returns {k: [Structure, ...]} -- end members give exactly one structure,
    intermediates up to ``max_configs_per_x`` distinct ones (Ewald-ranked when
    oxidation states are available, seeded-random otherwise), deduplicated
    with StructureMatcher.
    """
    ion_indices = [i for i, site in enumerate(supercell)
                   if site.specie.symbol == working_ion]
    n_sites = len(ion_indices)
    charges = _guess_site_charges(supercell, working_ion)
    matcher = StructureMatcher(primitive_cell=False, scale=False,
                               attempt_supercell=False)
    rng = random.Random(seed)

    out = {}
    for k in keep_counts:
        if k < 0 or k > n_sites:
            raise ValueError(f"ion count {k} outside [0, {n_sites}]")
        if k in (0, n_sites):
            keep = frozenset(ion_indices) if k else frozenset()
            out[k] = [_config_from_subset(supercell, ion_indices, keep)]
            continue

        subsets = _keep_subsets(ion_indices, k, rng)
        if charges is not None:
            if len(subsets) > EWALD_CAP:
                subsets = rng.sample(subsets, EWALD_CAP)
            ranked = sorted(
                subsets,
                key=lambda s: _ewald_energy(
                    _config_from_subset(supercell, ion_indices, s), charges))
        else:
            rng.shuffle(subsets)
            ranked = subsets

        accepted = []
        budget = OVERSAMPLE * max_configs_per_x
        for subset in islice(ranked, budget):
            candidate = _config_from_subset(supercell, ion_indices, subset)
            if any(matcher.fit(candidate, a) for a in accepted):
                continue
            accepted.append(candidate)
            if len(accepted) >= max_configs_per_x:
                break
        out[k] = accepted
    return out


def enumerate_deintercalation(host, working_ion, n_x_steps=4,
                              max_configs_per_x=8, supercell_max_atoms=128,
                              seed=42):
    """One-call driver used by the BatteryWorkChain.

    Returns a dict:
        supercell  : the common discharged supercell (Structure)
        n_sites    : N working-ion sites in it
        counts     : the ion-count grid (list[int], includes 0 and N)
        configs    : {k: [Structure, ...]}
        ewald_ranked : bool -- True when Ewald ranking was used
    """
    supercell, n_sites = build_supercell(_as_structure(host), working_ion,
                                         supercell_max_atoms)
    counts = x_counts(n_sites, n_x_steps)
    configs = enumerate_configs(supercell, working_ion, counts,
                                max_configs_per_x, seed)
    return {
        "supercell": supercell,
        "n_sites": n_sites,
        "counts": counts,
        "configs": configs,
        "ewald_ranked": _guess_site_charges(supercell, working_ion) is not None,
    }
