"""Symmetry-Aware Partial Substitution (SAPS) — GNoME-style structure generation.

Standalone module (pymatgen only); shipped to the compute node alongside
``gnome_generate.py`` exactly like ``_calculators.py`` is shipped next to the
relaxer scripts.

The method follows the *structural* generation framework of GNoME
(Merchant et al., *Nature* 2023, "Scaling deep learning for materials
discovery"): start from a pool of known template crystals, and propose new
candidates by replacing the species on symmetry-distinct Wyckoff orbits with
chemically plausible target species. Plausibility is scored with the
data-mined ionic-substitution probabilities of Hautier et al. (2011), exposed
in pymatgen as :class:`SubstitutionPredictor`. "Partial" means a *subset* of the
equivalent orbits of one species may be swapped, which is what lets a single
prototype seed many distinct (often off-stoichiometric / solid-solution)
candidates rather than a single full substitution.

Two driving modes:

* ``gen`` — target is a *set of elements* (a chemical system). Templates are
  partially substituted so the product lands inside the target chemical system;
  this yields a spread of compositions, mirroring MatterGen's de-novo branch.
* ``csp`` — target is an explicit *composition*. Only templates whose anonymous
  formula matches the target prototype are used, and every orbit is mapped so
  the product reduces exactly to the requested formula (prototype-substitution
  CSP).

The output is plain pymatgen ``Structure`` objects with oxidation-state
decoration stripped, ready for the same downstream relax/screen path as the
MatterGen output.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

from pymatgen.core import Composition, Species, Structure
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_prediction.substitution_probability import (
    SubstitutionPredictor,
)


# --------------------------------------------------------------------------- #
# substitution probabilities (the data-mined GNoME-style scoring)
# --------------------------------------------------------------------------- #
def predict_donors(target_species, threshold=1e-4, max_donors=12):
    """For each oxidation-decorated *target* species, rank the donor species it
    can plausibly substitute *for*.

    Returns ``{target_species: [(donor_species, probability), ...]}`` sorted by
    descending probability. ``donor`` is the species sitting in a template that
    we would overwrite with ``target``.
    """
    predictor = SubstitutionPredictor(threshold=threshold)
    donors = {}
    for tsp in target_species:
        ranked = []
        try:
            preds = predictor.list_prediction([tsp], to_this_composition=True)
        except Exception:
            preds = []
        for entry in preds:
            # entry['substitutions'] maps {donor_species: host(target)_species}
            for donor, host in entry["substitutions"].items():
                if get_el_sp(host) != tsp:
                    continue
                ranked.append((get_el_sp(donor), float(entry["probability"])))
        # keep the best probability per donor species
        best = {}
        for donor, prob in ranked:
            if donor not in best or prob > best[donor]:
                best[donor] = prob
        donors[tsp] = sorted(best.items(), key=lambda kv: -kv[1])[:max_donors]
    return donors


# --------------------------------------------------------------------------- #
# symmetry analysis
# --------------------------------------------------------------------------- #
def _decorate(structure):
    """Return an oxidation-state-decorated copy, or ``None`` if it can't be guessed."""
    dec = structure.copy()
    try:
        dec.add_oxidation_state_by_guess(max_sites=-80)
    except Exception:
        return None
    if not all(isinstance(sp, Species) for sp in dec.species):
        return None
    return dec


def _orbits(structure):
    """Symmetry-distinct site orbits of a decorated structure.

    Returns a list of ``(species, [site_indices], multiplicity)`` ordered by the
    original site index of the orbit representative, one entry per Wyckoff orbit.
    """
    sga = SpacegroupAnalyzer(structure, symprec=0.05, angle_tolerance=5)
    sym = sga.get_symmetrized_structure()
    orbits = []
    for group in sym.equivalent_indices:
        rep = structure[group[0]]
        orbits.append((rep.specie, list(group), len(group)))
    orbits.sort(key=lambda o: o[1][0])
    return orbits


# --------------------------------------------------------------------------- #
# substitution application
# --------------------------------------------------------------------------- #
def _apply(structure, replacements):
    """Return a new, undecorated structure with ``replacements`` applied.

    ``replacements`` maps site index -> target :class:`Species`. Sites not listed
    keep their (decorated) species. Oxidation state is stripped on the way out so
    the result matches the plain MatterGen structures.
    """
    new = structure.copy()
    for idx, target in replacements.items():
        new[idx] = target
    new.remove_oxidation_states()
    return new


def _same_anion_or_cation(donor, target):
    """Keep substitutions chemistry-consistent: anion->anion, cation->cation."""
    d = donor.oxi_state if hasattr(donor, "oxi_state") else 0
    t = target.oxi_state if hasattr(target, "oxi_state") else 0
    if d == 0 or t == 0:
        return True
    return (d > 0) == (t > 0)


# --------------------------------------------------------------------------- #
# mode: gen  (target = element set; partial substitutions allowed)
# --------------------------------------------------------------------------- #
def _best_target(specie, donor_lookup, target_elements):
    """Best target species this template ``specie`` can become, or ``None``.

    Charge-preserving substitutions are favoured (they keep the cell neutral);
    same-sign substitutions are allowed at a penalty. Returns
    ``(target_species, score)`` or ``None``.
    """
    best = None
    for tsp, prob in donor_lookup.get(specie, []):
        if tsp.element.symbol not in target_elements:
            continue
        if not _same_anion_or_cation(specie, tsp):
            continue
        charge_ok = getattr(specie, "oxi_state", 0) == getattr(tsp, "oxi_state", 0)
        score = prob * (2.0 if charge_ok else 1.0)
        if best is None or score > best[1]:
            best = (tsp, score)
    return best


def _generate_gen(template, donors, target_elements, max_per_template, partial):
    """Partial-substitution candidates whose elements all lie in the target set."""
    orbits = _orbits(template)
    donor_lookup = {}  # donor species -> list of (target species, prob)
    for tsp, dlist in donors.items():
        for dsp, prob in dlist:
            donor_lookup.setdefault(dsp, []).append((tsp, prob))

    mandatory = []   # orbits whose element is NOT yet in the target system
    optional = []    # orbits already in-system that could still be swapped
    for specie, idxs, mult in orbits:
        sub = _best_target(specie, donor_lookup, target_elements)
        in_system = specie.element.symbol in target_elements
        if not in_system:
            if sub is None:
                return []  # template can't be brought into the target system
            mandatory.append((idxs, sub))
        elif sub is not None and sub[0].element.symbol != specie.element.symbol:
            optional.append((idxs, sub))

    # base candidate: substitute every mandatory orbit to its best target
    base_repl = {}
    base_score = 1.0
    for idxs, (tsp, score) in mandatory:
        for s in idxs:
            base_repl[s] = tsp
        base_score *= score

    candidates = []

    def _emit(repl, score):
        new = _apply(template, repl)
        if set(el.symbol for el in new.composition.elements) <= target_elements:
            candidates.append((new, score))

    _emit(dict(base_repl), base_score)
    # partial variants: additionally swap subsets of the optional in-system orbits
    for k in range(1, min(partial, len(optional)) + 1):
        for combo in itertools.combinations(optional, k):
            repl = dict(base_repl)
            score = base_score
            for idxs, (tsp, sc) in combo:
                for s in idxs:
                    repl[s] = tsp
                score *= sc
            _emit(repl, score)
            if len(candidates) >= max_per_template * 4:
                break
        if len(candidates) >= max_per_template * 4:
            break

    candidates.sort(key=lambda c: -c[1])
    return candidates[:max_per_template]


# --------------------------------------------------------------------------- #
# mode: csp  (target = composition; prototype mapping -> exact stoichiometry)
# --------------------------------------------------------------------------- #
def _reduce_counts(counts):
    """Divide an iterable of integer multiplicities by their gcd."""
    from math import gcd
    counts = [int(round(c)) for c in counts]
    g = 0
    for c in counts:
        g = gcd(g, c)
    g = g or 1
    return [c // g for c in counts]


def _generate_csp(template, target_comp, target_species, donors, max_per_template):
    """Map a prototype's orbits onto the target composition (exact formula)."""
    if (Composition(template.composition.reduced_formula).anonymized_formula
            != target_comp.anonymized_formula):
        return []

    orbits = _orbits(template)

    # prototype: total multiplicity per (decorated) species
    proto_by_specie = defaultdict(int)
    proto_orbits = defaultdict(list)
    for specie, idxs, mult in orbits:
        proto_by_specie[specie] += mult
        proto_orbits[specie].append(idxs)

    proto_species = list(proto_by_specie)
    proto_reduced = dict(zip(proto_species,
                             _reduce_counts([proto_by_specie[s] for s in proto_species])))

    # target: reduced per-formula multiplicity per target species
    tsp_list = list(target_species)
    tgt_counts = [target_comp[_target_amount_key(target_comp, s)] for s in tsp_list]
    tgt_reduced = dict(zip(tsp_list, _reduce_counts(tgt_counts)))

    if sorted(proto_reduced.values()) != sorted(tgt_reduced.values()):
        return []

    candidates = []
    for perm in itertools.permutations(tsp_list):
        mapping = {}
        ok = True
        score = 1.0
        for psp, tsp in zip(proto_species, perm):
            if proto_reduced[psp] != tgt_reduced[tsp]:
                ok = False
                break
            if not _same_anion_or_cation(psp, tsp):
                ok = False
                break
            charge_ok = getattr(psp, "oxi_state", 0) == getattr(tsp, "oxi_state", 0)
            prob = _lookup_prob(donors, tsp, psp) * (2.0 if charge_ok else 1.0)
            score *= max(prob, 1e-6)
            mapping[psp] = tsp
        if not ok:
            continue
        replacements = {}
        for psp, idx_lists in proto_orbits.items():
            for idxs in idx_lists:
                for s in idxs:
                    replacements[s] = mapping[psp]
        new = _apply(template, replacements)
        if new.composition.reduced_formula != target_comp.reduced_formula:
            continue
        candidates.append((new, score))

    candidates.sort(key=lambda c: -c[1])
    return candidates[:max_per_template]


def _target_amount_key(comp, species):
    """Amount lookup that works whether the composition is decorated or not."""
    if species in comp:
        return species
    return species.element


def _as_species(sp):
    return sp if isinstance(sp, Species) else get_el_sp(sp)


def _lookup_prob(donors, target_species, donor_species):
    for tsp, dlist in donors.items():
        if tsp.symbol != getattr(target_species, "symbol", None):
            continue
        for dsp, prob in dlist:
            if dsp == donor_species:
                return prob
    return 1e-6


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def saps_generate(templates, target, mode, *, n_max=200, max_per_template=8,
                  threshold=1e-4, partial=2):
    """Generate candidate structures by SAPS.

    Parameters
    ----------
    templates : list[dict | Structure]
        Pool of template crystals (pymatgen dicts or Structures).
    target : str
        For ``mode='gen'`` a chemical system ``"Y-Ru-O"``; for ``mode='csp'`` a
        chemical formula ``"Y2Ru2O7"``.
    mode : {'gen', 'csp'}
    n_max : int
        Hard cap on returned candidates (kept by descending substitution score).
    max_per_template : int
        Cap on candidates produced from any single template.
    threshold : float
        Minimum data-mined substitution probability to consider a donor.
    partial : int
        Max number of orbits substituted simultaneously in ``gen`` mode.

    Returns
    -------
    list[Structure]
        Undecorated candidate structures, best-scored first.
    """
    if mode == "gen":
        target_elements = set(target.split("-"))
        target_species = _target_species_for_elements(target_elements)
    elif mode == "csp":
        target_comp = Composition(target)
        target_species = _target_species_for_composition(target_comp)
    else:
        raise ValueError(f"unknown SAPS mode '{mode}'")

    donors = predict_donors(target_species, threshold=threshold)

    scored = []
    for tmpl in templates:
        structure = tmpl if isinstance(tmpl, Structure) else Structure.from_dict(tmpl)
        decorated = _decorate(structure)
        if decorated is None:
            continue
        try:
            if mode == "gen":
                scored.extend(_generate_gen(
                    decorated, donors, target_elements,
                    max_per_template, partial))
            else:
                scored.extend(_generate_csp(
                    decorated, target_comp, target_species, donors,
                    max_per_template))
        except Exception:
            continue

    scored.sort(key=lambda c: -c[1])
    return [s for s, _ in scored[:n_max]]


def _target_species_for_elements(elements):
    """All common oxidation-decorated species for each target element.

    Keeping every common oxidation state (rather than a single guess) is what
    lets a template ``Ti4+`` site be recognised as a donor for ``Ru4+`` while a
    ``Na+`` site is recognised as a donor for ``Y3+`` etc.
    """
    species = []
    for el in elements:
        e = get_el_sp(el)
        states = e.common_oxidation_states or e.oxidation_states or (0,)
        for oxi in states:
            species.append(Species(el, oxi))
    return species


def _target_species_for_composition(comp):
    decorated = None
    try:
        decorated = comp.add_charges_from_oxi_state_guesses()
    except Exception:
        decorated = None
    if decorated is not None and all(isinstance(s, Species) for s in decorated.elements):
        return list(decorated.elements)
    return [_guess_species(el.symbol) for el in comp.elements]


def _guess_species(symbol):
    """Best single oxidation state for an element from its common states."""
    el = get_el_sp(symbol)
    try:
        common = el.common_oxidation_states or el.oxidation_states
        if common:
            # prefer the state with largest magnitude that is most typical:
            # anions negative, else the smallest positive (most ionic/common)
            pos = [o for o in common if o > 0]
            neg = [o for o in common if o < 0]
            if neg and not pos:
                oxi = max(neg, key=lambda o: abs(o))
            else:
                oxi = min(pos) if pos else common[0]
            return Species(symbol, oxi)
    except Exception:
        pass
    return Species(symbol, 0)
