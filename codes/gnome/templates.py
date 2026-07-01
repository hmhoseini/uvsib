"""Build the SAPS template (seed) pool for the GNoME generator -- local side.

Runs where ``uvsib.settings`` and the Materials Project API key are available
(inside the workchain ``setup``), NOT on the compute node. Produces a list of
pymatgen structure dicts that ``GNoMECalculation`` stages as ``templates.json``.

GNoME-style seeding: good templates are known crystals that are one ionic
substitution away from the target. For each target species we look up the donor
species it plausibly substitutes *for* (the same data-mined probability table
SAPS scores with), pull stable structures for the resulting analog chemistries
from MP, and add the target chemistry itself. If MP is unreachable, fall back to
the bundled r2SCAN entries so the branch still has something to chew on.
"""

import os
import sys
import json
from itertools import combinations

from pymatgen.core import Composition, Element, Species, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.structure_prediction.substitution_probability import (
    SubstitutionPredictor,
)

from uvsib.workflows import settings


def _target_species(symbol):
    el = Element(symbol)
    states = el.common_oxidation_states or el.oxidation_states or (0,)
    return [Species(symbol, oxi) for oxi in states]


def _donor_elements(symbol, k=3, threshold=1e-3):
    """Top-k distinct donor elements the target ``symbol`` can substitute for."""
    predictor = SubstitutionPredictor(threshold=threshold)
    ranked = []
    for tsp in _target_species(symbol):
        try:
            preds = predictor.list_prediction([tsp], to_this_composition=True)
        except Exception:
            continue
        for entry in preds:
            for donor in entry["substitutions"]:
                ranked.append((donor.element.symbol, float(entry["probability"])))
    best = {}
    for el, prob in ranked:
        if el == symbol:
            continue
        best[el] = max(best.get(el, 0.0), prob)
    return [el for el, _ in sorted(best.items(), key=lambda kv: -kv[1])[:k]]


def _analog_chemsys(elements, k):
    """Target system plus systems reachable by one donor->target element swap."""
    systems = {"-".join(sorted(elements))}
    for i, el in enumerate(elements):
        for donor in _donor_elements(el, k=k):
            analog = list(elements)
            analog[i] = donor
            if len(set(analog)) == len(analog):
                systems.add("-".join(sorted(analog)))
    return systems


def _analog_formulas(comp, k):
    """Target formula plus formulas reachable by one donor->target element swap."""
    formulas = {comp.reduced_formula}
    amounts = comp.get_el_amt_dict()
    elements = list(amounts)
    for el in elements:
        for donor in _donor_elements(el, k=k):
            if donor in amounts:
                continue
            new_amts = dict(amounts)
            new_amts[donor] = new_amts.pop(el)
            formulas.add(Composition(new_amts).reduced_formula)
    return formulas


def _mp_by_chemsys(systems, ehull, cap):
    from mp_api.client import MPRester
    structures = []
    with MPRester(settings.api_key) as mpr:
        for chemsys in systems:
            try:
                docs = mpr.materials.summary.search(
                    chemsys=chemsys, energy_above_hull=(0, ehull),
                    fields=["structure", "energy_above_hull"])
            except Exception:
                continue
            for doc in docs:
                structures.append((doc.energy_above_hull, doc.structure.as_dict()))
    structures.sort(key=lambda x: x[0])
    return [s for _, s in structures[:cap]]


def _mp_by_formula(formulas, ehull, cap):
    from mp_api.client import MPRester
    structures = []
    with MPRester(settings.api_key) as mpr:
        for formula in formulas:
            try:
                docs = mpr.materials.summary.search(
                    formula=formula, energy_above_hull=(0, ehull),
                    fields=["structure", "energy_above_hull"])
            except Exception:
                continue
            for doc in docs:
                structures.append((doc.energy_above_hull, doc.structure.as_dict()))
    structures.sort(key=lambda x: x[0])
    return [s for _, s in structures[:cap]]


# --------------------------------------------------------------------------- #
# same chemical space (target system + all subsystems) from MP
# --------------------------------------------------------------------------- #
def _subsystems(elements):
    """All chemical sub-systems of the element set (elements, binaries, ...)."""
    els = sorted(set(elements))
    systems = set()
    for n in range(1, len(els) + 1):
        for combo in combinations(els, n):
            systems.add("-".join(combo))
    return systems


def _mp_same_chemspace(elements, ehull, cap):
    """Real structures across the *whole* target chemical space (the target
    system and every subsystem: elements, binaries, ...) from MP -- the actual
    alloys/compounds, not just one-substitution analogs."""
    return _mp_by_chemsys(_subsystems(elements), ehull, cap)


# --------------------------------------------------------------------------- #
# icet-enumerated ordered alloys for the formula / chemical space
# --------------------------------------------------------------------------- #
_NOMINAL_A = {"fcc": 3.8, "bcc": 3.2, "sc": 3.0}


def _icet_alloy_seeds(elements, target_comp, max_size, lattices, cap):
    """ICET-enumerated symmetry-inequivalent ordered alloys spanning the target
    chemical space.

    Decorates common metallic parent lattices (fcc/bcc) with the system's
    elements and enumerates all inequivalent superstructures up to ``max_size``
    atoms (Hart & Forcade). For csp (``target_comp`` given) only decorations that
    reduce to the target formula are kept (e.g. the L1_2/L1_0 Cu-Au orderings for
    Cu3Au); for gen all decorations in the space are kept. Returns ``[]`` if
    icet/ase are unavailable so the branch degrades to MP-only seeds.
    """
    if len(set(elements)) < 2:
        return []  # nothing to alloy
    try:
        from ase.build import bulk
        from pymatgen.io.ase import AseAtomsAdaptor
        try:
            from icet.tools import enumerate_structures
        except Exception:
            from icet.tools.structure_enumeration import enumerate_structures
    except Exception:
        sys.stderr.write("gnome-templates: icet/ase unavailable, skipping icet seeds\n")
        return []

    species = list(dict.fromkeys(elements))           # unique, order-preserving
    target_rf = target_comp.reduced_formula if target_comp is not None else None
    if target_comp is not None:                       # ensure the cell can reach the target
        n_target = int(round(sum(target_comp.reduced_composition.values())))
        max_size = max(max_size, n_target)
    max_size = min(max_size, 8)                        # bound enumeration cost

    adaptor = AseAtomsAdaptor()
    seeds, seen = [], set()
    for lat in lattices:
        try:
            prim = bulk(species[0], lat, a=_NOMINAL_A.get(lat, 3.6))
        except Exception:
            continue
        try:
            for atoms in enumerate_structures(prim, range(1, max_size + 1), species):
                try:
                    s = adaptor.get_structure(atoms)
                except Exception:
                    continue
                rf = s.composition.reduced_formula
                if target_rf is not None and rf != target_rf:
                    continue
                key = (lat, rf, len(s))
                if key in seen:
                    continue
                seen.add(key)
                seeds.append(s.as_dict())
                if len(seeds) >= cap:
                    return seeds
        except Exception:
            continue
    return seeds


def _dedup_structures(struct_dicts):
    """Drop obvious duplicates (same reduced formula, atom count, volume/atom)."""
    seen, out = set(), []
    for sd in struct_dicts:
        try:
            s = Structure.from_dict(sd)
            key = (s.composition.reduced_formula, len(s),
                   round(s.lattice.volume / max(len(s), 1), 1))
        except Exception:
            out.append(sd)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(sd)
    return out


def _bundled_fallback(mode, target):
    """Offline seeds from the bundled r2SCAN entries."""
    path = os.path.join(settings.files_path, "r2scan_entries.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception:
        return []

    out = []
    if mode == "gen":
        want = set(target.split("-"))
        for entry in entries:
            try:
                cse = ComputedStructureEntry.from_dict(entry["entries"]["r2SCAN"])
            except Exception:
                continue
            if set(el.symbol for el in cse.composition.elements) & want:
                out.append(cse.structure.as_dict())
    else:
        anon = Composition(target).anonymized_formula
        for entry in entries:
            try:
                cse = ComputedStructureEntry.from_dict(entry["entries"]["r2SCAN"])
            except Exception:
                continue
            if cse.composition.anonymized_formula == anon:
                out.append(cse.structure.as_dict())
    return out


def build_template_pool(target, mode, *, k_donors=3, ehull=0.10, cap=60,
                        use_icet=True, icet_max_size=4, icet_lattices=("fcc", "bcc")):
    """Return a list of pymatgen structure dicts to seed SAPS.

    Seeds are gathered from three complementary sources and merged (priority
    order: most relevant first), de-duplicated, then capped:

    1. **same chemical space** -- real structures for the target system *and all
       its subsystems* (elements, binaries, ...) from MP;
    2. **icet alloys** -- symmetry-inequivalent ordered decorations of metallic
       parent lattices for the target formula / chemical space (``use_icet``);
    3. **analog chemistries** -- structures one ionic substitution away (the
       original GNoME-style donor seeding).

    Parameters
    ----------
    target : str
        Chemical system ``"Y-Ru-O"`` for ``gen``; formula ``"Y2Ru2O7"`` for ``csp``.
    mode : {"gen", "csp"}
    k_donors : int
        Donor elements considered per target element when forming analog seeds.
    ehull : float
        Max MP energy-above-hull for an MP seed to be pulled.
    cap : int
        Hard cap on the total number of seeds.
    use_icet : bool
        Include icet-enumerated ordered alloys (skipped silently if icet absent).
    icet_max_size : int
        Max atoms per icet-enumerated cell (raised to reach a csp target formula).
    icet_lattices : tuple[str]
        Parent lattices to enumerate decorations on.
    """
    if mode == "gen":
        elements = sorted(set(target.split("-")))
        target_comp = None
    else:
        target_comp = Composition(target)
        elements = sorted({el.symbol for el in target_comp.elements})

    pools = []
    # 1. same chemical space from MP (target system + subsystems)
    try:
        pools.append(_mp_same_chemspace(elements, ehull, cap))
    except Exception:
        pass
    # 2. icet-enumerated ordered alloys for the formula / space
    if use_icet:
        try:
            pools.append(_icet_alloy_seeds(elements, target_comp, icet_max_size,
                                           icet_lattices, cap))
        except Exception:
            pass
    # 3. analog chemistries (one substitution away) -- original donor seeding
    try:
        if mode == "gen":
            pools.append(_mp_by_chemsys(_analog_chemsys(target.split("-"), k_donors), ehull, cap))
        else:
            pools.append(_mp_by_formula(_analog_formulas(target_comp, k_donors), ehull, cap))
    except Exception:
        pass

    merged = _dedup_structures([s for pool in pools for s in pool])
    if not merged:
        merged = _bundled_fallback(mode, target)
    return merged[:cap]
