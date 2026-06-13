"""Synthesizability classifiers (pure pymatgen/numpy; no AiiDA/DB imports).

Three independent views on whether a generated structure is synthesizable,
exactly the three the literature splits into:

1. ``thermo_accessibility`` -- thermodynamic accessibility from the 0 K convex
   hull: energy above hull vs a metastability window (Sun/Ceder scale).
2. ``reaction_selectivity`` -- precursor -> product view: formation energy from
   the elemental precursors, the competing decomposition products the phase
   would form instead, and how selectively this phase wins at its composition.
3. ``pu_synthesizability`` -- a learned positive-unlabeled synthesizability
   score. Uses a model loaded from disk when available; otherwise a transparent,
   clearly-flagged surrogate (monotone in hull distance and chemical
   complexity) so the stage runs end-to-end without a trained model.

``classify_entries`` runs all three over a list of generated entries against a
single pymatgen ``PhaseDiagram`` and returns one record per structure with a
combined verdict. Kept free of uvsib/AiiDA imports so it is unit-testable
standalone; the AiiDA wrapper lives in ``synthesizability.py``.
"""

from __future__ import annotations

import math
import pickle
from collections import defaultdict

import numpy as np
from pymatgen.core import Composition, Element

# default thresholds (eV/atom) -- metals/alloys sit on the tight end of the
# Sun/Ceder metastability scale, so the metastable window is deliberately small.
DEFAULTS = {
    "ehull_tol": 0.005,        # at/below this = on the hull (a product)
    "meta_window": 0.05,       # accessible-metastable window above the hull
    "thermo_scale": 0.04,      # eV/atom decay scale for the accessibility score
    "complexity_penalty": 0.4, # PU surrogate penalty per element beyond binary
    "weights": [1.0, 1.0, 1.0],   # thermo, reaction, pu
    "synth_cut": 0.66,         # combined score -> "synthesizable"
    "maybe_cut": 0.33,         # combined score -> "maybe"
}

# composition features for the PU model / surrogate
_ELEMENT_PROPS = ("Z", "X", "group", "row", "atomic_mass", "atomic_radius", "mendeleev_no")


# --------------------------------------------------------------------------- #
# featurization (composition-based, matminer-free)
# --------------------------------------------------------------------------- #
def _prop(el, name):
    try:
        val = getattr(el, name)
    except Exception:
        return None
    if name == "X" and (val is None or (isinstance(val, float) and math.isnan(val))):
        return None
    if name == "atomic_radius" and val is not None:
        return float(val)
    return None if val is None else float(val)


def featurize_composition(comp):
    """Fraction-weighted mean and std of element properties + n_elements.

    Returns an ordered numpy vector; missing properties are mean-imputed to 0
    contribution (the surrogate never depends on these, only a trained model
    would, and it should be fit on the same featurizer).
    """
    comp = Composition(comp)
    fracs = comp.fractional_composition.get_el_amt_dict()
    els = [Element(s) for s in fracs]
    w = np.array([fracs[s] for s in fracs], dtype=float)

    feats = []
    for name in _ELEMENT_PROPS:
        vals = np.array([_prop(el, name) if _prop(el, name) is not None else np.nan
                         for el in els], dtype=float)
        if np.all(np.isnan(vals)):
            feats.extend([0.0, 0.0])
            continue
        mask = ~np.isnan(vals)
        ww = w[mask] / w[mask].sum()
        mean = float(np.dot(ww, vals[mask]))
        var = float(np.dot(ww, (vals[mask] - mean) ** 2))
        feats.extend([mean, math.sqrt(max(var, 0.0))])
    feats.append(float(len(els)))
    return np.array(feats, dtype=float)


def load_pu_model(path):
    """Load a pickled PU model exposing predict_proba; None if unavailable."""
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 1. thermodynamic accessibility
# --------------------------------------------------------------------------- #
def thermo_accessibility(ehull, params):
    """Accessibility from energy-above-hull (eV/atom)."""
    p = params
    if ehull <= p["ehull_tol"]:
        label = "product"          # sits on the hull
    elif ehull <= p["meta_window"]:
        label = "metastable"       # within the synthesizable metastability window
    else:
        label = "inaccessible"
    score = math.exp(-max(ehull, 0.0) / p["thermo_scale"])
    return {"ehull_eV_per_atom": round(float(ehull), 4), "label": label,
            "score": round(float(score), 4)}


# --------------------------------------------------------------------------- #
# 2. precursor -> product reaction selectivity
# --------------------------------------------------------------------------- #
def reaction_selectivity(entry, pd, ehull, comp_min_epa, params):
    """Formation from elemental precursors + competing decomposition products."""
    p = params
    try:
        eform = float(pd.get_form_energy_per_atom(entry))
    except Exception:
        eform = None
    try:
        decomp, _ = pd.get_decomp_and_e_above_hull(entry, allow_negative=True)
        products = sorted(
            ((e.composition.reduced_formula, round(float(amt), 3)) for e, amt in decomp.items()),
            key=lambda x: -x[1])
    except Exception:
        products = []

    # polymorph selectivity: gap (eV/atom) to the best generated structure at
    # the same composition (0 => this IS the preferred polymorph)
    rf = entry.composition.reduced_formula
    epa = entry.energy_per_atom
    poly_gap = float(epa - comp_min_epa.get(rf, epa))

    # this phase is the selective product when it is on/near the hull AND the
    # lowest polymorph at its composition
    if ehull <= p["ehull_tol"] and poly_gap <= p["ehull_tol"]:
        label = "selective_product"
    elif ehull <= p["meta_window"]:
        label = "competes"
    else:
        label = "outcompeted"

    score = math.exp(-max(ehull, 0.0) / p["thermo_scale"]) * math.exp(-max(poly_gap, 0.0) / p["thermo_scale"])
    return {
        "formation_energy_eV_per_atom": None if eform is None else round(eform, 4),
        "decomposition_products": products,
        "polymorph_gap_eV_per_atom": round(poly_gap, 4),
        "label": label,
        "score": round(float(score), 4),
    }


# --------------------------------------------------------------------------- #
# 3. positive-unlabeled synthesizability
# --------------------------------------------------------------------------- #
def _sigmoid(x):
    """Overflow-safe logistic. A far-above-hull structure gives a large negative
    ``x``; the naive ``1/(1+exp(-x))`` then overflows ``exp``. Evaluating the
    branch with the bounded exponent keeps the argument <= 0 (so exp underflows
    to 0 rather than overflowing) and maps such structures to score ~0."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def pu_synthesizability(entry, ehull, model, params):
    """Learned PU score if a model is given, else a transparent surrogate."""
    p = params
    feats = featurize_composition(entry.composition)
    if model is not None:
        try:
            proba = float(model.predict_proba(feats.reshape(1, -1))[0, 1])
            return {"score": round(proba, 4), "model": "loaded",
                    "label": "synthesizable" if proba >= p["synth_cut"]
                    else ("maybe" if proba >= p["maybe_cut"] else "unlikely")}
        except Exception:
            pass
    # surrogate: synthesizability falls off with hull distance and chemical
    # complexity -- a documented stand-in, NOT a trained classifier
    n_el = len(entry.composition.elements)
    x = 2.0 - 18.0 * max(ehull, 0.0) - p["complexity_penalty"] * max(n_el - 2, 0)
    proba = _sigmoid(x)
    return {"score": round(float(proba), 4), "model": "surrogate(ehull+complexity)",
            "label": "synthesizable" if proba >= p["synth_cut"]
            else ("maybe" if proba >= p["maybe_cut"] else "unlikely")}


# --------------------------------------------------------------------------- #
# combine + driver
# --------------------------------------------------------------------------- #
def combine(thermo, reaction, pu, params):
    w = params["weights"]
    scores = [thermo["score"], reaction["score"], pu["score"]]
    combined = float(np.average(scores, weights=w))
    # hard rule: an on-hull, selectively-preferred phase is synthesizable
    if thermo["label"] == "product" and reaction["label"] == "selective_product":
        label = "synthesizable"
    elif combined >= params["synth_cut"]:
        label = "synthesizable"
    elif combined >= params["maybe_cut"]:
        label = "maybe"
    else:
        label = "unlikely"
    return {"score": round(combined, 4), "label": label}


def finite_T_entry(entry, corr_per_atom):
    """A copy of ``entry`` shifted by a per-atom free-energy correction (eV/atom).

    Used to turn a 0 K ComputedEntry into a finite-T one: energy -> energy +
    F_vib(T) * n_atoms, so a hull built from these is the Gibbs hull at T.
    """
    from pymatgen.entries.computed_entries import ComputedEntry
    return ComputedEntry(entry.composition,
                         entry.energy + corr_per_atom * entry.composition.num_atoms)


def classify_entries(generated, pd, params=None, pu_model=None, finite_T=None):
    """Classify generated structures.

    Parameters
    ----------
    generated : list[tuple[str, ComputedEntry]]
        (structure_uuid, 0 K entry) pairs to classify. Entries must lie in
        ``pd``'s chemical space.
    pd : pymatgen PhaseDiagram
        0 K hull (generated + references + elements).
    params : dict | None
        Overrides for ``DEFAULTS``.
    pu_model : object | None
        Loaded PU model with ``predict_proba``; falls back to the surrogate.
    finite_T : dict | None
        When given, score on the finite-T (Gibbs) hull instead of 0 K. Keys:
        ``"T"`` (float, K), ``"pd"`` (PhaseDiagram of finite-T-corrected entries),
        ``"corr"`` (dict struct_uuid -> per-atom free-energy correction, eV/atom).
        Each structure's 0 K e_above_hull is still reported for comparison.
    """
    p = dict(DEFAULTS)
    if params:
        p.update(params)

    T = finite_T["T"] if finite_T else None
    pd_eff = finite_T["pd"] if finite_T else pd
    corr = finite_T.get("corr", {}) if finite_T else {}

    # effective (finite-T corrected) entries, when running at temperature
    eff = []
    for uuid, e0 in generated:
        e_eff = finite_T_entry(e0, corr.get(str(uuid), 0.0)) if finite_T else e0
        eff.append((uuid, e0, e_eff))

    comp_min_epa = defaultdict(lambda: math.inf)
    for _, _, e_eff in eff:
        rf = e_eff.composition.reduced_formula
        comp_min_epa[rf] = min(comp_min_epa[rf], e_eff.energy_per_atom)

    results = []
    for uuid, e0, e_eff in eff:
        try:
            ehull0 = float(pd.get_e_above_hull(e0))
        except Exception:
            results.append({"uuid": str(uuid),
                            "formula": e0.composition.reduced_formula,
                            "error": "outside phase-diagram chemical space"})
            continue

        if finite_T:
            try:
                eff_ehull = float(pd_eff.get_e_above_hull(e_eff))
            except Exception:
                eff_ehull = ehull0
        else:
            eff_ehull = ehull0

        thermo = thermo_accessibility(eff_ehull, p)
        thermo["ehull_0K_eV_per_atom"] = round(float(ehull0), 4)
        if finite_T:
            thermo["T_K"] = T
        reaction = reaction_selectivity(e_eff, pd_eff, eff_ehull, comp_min_epa, p)
        pu = pu_synthesizability(e_eff, eff_ehull, pu_model, p)
        results.append({
            "uuid": str(uuid),
            "formula": e0.composition.reduced_formula,
            "thermo": thermo,
            "reaction": reaction,
            "pu": pu,
            "combined": combine(thermo, reaction, pu, p),
        })
    return results


def summarize(results):
    """Counts per combined label + the synthesizable candidates."""
    counts = defaultdict(int)
    synthesizable = []
    for r in results:
        if "combined" not in r:
            counts["error"] += 1
            continue
        counts[r["combined"]["label"]] += 1
        if r["combined"]["label"] == "synthesizable":
            synthesizable.append({"uuid": r["uuid"], "formula": r["formula"],
                                  "score": r["combined"]["score"]})
    synthesizable.sort(key=lambda x: -x["score"])
    return {"counts": dict(counts), "n_total": len(results),
            "synthesizable": synthesizable}
