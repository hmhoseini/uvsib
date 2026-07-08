"""Convex-hull + surface-energy + O-vacancy analysis for SQS output.

Consumes the structures + metadata that ``codes/files/sqs.py`` writes -- each
pymatgen Structure dict carries ``properties["predicted_energy"]`` (eV, MLIP
relaxed) and the parallel ``metadata`` list carries ``kind``,
``parent_label``, ``composition`` (cation sublattice fractions),
``n_parent_atoms``, ``natoms``, ``miller``, ``vacancy_frac``, ``sqs_seed``.

Public API used by ``SQSWorkChain.create_hull``:

    split_by_kind(structures, metadata) -> (bulk, slab, slab_vac)
    analyse_bulk(bulk_items, functional) -> [bulk_result, ...]
    analyse_surface(slab_items, bulk_results) -> [slab_result, ...]
    analyse_ovac(slab_vac_items, slab_results, mu_O2) -> [vac_result, ...]

All four functions are pure (no AiiDA imports here) -- pymatgen +
``uvsib.workchains.utils`` (which itself only uses load_profile transitively).

Caveats worth knowing:

* The bulk hull uses elemental references from ``codes/files/r2scan_entries.json``
  / ``gga_ggau_entries.json`` (selected by ``functional``). If the SQS run
  contains elements absent from those reference files, formation energies and
  e_above_hull come out as NaN; raw ``energy_per_atom`` is still reported.

* Surface energy gamma is reduced to ``(E_slab - N * mu_bulk_per_atom) / (2 A)``
  -- valid only when the slab is a stoichiometric multiple of the bulk
  reference at the same cation composition. Non-stoichiometric terminations
  need a full Sun-Ceder chemical-potential frame (a TODO).

* O-vacancy formation energy uses the bulk-reference convention
  ``dE_vac = (E_vac + 0.5 n_vac * mu_O2 - E_pristine) / n_vac``. Pass mu_O2 in
  eV per O2 *molecule*. If mu_O2 is None, the unreferenced
  ``(E_vac - E_pristine) / n_vac`` is reported and ``reference`` is set
  accordingly so downstream code can spot it.
"""
import numpy as np
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram

from uvsib.workchains.utils import matcher, get_primitive_cell
from uvsib.codes.utils import get_element_entries


# ---------------------------------------------------------------------------
# bucketing + entry construction
# ---------------------------------------------------------------------------

def _comp_key(comp):
    """Canonical hashable key for the metadata composition block.

    ``comp`` is either ``{label: {sp: frac}}`` (multi-sublattice) or
    ``{sp: frac}`` (single-sublattice). Sort keys at every level so equivalent
    composition points hash equal even if the caller built them in different
    orders.
    """
    if not comp:
        return "pure"
    if all(isinstance(v, (int, float)) for v in comp.values()):
        return ";".join(f"{k}:{v:.4f}" for k, v in sorted(comp.items()))
    return "|".join(
        f"{lab}=" + ";".join(f"{k}:{v:.4f}" for k, v in sorted(d.items()))
        for lab, d in sorted(comp.items())
    )


def split_by_kind(structures, metadata):
    bulk, slab, slab_vac = [], [], []
    for s, m in zip(structures, metadata):
        k = m.get("kind")
        if k == "bulk_sqs":
            bulk.append((s, m))
        elif k == "slab":
            slab.append((s, m))
        elif k == "slab_Ovac":
            slab_vac.append((s, m))
    return bulk, slab, slab_vac


def _entry(struct_dict, meta):
    struct = Structure.from_dict(struct_dict)
    energy = struct.properties.get("predicted_energy")
    if energy is None:
        return None
    data = dict(meta)
    data["comp_key"] = _comp_key(meta.get("composition", {}))
    return ComputedStructureEntry(
        composition=struct.composition,
        structure=struct,
        energy=float(energy),
        data=data,
    )


def _make_entries(items):
    return [e for e in (_entry(s, m) for s, m in items) if e is not None]


def _dedup(entries, bucket_key):
    """Within each bucket, drop StructureMatcher-equivalent duplicates.

    Sorted by per-atom energy ascending so the lowest survives. The matcher
    config matches ``workchains.utils.matcher`` (primitive_cell=True).
    """
    buckets = {}
    for e in entries:
        buckets.setdefault(bucket_key(e), []).append(e)
    kept = []
    for ents in buckets.values():
        ents.sort(key=lambda e: e.energy / max(1, len(e.structure)))
        prims = []
        for e in ents:
            try:
                prim = get_primitive_cell(e.structure.as_dict())
            except Exception:
                prim = e.structure
            if any(matcher.fit(prim, p) for p in prims):
                continue
            prims.append(prim)
            kept.append(e)
    return kept


# ---------------------------------------------------------------------------
# bulk hull
# ---------------------------------------------------------------------------

def analyse_bulk(bulk_items, functional, element_entries=None):
    """``element_entries``: on-method elemental hull endpoints (from
    ``element_reference_entries``). Without them PhaseDiagram lacks terminal
    entries for any multi-element space, the try/except below eats the error
    and every ehull/eform comes out NaN -- pass them for real hull flags."""
    entries = _dedup(
        _make_entries(bulk_items),
        bucket_key=lambda e: (e.data["parent_label"], e.data["comp_key"]),
    )

    by_parent = {}
    for e in entries:
        by_parent.setdefault(e.data["parent_label"], []).append(e)

    results = []
    for parent, ents in by_parent.items():
        ref = list(element_entries) if element_entries else []
        try:
            pd = PhaseDiagram(ents + ref)
        except Exception:
            pd = None
        for e in ents:
            if pd is None:
                ehull, eform = float("nan"), float("nan")
            else:
                try:
                    ehull = pd.get_e_above_hull(e)
                    eform = pd.get_form_energy_per_atom(e)
                except Exception:
                    ehull, eform = float("nan"), float("nan")
            n = len(e.structure)
            results.append({
                "parent_label":    parent,
                "composition":     e.data.get("composition"),
                "comp_key":        e.data["comp_key"],
                "n_atoms":         n,
                "energy_per_atom": e.energy / n,
                "eform_per_atom":  eform,
                "ehull":           ehull,
                "sqs_seed":        e.data.get("sqs_seed"),
                "structure":       e.structure.as_dict(),
                "is_ground_state": bool(np.isfinite(ehull) and ehull <= 1e-6),
            })
    return results


# ---------------------------------------------------------------------------
# slab gamma
# ---------------------------------------------------------------------------

def _slab_area(structure):
    a, b, _ = structure.lattice.matrix
    return float(np.linalg.norm(np.cross(a, b)))


def _ground_state_per_atom(bulk_results):
    """(parent, comp_key) -> lowest energy_per_atom across bulk SQS samples."""
    best = {}
    for r in bulk_results:
        key = (r["parent_label"], r["comp_key"])
        if key not in best or r["energy_per_atom"] < best[key]["energy_per_atom"]:
            best[key] = r
    return {k: v["energy_per_atom"] for k, v in best.items()}


def analyse_surface(slab_items, bulk_results):
    mu = _ground_state_per_atom(bulk_results)
    entries = _dedup(
        _make_entries(slab_items),
        bucket_key=lambda e: (
            e.data["parent_label"],
            e.data["comp_key"],
            tuple(e.data.get("miller") or ()),
        ),
    )

    out = []
    for e in entries:
        parent = e.data["parent_label"]
        ck = e.data["comp_key"]
        n = len(e.structure)
        area = _slab_area(e.structure)
        mu_pa = mu.get((parent, ck))
        notes = []
        if mu_pa is None:
            gamma = None
            notes.append("no bulk reference at this composition")
        else:
            gamma = (e.energy - n * mu_pa) / (2.0 * area)
        out.append({
            "parent_label":      parent,
            "composition":       e.data.get("composition"),
            "comp_key":          ck,
            "miller":            e.data.get("miller"),
            "n_atoms":           n,
            "area_A2":           area,
            "E_slab":            e.energy,
            "mu_bulk_per_atom":  mu_pa,
            "gamma_eV_A2":       gamma,
            "gamma_J_m2":        (gamma * 16.0218) if gamma is not None else None,
            "sqs_seed":          e.data.get("sqs_seed"),
            "structure":         e.structure.as_dict(),
            "notes":             notes,
        })
    return out


# ---------------------------------------------------------------------------
# surface O-vacancy energetics
# ---------------------------------------------------------------------------

def _pristine_slab_index(slab_results):
    """(parent, comp_key, miller_tuple) -> lowest-E_slab pristine slab dict."""
    idx = {}
    for r in slab_results:
        key = (r["parent_label"], r["comp_key"], tuple(r.get("miller") or ()))
        if key not in idx or r["E_slab"] < idx[key]["E_slab"]:
            idx[key] = r
    return idx


def analyse_ovac(slab_vac_items, slab_results, mu_O2):
    pristine = _pristine_slab_index(slab_results)
    entries = _dedup(
        _make_entries(slab_vac_items),
        bucket_key=lambda e: (
            e.data["parent_label"],
            e.data["comp_key"],
            tuple(e.data.get("miller") or ()),
            round(float(e.data.get("vacancy_frac") or 0.0), 4),
        ),
    )

    out = []
    for e in entries:
        parent = e.data["parent_label"]
        ck = e.data["comp_key"]
        miller = tuple(e.data.get("miller") or ())
        p = pristine.get((parent, ck, miller))
        n_vac = (p["n_atoms"] - len(e.structure)) if p else None

        if not (p and n_vac and n_vac > 0):
            dE, reference = None, "no pristine slab"
        elif mu_O2 is None:
            dE = (e.energy - p["E_slab"]) / n_vac
            reference = "no O2 ref"
        else:
            dE = (e.energy + 0.5 * n_vac * mu_O2 - p["E_slab"]) / n_vac
            reference = "with mu_O2"

        out.append({
            "parent_label":     parent,
            "composition":      e.data.get("composition"),
            "comp_key":         ck,
            "miller":           list(miller) if miller else None,
            "vacancy_frac":     e.data.get("vacancy_frac"),
            "n_vacancies":      n_vac,
            "E_slab_vac":       e.energy,
            "E_slab_pristine":  p["E_slab"] if p else None,
            "dE_vac_per_O":     dE,
            "reference":        reference,
            "sqs_seed":         e.data.get("sqs_seed"),
            "structure":        e.structure.as_dict(),
        })
    return out
