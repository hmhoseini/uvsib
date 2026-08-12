"""Post-pipeline reporting for
``PhaseDiagramMLWorkChain -> SurfaceBuilderWorkChain -> AdsorbatesWorkChain``.

Each stage writes its results to a different table (``DBComposition.stable_struct``
for bulk hull stability, ``DBSurface`` for the relaxed slabs SurfaceBuilderWorkChain
kept, ``DBSurfaceMLAdsorbate`` for the reaction-path free energies AdsorbatesWorkChain
computed on top of those slabs). This module joins the three by ``structure_uuid``
so that, per candidate bulk, one place answers "is this bulk any good for this
reaction/reaction_path?": is it on/near the hull, which surfaces were stable enough
to keep, and what does the resulting free-energy diagram look like.

Public API:

    bulk_candidates(chemical_formula) -> [bulk dict, ...]
    surfaces_for_bulk(chemical_formula) -> {structure_uuid: [surface dict, ...]}
    reaction_results(chemical_formula, reaction, reaction_path) -> {structure_uuid: [candidate dict, ...]}
    step_labels(reaction, reaction_path) -> [str, ...] | None
    summarize(chemical_formula, reaction, reaction_path) -> [bulk summary dict, ...]
    report(chemical_formula, reaction, reaction_path, plot_dir=None) -> [bulk summary dict, ...]

Plotting (matplotlib imported lazily -- only needed if you call these):

    plot_free_energy_diagram(dg_cumulative, labels=None, ..., ax=None)
    plot_bulk_comparison(summaries, ax_ehull=None, ax_eta=None)
    plot_bulk_detail(summary, reaction, reaction_path, ax_surfaces=None, ax_fed=None)
    save_report_figures(summaries, reaction, reaction_path, output_dir) -> {uuid: {...path}, "comparison": path}

Caveats worth knowing:

* Importing this module, and calling ``bulk_candidates``/``surfaces_for_bulk``/
  ``reaction_results``/``summarize``/``report(..., plot_dir=None)``, never
  needs an AiiDA profile -- they talk to the Postgres tables directly, like
  ``akmc_analysis.py``. Only ``step_labels()`` -- and therefore
  ``plot_bulk_detail()``/``save_report_figures()``/``report(..., plot_dir=...)``,
  which call it for FED x-axis labels -- lazily imports the matching
  ``uvsib.workchains.<reaction>`` module, and every one of those transitively
  imports ``uvsib.workflows.settings``, which DOES load an AiiDA profile. Run
  those specific calls from an environment where a profile is already
  available (e.g. the same process/script that ran the pipeline).
* "Surfaces found" means "slabs SurfaceBuilderWorkChain kept after relaxation
  and ranked by formation energy" -- it already dropped non-converged slabs
  before storing (see ``inspect_relax`` in ``surface_builder.py``); there is no
  further stability filter applied here.
* Every ``reaction_results`` row already passed AdsorbatesWorkChain's own
  screening (eta <= 2.0 eV, see ``reaction_map`` in ``adsorbates.py``). A bulk
  with zero reaction_results rows for a given reaction/reaction_path either had
  no candidate under that threshold, or is missing a pathway intermediate that
  dissociated during relaxation (see the ``KeyError`` handling in
  ``AdsorbatesWorkChain.store_results_ml``) -- not necessarily "bad", possibly
  "not evaluated".
"""
from collections import defaultdict

from uvsib.db.tables import DBComposition, DBSurface, DBSurfaceMLAdsorbate
from uvsib.db.utils import query_by_columns


def bulk_candidates(chemical_formula):
    """Every bulk structure PhaseDiagramMLWorkChain kept as stable for
    ``chemical_formula``, sorted by ascending E-above-hull (most stable
    first)."""
    rows = query_by_columns(DBComposition, {"composition": chemical_formula})
    if not rows:
        return []
    stable_struct = rows[0].stable_struct or {}
    threshold = stable_struct.get("ml_ehull_threshold")
    model = stable_struct.get("ml_bulk_model")

    candidates = [
        {
            "structure_uuid": entry["uuid"],
            "ehull": entry["ehull"],
            "selected_above_threshold": entry.get("selected_above_threshold", False),
            "ehull_threshold": threshold,
            "ml_bulk_model": model,
        }
        for entry in stable_struct.get("ml_selection", [])
    ]
    candidates.sort(key=lambda c: c["ehull"])
    return candidates


def surfaces_for_bulk(chemical_formula):
    """DBSurface rows for ``chemical_formula``, grouped by bulk structure_uuid
    and ranked within each bulk by ascending surface formation energy (most
    stable surface first)."""
    rows = query_by_columns(DBSurface, {"composition": chemical_formula})
    by_uuid = defaultdict(list)
    for row in rows:
        slab = row.slab or {}
        by_uuid[str(row.structure_uuid)].append({
            "surface_id": row.id,
            "miller_index": slab.get("miller_index"),
            "formation_energy": row.formation_energy,
        })
    for surfaces in by_uuid.values():
        surfaces.sort(key=lambda s: s["formation_energy"]
                      if s["formation_energy"] is not None else float("inf"))
    return dict(by_uuid)


def reaction_results(chemical_formula, reaction, reaction_path):
    """DBSurfaceMLAdsorbate rows for one (composition, reaction, reaction_path),
    grouped by bulk structure_uuid and ranked within each bulk by ascending eta
    (kinetically/thermodynamically best candidate first)."""
    rows = query_by_columns(DBSurfaceMLAdsorbate, {
        "composition": chemical_formula,
        "reaction": reaction,
        "reaction_path": reaction_path,
    })
    by_uuid = defaultdict(list)
    for row in rows:
        by_uuid[str(row.structure_uuid)].append({
            "row_id": row.id,
            "surface_id": row.surface_id,
            "miller_index": row.surface_miller_index,
            "site_type": row.site_type,
            "ads_coord": row.ads_coord,
            "eta": row.eta,
            "dG_steps": row.dG_steps,
            "dG_cumulative": row.dG_cumulative,
        })
    for candidates in by_uuid.values():
        candidates.sort(key=lambda c: c["eta"])
    return dict(by_uuid)


_OER_LABELS = ["*", "*OH", "*O", "*OOH", "O2 + *"]

_REACTION_MODULES = {
    "CO2RR": ("co2rr", "CO2RR_PATHWAYS"),
    "CER": ("cer", "CER_PATHWAYS"),
    "HER": ("her", "HER_PATHWAYS"),
    "NRR": ("nrr", "NRR_PATHWAYS"),
    "NOXRR": ("noxrr", "NOXRR_PATHWAYS"),
    "ORR": ("orr", "ORR_PATHWAYS"),
}


def step_labels(reaction, reaction_path):
    """Free-energy-diagram x-axis labels lining up with one candidate's
    ``dG_cumulative``, derived from the SAME ``*_PATHWAYS`` step dict
    AdsorbatesWorkChain used to compute it -- each step's newly formed
    ``*``-prefixed species (coefficient +1) -- so labels can never drift out of
    sync with the numbers. Returns ``None`` if the reaction/reaction_path is
    unrecognized. OER has no ``*_PATHWAYS`` dict (its steps are hard-coded in
    ``oer.py``), so its 5 labels are hard-coded here to match."""
    reaction = reaction.strip().upper()
    if reaction == "OER":
        return list(_OER_LABELS)

    if reaction not in _REACTION_MODULES:
        return None
    mod_name, dict_name = _REACTION_MODULES[reaction]
    import importlib
    module = importlib.import_module(f"uvsib.workchains.{mod_name}")
    pathways = getattr(module, dict_name)

    reaction_path = reaction_path.strip().lower()
    if reaction_path not in pathways:
        return None

    labels = ["*"]
    for step in pathways[reaction_path]["steps"][1:]:
        formed = [species for species, coeff in step.items()
                  if coeff == 1 and species.startswith("*")]
        labels.append(formed[0] if formed else f"step {len(labels)}")
    return labels


def summarize(chemical_formula, reaction, reaction_path):
    """Per-bulk report: hull stability, the surfaces SurfaceBuilderWorkChain
    found, and the reaction-path candidates AdsorbatesWorkChain stored on top
    of them. One dict per bulk structure_uuid, bulks with a known hull
    position sorted by ascending ehull first (any leftover uuid that only
    shows up in DBSurface/DBSurfaceMLAdsorbate -- e.g. a bulk selection that
    changed between pipeline reruns -- is appended after, with ``bulk=None``)."""
    bulks = {b["structure_uuid"]: b for b in bulk_candidates(chemical_formula)}
    surfaces_by_uuid = surfaces_for_bulk(chemical_formula)
    results_by_uuid = reaction_results(chemical_formula, reaction, reaction_path)

    all_uuids = set(bulks) | set(surfaces_by_uuid) | set(results_by_uuid)
    summaries = []
    for uid in all_uuids:
        surfaces = surfaces_by_uuid.get(uid, [])
        candidates = results_by_uuid.get(uid, [])
        summaries.append({
            "structure_uuid": uid,
            "bulk": bulks.get(uid),
            "surfaces": surfaces,
            "n_surfaces": len(surfaces),
            "best_surface_formation_energy": surfaces[0]["formation_energy"] if surfaces else None,
            "reaction_candidates": candidates,
            "n_reaction_candidates": len(candidates),
            "best_candidate": candidates[0] if candidates else None,
        })

    summaries.sort(key=lambda s: (s["bulk"] is None,
                                   s["bulk"]["ehull"] if s["bulk"] else 0.0))
    return summaries


def report(chemical_formula, reaction, reaction_path, plot_dir=None):
    """Convenience entry point: ``summarize()``, optionally also rendering and
    saving the comparison + per-bulk figures to ``plot_dir`` (created if
    missing) and attaching their paths onto each summary as ``"figures"``."""
    summaries = summarize(chemical_formula, reaction, reaction_path)
    if plot_dir is not None:
        paths = save_report_figures(summaries, reaction, reaction_path, plot_dir)
        for summary in summaries:
            summary["figures"] = paths.get(summary["structure_uuid"])
    return summaries


################################################################################
# Plotting -- matplotlib is only imported once one of these is actually called.
################################################################################

def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plotting needs matplotlib (pip install matplotlib); "
            "the rest of this module works without it."
        ) from exc
    return plt


def plot_free_energy_diagram(dg_cumulative, labels=None, equilibrium_potential=None,
                              title=None, ax=None):
    """Staircase free-energy diagram for one candidate's ``dG_cumulative``
    (the same values AdsorbatesWorkChain derived eta from). The
    potential-determining step -- the single largest rise, which sets eta -- is
    highlighted in red."""
    plt = _require_matplotlib()

    dg_cumulative = list(dg_cumulative)
    n = len(dg_cumulative)
    if labels is None:
        labels = [str(i) for i in range(n)]
    step_deltas = [dg_cumulative[i + 1] - dg_cumulative[i] for i in range(n - 1)]
    pds_index = max(range(len(step_deltas)), key=lambda i: step_deltas[i]) if step_deltas else None

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(1.6 * n + 1, 4))

    for i, g in enumerate(dg_cumulative):
        on_pds = pds_index is not None and i in (pds_index, pds_index + 1)
        ax.hlines(g, i - 0.3, i + 0.3, linewidth=3, color="crimson" if on_pds else "steelblue")
    for i in range(n - 1):
        ax.plot([i + 0.3, i + 1 - 0.3], [dg_cumulative[i], dg_cumulative[i + 1]],
                linestyle="--", linewidth=1.5,
                color="crimson" if i == pds_index else "gray")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(r"$\Delta G$ (eV)")
    ax.axhline(0, color="black", linewidth=0.5)

    subtitle = ""
    if pds_index is not None:
        subtitle += (f"  |  PDS: {labels[pds_index]} -> {labels[pds_index + 1]} "
                     f"({step_deltas[pds_index]:.2f} eV)")
    if equilibrium_potential is not None:
        subtitle += f"  |  U_eq = {equilibrium_potential:.2f} V"
    ax.set_title((title or "") + subtitle, fontsize=10)

    if own_fig:
        fig.tight_layout()
        return fig, ax
    return ax


def plot_bulk_comparison(summaries, ax_ehull=None, ax_eta=None):
    """Two-panel across-bulk comparison for one composition: E-above-hull per
    bulk (with the ML selection threshold drawn in) and the best reaction-path
    eta found on each bulk -- the two numbers that most directly say which bulk
    is worth pursuing further."""
    plt = _require_matplotlib()

    own_fig = ax_ehull is None and ax_eta is None
    if own_fig:
        fig, (ax_ehull, ax_eta) = plt.subplots(1, 2, figsize=(max(6, 1.4 * len(summaries)), 4))

    tick_labels = [s["structure_uuid"][:8] for s in summaries]

    ehulls = [s["bulk"]["ehull"] if s["bulk"] else float("nan") for s in summaries]
    threshold = next((s["bulk"]["ehull_threshold"] for s in summaries if s["bulk"]), None)
    bar_colors = ["goldenrod" if (s["bulk"] and s["bulk"]["selected_above_threshold"]) else "seagreen"
                  for s in summaries]
    ax_ehull.bar(tick_labels, ehulls, color=bar_colors)
    if threshold is not None:
        ax_ehull.axhline(threshold, color="black", linestyle="--", linewidth=1,
                          label=f"threshold = {threshold:.2f}")
        ax_ehull.legend(fontsize=8)
    ax_ehull.set_ylabel("E above hull (eV/atom)")
    ax_ehull.set_title("Bulk stability")
    ax_ehull.tick_params(axis="x", rotation=45)

    best_eta = [s["best_candidate"]["eta"] if s["best_candidate"] else float("nan") for s in summaries]
    ax_eta.bar(tick_labels, best_eta, color="steelblue")
    ax_eta.set_ylabel(r"best $\eta$ (V)")
    ax_eta.set_title("Best reaction-path candidate per bulk")
    ax_eta.tick_params(axis="x", rotation=45)

    if own_fig:
        fig.tight_layout()
        return fig, (ax_ehull, ax_eta)
    return ax_ehull, ax_eta


def plot_bulk_detail(summary, reaction, reaction_path, ax_surfaces=None, ax_fed=None):
    """Two-panel per-bulk figure: surface formation energies for every stable
    surface SurfaceBuilderWorkChain found on this bulk, and the free-energy
    diagram of its best reaction-path candidate."""
    plt = _require_matplotlib()

    own_fig = ax_surfaces is None and ax_fed is None
    if own_fig:
        fig, (ax_surfaces, ax_fed) = plt.subplots(1, 2, figsize=(10, 4))

    surfaces = summary["surfaces"]
    if surfaces:
        miller_labels = [str(tuple(s["miller_index"])) if s["miller_index"] else "?" for s in surfaces]
        energies = [s["formation_energy"] for s in surfaces]
        ax_surfaces.bar(miller_labels, energies, color="slateblue")
        ax_surfaces.tick_params(axis="x", rotation=45)
    else:
        ax_surfaces.text(0.5, 0.5, "no stable surfaces found", ha="center", va="center")
        ax_surfaces.set_axis_off()
    ax_surfaces.set_ylabel(r"surface formation energy (eV/$\AA^2$)")
    ax_surfaces.set_title(f"Surfaces found ({len(surfaces)})")

    best = summary["best_candidate"]
    if best:
        labels = step_labels(reaction, reaction_path) or \
            [str(i) for i in range(len(best["dG_cumulative"]))]
        plot_free_energy_diagram(
            best["dG_cumulative"], labels=labels,
            title=f"Best candidate (site={best['site_type']}, eta={best['eta']:.2f} V)",
            ax=ax_fed,
        )
    else:
        ax_fed.text(0.5, 0.5, "no reaction-path candidates", ha="center", va="center")
        ax_fed.set_axis_off()

    if own_fig:
        ehull_bit = f"E_hull = {summary['bulk']['ehull']:.3f} eV/atom" if summary["bulk"] else "E_hull unknown"
        fig.suptitle(f"{summary['structure_uuid'][:8]}  ({ehull_bit})")
        fig.tight_layout()
        return fig, (ax_surfaces, ax_fed)
    return ax_surfaces, ax_fed


def save_report_figures(summaries, reaction, reaction_path, output_dir):
    """Render and save the comparison figure plus one detail figure per bulk
    into ``output_dir`` (created if missing). Returns
    ``{structure_uuid: path, ..., "comparison": path}``."""
    import os
    plt = _require_matplotlib()
    os.makedirs(output_dir, exist_ok=True)

    paths = {}

    fig, _ = plot_bulk_comparison(summaries)
    comparison_path = os.path.join(output_dir, "bulk_comparison.png")
    fig.savefig(comparison_path, dpi=150)
    plt.close(fig)
    paths["comparison"] = comparison_path

    for summary in summaries:
        fig, _ = plot_bulk_detail(summary, reaction, reaction_path)
        detail_path = os.path.join(output_dir, f"bulk_{summary['structure_uuid'][:8]}.png")
        fig.savefig(detail_path, dpi=150)
        plt.close(fig)
        paths[summary["structure_uuid"]] = detail_path

    return paths
