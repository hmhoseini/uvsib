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

    bulk_candidates(chemical_formula) -> [bulk dict, ...]  # now includes "source"/"ml_bulk_model"
    surfaces_for_bulk(chemical_formula) -> {structure_uuid: [surface dict, ...]}  # now includes "n_atoms"/"area"/"shift"
    reaction_results(chemical_formula, reaction, reaction_path) -> {structure_uuid: [candidate dict, ...]}  # now includes "repeat"/"coverage" (adsorbate concentration, in ML)
    step_labels(reaction, reaction_path) -> [str, ...] | None
    summarize(chemical_formula, reaction, reaction_path) -> [bulk summary dict, ...]  # each surface now also carries its OWN reaction_candidates/best_candidate
    report(chemical_formula, reaction, reaction_path, plot_dir=None) -> [bulk summary dict, ...]

Plotting (matplotlib imported lazily -- only needed if you call these):

    plot_free_energy_diagram(dg_cumulative, labels=None, ..., ax=None)
    plot_bulk_comparison(summaries, ax_ehull=None, ax_eta=None)
    plot_surface_fed(surface, reaction, reaction_path, ax=None)  # FED for one surface's best candidate
    plot_bulk_detail(summary, reaction, reaction_path)  # one FED panel per surface, stacked in a column
    save_report_figures(summaries, reaction, reaction_path, output_dir) -> {structure_uuid: path, ...}

HTML report (also lazily needs matplotlib; see result-sample.html for the reference layout):

    raw_data(chemical_formula, reaction, reaction_path, summaries=None) -> [bulk dict, ...]
        -> summarize()'s output, plus the full bulk structure (pymatgen
        Structure dict) and full surface slab structure (pymatgen Slab dict)
        for every bulk/surface -- everything render_html_report() writes to
        "raw_data.json" for the page's download button.

    render_html_report(chemical_formula, reaction, reaction_path, summaries=None, output_path="report.html")
        -> writes a self-contained HTML file (Tailwind/lucide chrome, plots
        embedded as base64 PNGs): a stable-bulks table (uuid + source), one
        surfaces table per bulk (miller index, surface energy, size, best
        candidate's supercell repeat), and one reaction-path diagram per
        surface (titled with its site + eta + coverage). The size (# atoms/
        area) shown per surface is scaled to its BEST candidate's own
        supercell repeat (see _repeat_multiplier()), not the bare relaxed
        slab -- different candidates on the same surface can use different
        repeats for different coverages, so this scaling is per-candidate,
        not a fixed per-surface number. Does NOT include the cross-bulk
        comparison chart (E-above-hull + best eta per bulk) -- that plot
        (plot_bulk_comparison()) is not rendered as part of the standard
        pipeline report; call it directly if you want it. Also writes
        "raw_data.json" (see raw_data()) next to output_path and
        links a download button to it. Also returns the HTML string.

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
  available (e.g. the same process/script that ran the pipeline). This is why
  ``MainWorkChain.pipeline_report`` (``uvsib/workchains/main.py``) -- which
  calls ``report()`` and ``render_html_report()`` as its own outline step
  right after AKMC, writing into ``settings.REPORTS_DIR/<composition>_
  <reaction>_<reaction_path>/`` (a fixed directory next to the ``uvsib``
  package, not the per-run ``settings.run_dir``) -- works safely: it already
  runs inside a profile-loaded AiiDA worker process. ``render_html_report()`` ALSO always
  tries this same profile-loading import via ``_ml_surface_model()``/
  ``_ml_stage_head()`` (for the "ML Bulk/Surface Model" + task metadata
  fields), but defensively -- unlike ``step_labels()``, it catches the
  failure and just shows "&mdash;" rather than raising, so calling
  ``render_html_report()`` outside a profile-loaded environment still works
  (just without those fields). ``raw_data()`` itself stays DB-only like the
  functions above it -- no profile needed.
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
from uvsib.db.utils import query_by_columns, query_structure


def _bulk_source(structure_uuid, method=None):
    """Provenance of one bulk structure: the ``source``/``method`` of its
    ``DBStructureVersion`` row (e.g. source="MPDB_stb"/"csp"/"generated",
    method="MACE"/... -- see ``add_structures`` in ``db/utils.py``). Prefers
    the version matching ``method`` (the composition's ``ml_bulk_model``,
    i.e. the version PhaseDiagramMLWorkChain actually ranked) when given and
    present; otherwise falls back to the first version found. Returns
    ``None`` if the structure has no stored version at all."""
    versions = query_structure({"uuid": structure_uuid})
    if not versions:
        return None
    if method is not None:
        for v in versions:
            if v.method == method:
                return {"source": v.source, "method": v.method}
    v = versions[0]
    return {"source": v.source, "method": v.method}


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

    candidates = []
    for entry in stable_struct.get("ml_selection", []):
        provenance = _bulk_source(entry["uuid"], method=model)
        candidates.append({
            "structure_uuid": entry["uuid"],
            "ehull": entry["ehull"],
            "selected_above_threshold": entry.get("selected_above_threshold", False),
            "ehull_threshold": threshold,
            "ml_bulk_model": model,
            "source": provenance["source"] if provenance else None,
        })
    candidates.sort(key=lambda c: c["ehull"])
    return candidates


def _slab_area(slab):
    """Surface area (Å²) of a pymatgen ``Slab.as_dict()`` blob, from the
    in-plane lattice vectors a, b (|a x b|). Returns ``None`` if the slab has
    no lattice matrix."""
    matrix = (slab.get("lattice") or {}).get("matrix")
    if not matrix:
        return None
    a, b = matrix[0], matrix[1]
    cx = a[1] * b[2] - a[2] * b[1]
    cy = a[2] * b[0] - a[0] * b[2]
    cz = a[0] * b[1] - a[1] * b[0]
    return (cx ** 2 + cy ** 2 + cz ** 2) ** 0.5


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
            "n_atoms": len(slab.get("sites") or []),
            "area": _slab_area(slab),
            "shift": slab.get("shift"),
        })
    for surfaces in by_uuid.values():
        surfaces.sort(key=lambda s: s["formation_energy"]
                      if s["formation_energy"] is not None else float("inf"))
    return dict(by_uuid)


def _parse_repeat(repeat):
    """Parse a ``DBSurfaceMLAdsorbate.repeat`` value into an ``(nx, ny, nz)``
    int tuple (z is always 1; only nx, ny multiply in-plane coverage -- see
    ``get_multipliers()``/``make_supercell()`` in ``codes/files/adsorbates.py``).
    In practice (confirmed against the live DB), ``query_by_columns`` returns
    this ``Text`` column as a POSTGRES ARRAY LITERAL, e.g. ``"{1,2,1}"`` --
    curly braces, not JSON/Python list brackets. That string is deliberately
    NOT run through ``ast.literal_eval``: Python parses ``{1,2,1}`` as a SET
    literal, which silently collapses duplicate entries (``"{1,1,1}"`` ->
    ``{1}``, a 1-element set; ``"{2,2,1}"`` -> the wrong 2-element ``{1, 2}``)
    -- exactly the repeat values that matter most (no repeat, or equal x/y
    repeat) would then parse to the wrong shape or silently fail. Also
    tolerates an already-parsed list/tuple, or (for robustness against a
    differently-configured column) JSON/Python list syntax like
    ``"[1, 2, 1]"``. Returns ``None`` if unparseable."""
    if repeat is None:
        return None
    if isinstance(repeat, (list, tuple)):
        values = repeat
    else:
        text = str(repeat).strip()
        if text.startswith("{") and text.endswith("}"):
            inner = text[1:-1].strip()
            values = [v.strip() for v in inner.split(",")] if inner else []
        else:
            import ast
            try:
                values = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return None
    try:
        return tuple(int(v) for v in values)
    except (TypeError, ValueError):
        return None


def _coverage(repeat):
    """Adsorbate coverage (surface concentration), in monolayers, from the
    in-plane (nx, ny) supercell repeat: one adsorbate per nx*ny repeated
    surface cells -> 1/(nx*ny) ML. Returns ``None`` if ``repeat`` is
    unparseable or has a zero in-plane multiplier."""
    parsed = _parse_repeat(repeat)
    if not parsed or len(parsed) < 2:
        return None
    nx, ny = parsed[0], parsed[1]
    if not nx or not ny:
        return None
    return 1.0 / (nx * ny)


def _coverage_label(repeat):
    """Human-readable coverage label, e.g. ``"1/4 ML (2x2)"``, from a raw
    ``repeat`` value. Returns ``None`` if ``repeat`` is unparseable."""
    parsed = _parse_repeat(repeat)
    if not parsed or len(parsed) < 2:
        return None
    nx, ny = parsed[0], parsed[1]
    if not nx or not ny:
        return None
    n = nx * ny
    return f"1/{n} ML ({nx}x{ny})" if n != 1 else "1 ML (1x1)"


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
            "repeat": _parse_repeat(row.repeat),
            "coverage": _coverage(row.repeat),
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
    changed between pipeline reruns -- is appended after, with ``bulk=None``).

    Each surface dict under ``"surfaces"`` also carries its OWN
    ``"reaction_candidates"``/``"best_candidate"`` (the reaction_results rows
    for that specific surface_id, ranked by ascending eta) alongside the
    bulk-wide versions -- so a per-surface reaction-path plot always has
    exactly the candidates that surface produced, not the bulk's best overall."""
    bulks = {b["structure_uuid"]: b for b in bulk_candidates(chemical_formula)}
    surfaces_by_uuid = surfaces_for_bulk(chemical_formula)
    results_by_uuid = reaction_results(chemical_formula, reaction, reaction_path)

    all_uuids = set(bulks) | set(surfaces_by_uuid) | set(results_by_uuid)
    summaries = []
    for uid in all_uuids:
        surfaces = surfaces_by_uuid.get(uid, [])
        candidates = results_by_uuid.get(uid, [])

        candidates_by_surface = defaultdict(list)
        for c in candidates:
            candidates_by_surface[c["surface_id"]].append(c)
        for surf in surfaces:
            # already eta-sorted: candidates arrives eta-sorted from
            # reaction_results(), and grouping preserves that order.
            surf_candidates = candidates_by_surface.get(surf["surface_id"], [])
            surf["reaction_candidates"] = surf_candidates
            surf["best_candidate"] = surf_candidates[0] if surf_candidates else None

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
        ax.hlines(g, i - 0.3, i + 0.3, linewidth=4, color="crimson" if on_pds else "steelblue")
    for i in range(n - 1):
        ax.plot([i + 0.3, i + 1 - 0.3], [dg_cumulative[i], dg_cumulative[i + 1]],
                linestyle="--", linewidth=2,
                color="crimson" if i == pds_index else "gray")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel(r"$\Delta G$ (eV)", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.5)

    subtitle_parts = []
    if pds_index is not None:
        subtitle_parts.append(f"PDS: {labels[pds_index]} -> {labels[pds_index + 1]} "
                               f"({step_deltas[pds_index]:.2f} eV)")
    if equilibrium_potential is not None:
        subtitle_parts.append(f"U_eq = {equilibrium_potential:.2f} V")
    subtitle = "  |  ".join(subtitle_parts)
    # subtitle on its own line rather than appended to the same line -- a
    # single-line title (miller + site + eta + coverage + PDS + U_eq) easily
    # overflows a fixed-width panel and gets clipped at the right edge.
    full_title = f"{title or ''}\n{subtitle}" if subtitle else (title or "")
    ax.set_title(full_title, fontsize=11)

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


def plot_surface_fed(surface, reaction, reaction_path, ax=None):
    """Free-energy diagram for ONE surface's best (lowest-eta)
    reaction-path candidate -- the per-surface counterpart to
    ``plot_free_energy_diagram``, which plots one already-chosen candidate's
    numbers. Every stable surface gets its own diagram, since different
    surfaces of the same bulk can favor different sites/PDS."""
    plt = _require_matplotlib()

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4, 4))

    miller = str(tuple(surface["miller_index"])) if surface["miller_index"] else "?"
    best = surface.get("best_candidate")
    if best is None:
        ax.text(0.5, 0.5, "no reaction-path\ncandidates", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title(miller, fontsize=10)
    else:
        labels = step_labels(reaction, reaction_path) or \
            [str(i) for i in range(len(best["dG_cumulative"]))]
        coverage_bit = _coverage_label(best.get("repeat"))
        coverage_bit = f", {coverage_bit}" if coverage_bit else ""
        plot_free_energy_diagram(
            best["dG_cumulative"], labels=labels,
            title=f"{miller}  site={best['site_type']}, eta={best['eta']:.2f} V{coverage_bit}",
            ax=ax,
        )

    if own_fig:
        fig.tight_layout()
        return fig, ax
    return ax


def plot_bulk_detail(summary, reaction, reaction_path):
    """Per-bulk figure: one reaction-path free-energy diagram (via
    ``plot_surface_fed``) for each surface that actually HAS a
    reaction-path candidate, stacked in a SINGLE COLUMN. A row layout would
    rescale every panel to fit a fixed width, so a bulk with 5 surfaces gets
    cramped panels while a bulk with 1 gets an oversized one -- the number of
    surfaces per bulk is unpredictable, so a column keeps each panel the
    same fixed, readable size regardless of count; only the figure's total
    height grows. Surfaces with no candidate are skipped rather than padded
    with an empty "no candidates" panel. Surface formation energy is already
    in the surfaces table alongside this figure, so it is not duplicated
    here as a bar chart."""
    plt = _require_matplotlib()

    fed_surfaces = [s for s in summary["surfaces"] if s["best_candidate"]]
    n_fed_panels = len(fed_surfaces)
    ehull_bit = f"E_hull = {summary['bulk']['ehull']:.3f} eV/atom" if summary["bulk"] else "E_hull unknown"

    # tight_layout() does not reserve room for suptitle() -- without an
    # explicit rect, the bulk-level suptitle collides with (renders on top
    # of) the topmost panel's own (2-line) title. Reserve a FIXED number of
    # inches at the top regardless of total figure height, since that height
    # scales with n_fed_panels and a fixed top *fraction* would over- or
    # under-reserve depending on panel count.
    suptitle_inches = 0.55

    if not n_fed_panels:
        fig_height = 3.0
        fig, ax = plt.subplots(figsize=(6, fig_height))
        ax.text(0.5, 0.5, "no reaction-path candidates for this bulk's surfaces",
                ha="center", va="center", wrap=True)
        ax.set_axis_off()
        fig.suptitle(f"{summary['structure_uuid'][:8]}  ({ehull_bit})", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 1 - suptitle_inches / fig_height))
        return fig

    fig_height = 4.8 * n_fed_panels
    fig, axes = plt.subplots(n_fed_panels, 1, figsize=(6.5, fig_height))
    if n_fed_panels == 1:
        axes = [axes]
    for ax, surface in zip(axes, fed_surfaces):
        plot_surface_fed(surface, reaction, reaction_path, ax=ax)

    fig.suptitle(f"{summary['structure_uuid'][:8]}  ({ehull_bit})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 1 - suptitle_inches / fig_height))
    return fig


def save_report_figures(summaries, reaction, reaction_path, output_dir):
    """Render and save one detail figure per bulk (one reaction-path FED
    panel per surface, stacked in a column) into ``output_dir`` (created if
    missing). Returns ``{structure_uuid: path, ...}``."""
    import os
    plt = _require_matplotlib()
    os.makedirs(output_dir, exist_ok=True)

    paths = {}

    for summary in summaries:
        fig = plot_bulk_detail(summary, reaction, reaction_path)
        detail_path = os.path.join(output_dir, f"bulk_{summary['structure_uuid'][:8]}.png")
        fig.savefig(detail_path, dpi=150)
        plt.close(fig)
        paths[summary["structure_uuid"]] = detail_path

    return paths


################################################################################
# HTML report -- mirrors the layout of result-sample.html (Tailwind + lucide).
# Plots are embedded as base64 PNGs so the output is a single, shareable file.
################################################################################

def _fig_to_data_uri(fig):
    """Render a matplotlib figure to a base64 PNG data URI and close it."""
    import io
    import base64
    plt = _require_matplotlib()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _ml_surface_model():
    """The face-build (surface relaxation) ML model name, read the SAME way
    SurfaceBuilderWorkChain does -- ``settings.inputs["face_build"]["model"]``
    (see ``_model_cmdline()`` in ``workchains/surface_builder.py``). This is a
    run-level config value, not a per-row DB field, so it lazily imports
    ``uvsib.workflows.settings`` -- which DOES load an AiiDA profile (see the
    module-level caveat above) -- and returns ``None`` if that fails (e.g.
    called outside a profile-loaded environment)."""
    try:
        from uvsib.workflows import settings
        return settings.inputs["face_build"]["model"]
    except Exception:
        return None


def _ml_stage_head(stage):
    """The ``head`` (task) an ML stage's model was run with -- e.g. UMA's
    "omat" bulk-relax head vs "oc20" face-build head -- read the SAME way
    ``_model_cmdline()`` (``workchains/surface_builder.py``) passes it on as
    ``--task_name`` for face-build (``bulk_relax`` is analogous; see
    ``input.yaml``'s ``bulk_relax``/``face_build`` blocks, each a
    ``model``/``head``/... dict). ``stage`` is ``"bulk_relax"`` or
    ``"face_build"``. Like ``_ml_surface_model()``, this is a run-level
    config value, not a per-row DB field, so it lazily imports
    ``uvsib.workflows.settings`` (AiiDA-profile-loading) and returns ``None``
    if that fails, or if the stage has no ``head`` configured."""
    try:
        from uvsib.workflows import settings
        return settings.inputs[stage].get("head")
    except Exception:
        return None


def _model_label(model, task):
    """Display label combining an ML model name with its task/head, e.g.
    ``"UMA (omat)"`` -- or just the model name if no task is available.
    Returns ``None`` if ``model`` itself is unavailable."""
    if model is None:
        return None
    return f"{model} ({task})" if task else model


def _repeat_multiplier(parsed_repeat):
    """In-plane atom/area multiplier (``nx * ny``) implied by an already
    ``_parse_repeat``-d ``(nx, ny, nz)`` repeat tuple -- the factor a
    surface's base (1x1, as stored on ``DBSurface``) ``n_atoms``/``area``
    must be scaled by to match the actual supercell a given reaction
    candidate was computed on (``make_supercell((nx, ny, 1))`` in
    ``generate_adsorbed_structures()``, ``codes/files/adsorbates.py`` --
    z never repeats). Different candidates on the SAME surface can use
    different repeats (different coverages), so this must be applied
    per-candidate, not once per surface. Returns 1 (no scaling, i.e. the
    base cell) if ``parsed_repeat`` is ``None``/unparseable/has a zero
    in-plane multiplier."""
    if not parsed_repeat or len(parsed_repeat) < 2:
        return 1
    nx, ny = parsed_repeat[0], parsed_repeat[1]
    return nx * ny if nx and ny else 1


def raw_data(chemical_formula, reaction, reaction_path, summaries=None):
    """Comprehensive raw-data export for one (composition, reaction,
    reaction_path): everything ``summarize()`` computes (ehull, surface
    formation energies, eta, dG, coverage, ...) PLUS the full bulk structure
    (pymatgen ``Structure.as_dict()``, from ``DBStructureVersion.structure``)
    and full surface slab structure (pymatgen ``Slab.as_dict()``, from
    ``DBSurface.slab``) for every bulk/surface listed in the report -- the
    underlying data behind every number the HTML page shows, for anyone who
    wants to reproduce or further analyze it outside this report.

    ``summaries`` lets a caller reuse an already-computed ``summarize()``
    result instead of re-querying the database."""
    if summaries is None:
        summaries = summarize(chemical_formula, reaction, reaction_path)

    bulk_structures = {}
    for s in summaries:
        uid = s["structure_uuid"]
        model = s["bulk"]["ml_bulk_model"] if s["bulk"] else None
        versions = query_structure({"uuid": uid})
        version = next((v for v in versions if v.method == model), versions[0]) if versions else None
        bulk_structures[uid] = version.structure if version else None

    slab_by_surface_id = {
        row.id: row.slab
        for row in query_by_columns(DBSurface, {"composition": chemical_formula})
    }

    data = []
    for s in summaries:
        entry = dict(s)
        entry["bulk_structure"] = bulk_structures.get(s["structure_uuid"])
        entry["surfaces"] = [
            {**surf, "slab_structure": slab_by_surface_id.get(surf["surface_id"])}
            for surf in s["surfaces"]
        ]
        data.append(entry)
    return data


def render_html_report(chemical_formula, reaction, reaction_path, summaries=None,
                        output_path="report.html"):
    """Render a self-contained HTML report (layout mirrors ``result-sample.html``)
    for one (composition, reaction, reaction_path): a table of stable bulks
    (uuid + provenance source), one surfaces table per bulk (miller index,
    surface formation energy, size), and one reaction-path free-energy
    diagram per surface. Figures are embedded as base64 PNGs, so the output
    file has no external dependencies besides the Tailwind/lucide CDN
    scripts used for layout/icons.

    Also writes ``raw_data.json`` next to ``output_path`` (via ``raw_data()``)
    and links a "Download Raw Data (JSON)" button to it -- every number and
    structure behind the page, not just what's rendered into the tables.

    ``summaries`` lets a caller reuse an already-computed ``summarize()`` (or
    ``report()``) result instead of re-querying the database; otherwise
    ``summarize()`` is called here. Writes to ``output_path`` and also
    returns the HTML string."""
    import os
    import json
    import html as _html

    if summaries is None:
        summaries = summarize(chemical_formula, reaction, reaction_path)

    raw_data_filename = "raw_data.json"

    n_bulks = len(summaries)
    n_surfaces = sum(s["n_surfaces"] for s in summaries)
    n_candidates = sum(s["n_reaction_candidates"] for s in summaries)
    etas = [s["best_candidate"]["eta"] for s in summaries if s["best_candidate"]]
    best_eta = min(etas) if etas else None
    model = next((s["bulk"]["ml_bulk_model"] for s in summaries if s["bulk"]), None)
    bulk_task = _ml_stage_head("bulk_relax")
    surface_model = _ml_surface_model()
    surface_task = _ml_stage_head("face_build")
    threshold = next((s["bulk"]["ehull_threshold"] for s in summaries if s["bulk"]), None)

    def esc(x):
        return _html.escape(str(x)) if x is not None else "&mdash;"

    # -- stable-bulks table ----------------------------------------------
    bulk_rows = []
    for s in summaries:
        b, uid = s["bulk"], s["structure_uuid"]
        if b:
            ehull_cell, source_cell = f'{b["ehull"]:.4f}', esc(b["source"])
        else:
            ehull_cell = source_cell = "&mdash;"
        bulk_rows.append(f"""
                  <tr class="border-t">
                    <td class="px-6 py-4 font-mono text-xs text-slate-900" title="{esc(uid)}">{esc(uid[:8])}</td>
                    <td class="px-6 py-4">{source_cell}</td>
                    <td class="px-6 py-4">{ehull_cell}</td>
                    <td class="px-6 py-4">{s["n_surfaces"]}</td>
                  </tr>""")

    # -- per-bulk sections: surfaces table + reaction-path FED grid ------
    bulk_sections = []
    for s in summaries:
        uid, b = s["structure_uuid"], s["bulk"]
        ehull_bit = f'E<sub>hull</sub> = {b["ehull"]:.4f} eV/atom' if b else "E<sub>hull</sub> unknown"
        source_bit = esc(b["source"]) if b else "&mdash;"

        surface_rows = []
        for surf in s["surfaces"]:
            miller = str(tuple(surf["miller_index"])) if surf["miller_index"] else "?"
            best = surf["best_candidate"]
            eta_cell = f'{best["eta"]:.3f} V' if best else "&mdash;"
            # surf["n_atoms"]/["area"] are the base (1x1) slab DBSurface
            # stored; the best candidate on this surface may have been
            # computed on a repeated supercell (different candidates on the
            # SAME surface can use different repeats for different
            # coverages), so scale by that candidate's own repeat rather
            # than showing the un-repeated base numbers alongside its eta.
            mult = _repeat_multiplier(best.get("repeat")) if best else 1
            n_atoms_cell = surf["n_atoms"] * mult if surf["n_atoms"] is not None else "&mdash;"
            area_cell = f'{surf["area"] * mult:.2f}' if surf["area"] is not None else "&mdash;"
            fe_cell = f'{surf["formation_energy"]:.4f}' if surf["formation_energy"] is not None else "&mdash;"
            repeat = best.get("repeat") if best else None
            repeat_cell = f"({repeat[0]}, {repeat[1]})" if repeat and len(repeat) >= 2 else "&mdash;"
            surface_rows.append(f"""
                      <tr class="border-t">
                        <td class="px-4 py-3 font-mono text-xs">{surf["surface_id"]}</td>
                        <td class="px-4 py-3">{esc(miller)}</td>
                        <td class="px-4 py-3">{fe_cell}</td>
                        <td class="px-4 py-3">{n_atoms_cell}</td>
                        <td class="px-4 py-3">{area_cell}</td>
                        <td class="px-4 py-3">{eta_cell}</td>
                        <td class="px-4 py-3">{repeat_cell}</td>
                      </tr>""")
        surface_rows_html = "".join(surface_rows) or (
            '<tr><td colspan="7" class="px-4 py-4 text-slate-400 italic">'
            'no stable surfaces found</td></tr>')

        if any(surf["best_candidate"] for surf in s["surfaces"]):
            fed_block = (f'<img src="{_fig_to_data_uri(plot_bulk_detail(s, reaction, reaction_path))}" '
                         f'alt="Reaction path diagrams for bulk {esc(uid[:8])}" '
                         f'class="w-full rounded-lg border" />')
        elif s["surfaces"]:
            fed_block = ('<p class="text-sm text-slate-400 italic">'
                         "no reaction-path candidates for this bulk's surfaces</p>")
        else:
            fed_block = '<p class="text-sm text-slate-400 italic">no stable surfaces to plot</p>'

        bulk_sections.append(f"""
        <div class="rounded-xl border bg-white p-6">
          <h3 class="font-bold text-slate-900 mb-1">Bulk {esc(uid[:8])}</h3>
          <p class="text-xs text-slate-400 font-mono mb-3">{esc(uid)}</p>
          <p class="text-sm text-slate-600 mb-4">{ehull_bit} &nbsp;|&nbsp; source: {source_bit}</p>

          <div class="overflow-x-auto mb-6 border rounded-lg">
            <table class="w-full text-sm text-left text-slate-700">
              <thead class="bg-slate-50 text-slate-500 uppercase text-xs">
                <tr>
                  <th class="px-4 py-3">Surface ID</th>
                  <th class="px-4 py-3">Miller Index</th>
                  <th class="px-4 py-3">Surface Energy (eV/&Aring;&sup2;)</th>
                  <th class="px-4 py-3" title="Scaled to the best candidate's supercell repeat, not the bare relaxed slab">
                    # Atoms</th>
                  <th class="px-4 py-3" title="Scaled to the best candidate's supercell repeat, not the bare relaxed slab">
                    Area (&Aring;&sup2;)</th>
                  <th class="px-4 py-3">Best &eta;</th>
                  <th class="px-4 py-3" title="In-plane supercell repeat (nx, ny) the best candidate's clean slab was built on, before the adsorbate was placed">
                    Repeat (n<sub>x</sub>, n<sub>y</sub>)</th>
                </tr>
              </thead>
              <tbody>{surface_rows_html}</tbody>
            </table>
          </div>

          <h4 class="font-bold text-slate-900 mb-2 text-sm">Reaction Path Diagrams</h4>
          {fed_block}
        </div>""")

    best_eta_cell = f"{best_eta:.3f} V" if best_eta is not None else "&mdash;"
    threshold_cell = f"{threshold:.3f} eV/atom" if threshold is not None else "&mdash;"

    html_doc = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{esc(chemical_formula)} &mdash; {esc(reaction)} / {esc(reaction_path)}</title>

    <script src="https://cdn.tailwindcss.com"></script>
    <script>
      tailwind.config = {{ theme: {{ extend: {{ colors: {{ primary: "#2563eb" }} }} }} }};
    </script>
    <script src="https://unpkg.com/lucide@latest"></script>
  </head>

  <body class="flex flex-col min-h-screen bg-slate-50">
    <div class="bg-white border-b px-6 py-6 sticky top-0 z-10">
      <div class="max-w-7xl mx-auto flex flex-col gap-2">
        <h1 class="text-slate-900 text-2xl font-bold tracking-tight">
          {esc(chemical_formula)} &mdash; {esc(reaction)} / {esc(reaction_path)}
        </h1>
        <p class="text-slate-500 text-sm">
          PhaseDiagramMLWorkChain &rarr; SurfaceBuilderWorkChain &rarr; AdsorbatesWorkChain
        </p>
      </div>
    </div>

    <div class="p-6 md:p-12 max-w-7xl mx-auto w-full grid grid-cols-1 lg:grid-cols-3 gap-8">
      <div class="lg:col-span-2 flex flex-col gap-8">

        <section>
          <div class="flex items-center gap-2 mb-4">
            <i data-lucide="info" class="w-5 h-5 text-primary"></i>
            <h2 class="text-xl font-bold text-slate-900">Executive Summary</h2>
          </div>
          <div class="rounded-xl border bg-white shadow-sm p-6">
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div class="p-3 rounded-xl bg-slate-50 border border-slate-100">
                <p class="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Bulk Candidates</p>
                <p class="text-lg font-bold text-slate-900">{n_bulks}</p>
              </div>
              <div class="p-3 rounded-xl bg-slate-50 border border-slate-100">
                <p class="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Surfaces Found</p>
                <p class="text-lg font-bold text-slate-900">{n_surfaces}</p>
              </div>
              <div class="p-3 rounded-xl bg-slate-50 border border-slate-100">
                <p class="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Reaction Candidates</p>
                <p class="text-lg font-bold text-slate-900">{n_candidates}</p>
              </div>
              <div class="p-3 rounded-xl bg-slate-50 border border-slate-100">
                <p class="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Best &eta;</p>
                <p class="text-lg font-bold text-slate-900">{best_eta_cell}</p>
              </div>
            </div>
          </div>
        </section>

        <section>
          <div class="flex items-center gap-2 mb-4">
            <i data-lucide="layers" class="w-5 h-5 text-primary"></i>
            <h2 class="text-xl font-bold text-slate-900">Stable Bulk Structures</h2>
          </div>
          <div class="bg-white border rounded-xl overflow-hidden">
            <div class="overflow-x-auto">
              <table class="w-full text-sm text-left text-slate-700">
                <thead class="bg-slate-50 text-slate-500 uppercase text-xs">
                  <tr>
                    <th class="px-6 py-3">UUID</th>
                    <th class="px-6 py-3">Source</th>
                    <th class="px-6 py-3">E above hull (eV/atom)</th>
                    <th class="px-6 py-3">Surfaces</th>
                  </tr>
                </thead>
                <tbody>{"".join(bulk_rows) or '<tr><td colspan="4" class="px-6 py-4 text-slate-400 italic">no bulk candidates found</td></tr>'}</tbody>
              </table>
            </div>
          </div>
        </section>

        <section class="flex flex-col gap-6">
          <div class="flex items-center gap-2">
            <i data-lucide="bar-chart-3" class="w-5 h-5 text-primary"></i>
            <h2 class="text-xl font-bold text-slate-900">Per-Bulk Detail: Surfaces &amp; Reaction Paths</h2>
          </div>
          {"".join(bulk_sections)}
        </section>

      </div>

      <div class="flex flex-col gap-8">
        <div class="rounded-xl border bg-white shadow-sm p-6">
          <h3 class="text-lg font-bold text-slate-900 mb-6">Pipeline Metadata</h3>
          <div class="space-y-5">
            <div class="flex items-center gap-4">
              <div class="size-10 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center shrink-0">
                <i data-lucide="beaker" class="w-6 h-6"></i>
              </div>
              <div>
                <p class="text-xs font-bold text-slate-400 uppercase tracking-wider">Composition</p>
                <p class="text-sm font-bold text-slate-900">{esc(chemical_formula)}</p>
              </div>
            </div>
            <div class="flex items-center gap-4">
              <div class="size-10 rounded-full bg-purple-100 text-purple-600 flex items-center justify-center shrink-0">
                <i data-lucide="flask-conical" class="w-6 h-6"></i>
              </div>
              <div>
                <p class="text-xs font-bold text-slate-400 uppercase tracking-wider">Reaction</p>
                <p class="text-sm font-bold text-slate-900">{esc(reaction)}</p>
              </div>
            </div>
            <div class="flex items-center gap-4">
              <div class="size-10 rounded-full bg-orange-100 text-orange-600 flex items-center justify-center shrink-0">
                <i data-lucide="git-branch" class="w-6 h-6"></i>
              </div>
              <div>
                <p class="text-xs font-bold text-slate-400 uppercase tracking-wider">Reaction Path</p>
                <p class="text-sm font-bold text-slate-900">{esc(reaction_path)}</p>
              </div>
            </div>
            <div class="flex items-center gap-4">
              <div class="size-10 rounded-full bg-slate-100 text-slate-600 flex items-center justify-center shrink-0">
                <i data-lucide="cpu" class="w-6 h-6"></i>
              </div>
              <div>
                <p class="text-xs font-bold text-slate-400 uppercase tracking-wider">ML Bulk Model</p>
                <p class="text-sm font-bold text-slate-900">{esc(_model_label(model, bulk_task))}</p>
              </div>
            </div>
            <div class="flex items-center gap-4">
              <div class="size-10 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center shrink-0">
                <i data-lucide="layers-3" class="w-6 h-6"></i>
              </div>
              <div>
                <p class="text-xs font-bold text-slate-400 uppercase tracking-wider">ML Surface Model</p>
                <p class="text-sm font-bold text-slate-900">{esc(_model_label(surface_model, surface_task))}</p>
              </div>
            </div>
            <div class="flex items-center gap-4">
              <div class="size-10 rounded-full bg-green-100 text-green-600 flex items-center justify-center shrink-0">
                <i data-lucide="gauge" class="w-6 h-6"></i>
              </div>
              <div>
                <p class="text-xs font-bold text-slate-400 uppercase tracking-wider">E-hull Threshold</p>
                <p class="text-sm font-bold text-slate-900">{threshold_cell}</p>
              </div>
            </div>
          </div>
        </div>

        <div class="rounded-xl bg-primary text-white p-6">
          <h3 class="text-lg font-bold mb-2">Need the raw data?</h3>
          <p class="text-white/80 text-sm mb-6">
            Download every number and structure behind this page -- bulk and
            surface structures, ehull, surface energies, eta, and dG -- as
            one JSON file.
          </p>
          <a
            href="{esc(raw_data_filename)}"
            download
            class="block w-full text-center bg-white text-primary rounded-md py-2 font-bold"
          >
            Download Raw Data (JSON)
          </a>
        </div>
      </div>
    </div>

    <script>
      lucide.createIcons();
    </script>
  </body>
</html>
"""

    raw_data_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), raw_data_filename)
    with open(raw_data_path, "w") as f:
        json.dump(raw_data(chemical_formula, reaction, reaction_path, summaries=summaries),
                   f, indent=2, default=str)

    with open(output_path, "w") as f:
        f.write(html_doc)
    return html_doc
