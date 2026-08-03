"""Stage 1 of the solid-state cell path: does this pair survive contact?

Pure functions, no AiiDA -- importable from a workchain, a runner or a test.

THE POINT OF RUNNING THIS FIRST. Most electrode/electrolyte pairs react on
contact. Building an atomistic interface (Zur-McGill supercell + termination
enumeration + MLIP relaxation) for a pair that decomposes is wasted compute,
and the reaction energy costs nothing beyond a phase diagram we already build.
See docs/interfaces.md.

THE ALGORITHM is Richards, Miara, Wang, Kim and Ceder, *Interface Stability in
Solid-State Batteries*, Chem. Mater. 28, 266 (2016). Mix the two phases in
ratio x and compare the linear mixture against the phase equilibrium at that
composition:

    dE_D(x) = E_hull(x*c_A + (1-x)*c_B) - [ x*E_A + (1-x)*E_B ]

minimised over x. It is <= 0 by construction; strongly negative means the two
phases react. Under an applied potential the same construction runs on a GRAND
POTENTIAL phase diagram open to the working ion, which is what makes it
electrochemically meaningful rather than merely chemical (Nat. Rev. Mater. 5,
105 (2020); Angew. Chem. 2026 applies it to cathode/SE, SE/interlayer and
SE/coating interfaces).

We do NOT reimplement it: pymatgen ships it as
``pymatgen.analysis.interface_reactions.InterfacialReactivity`` and
``GrandPotentialInterfacialReactivity``. This module supplies the battery
framing -- open element = the working ion, mu swept between the discharged and
charged limits -- and a verdict the pipeline can gate on.

BEYOND THE DFT DATABASES. The point of uvsib is structures, interfaces and
reactions that are NOT in Materials Project, so `entries` is deliberately a
plain argument rather than an MPRester call: for a GNoME-generated,
CSP-generated or SQS-substituted phase the hull has to be built from OUR OWN
MLIP-relaxed entries (`workchains.utils.get_entries_from_db`). MP is then only
one possible source, useful for validating the machinery against known
chemistry -- not the source of truth.

**ONE ENERGY REFERENCE PER HULL.** This is the trap that comes with it. A hull
mixing MP's DFT entries with our MLIP entries produces a confident and
meaningless reaction energy: the two sit on different absolute references, so
`E_hull - E_mixture` is contaminated by a constant that does not cancel. The
same failure ruined the Pt corrosion tune set through a different route. Build
the hull from ONE source, and if a specific competing phase is missing from the
MLIP set, compute it with the MLIP rather than importing it from MP.
`assert_single_reference` below enforces this where the entries carry the
provenance to check it.
"""
from pymatgen.core import Composition, Element
from pymatgen.analysis.phase_diagram import PhaseDiagram, GrandPotentialPhaseDiagram
from pymatgen.analysis.interface_reactions import (InterfacialReactivity,
                                                   GrandPotentialInterfacialReactivity)

# eV/atom. Above this the pair is treated as compatible; a decomposition energy
# more negative than this means the junction reacts and an atomistic interface
# for it would be describing a phase that does not survive assembly. The
# threshold is a screening convention, not a law of nature -- Chem. Mater. 28,
# 266 reports many working cells in the -0.05 .. 0 band where a passivating
# interphase forms rather than runaway decomposition, which is why
# `reacts` and `severe` are reported separately.
REACTION_TOL = -0.020
SEVERE_TOL = -0.100


def select_head(entries, head):
    """Keep only the entries produced by `head` (untagged ones pass through).

    The heads DIFFER BY DESIGN across input.yaml stages -- bulk_relax and
    battery run matpes_r2scan while adsorbates and SQS run Default -- so a
    `method="MACE"` query legitimately returns a mix even from a freshly
    cleaned database. Refusing that mix would block a healthy campaign; the
    correct move is to select the reference this stage was configured for.

    Entries with no `model_head` tag are kept: they predate the tagging in
    phase_diagram.get_entries_from_db and dropping them would silently shrink
    an older hull.

    Returns (kept, n_dropped).
    """
    if not head:
        return list(entries), 0
    kept = []
    for e in entries:
        src = {}
        src.update(getattr(e, "parameters", None) or {})
        src.update(getattr(e, "data", None) or {})
        tag = src.get("model_head")
        if tag is None or tag == head:
            kept.append(e)
    return kept, len(entries) - len(kept)


def assert_single_reference(entries):
    """Refuse a hull built from mixed energy references.

    Entries carrying a recognisable provenance tag (`run_type`, `functional`,
    `model` or `model_head` in `entry.parameters` / `entry.data`) must all
    agree. Entries with no tag are passed over rather than assumed compatible
    -- this catches MP `ComputedEntry` objects (which carry `run_type`) mixed
    into an MLIP set, without failing on a homogeneous set that has no
    metadata.

    Heads are handled by `select_head`, not here: they differ across stages by
    design, so a mix is normal and must be SELECTED from rather than rejected.
    This function is for the mixing that is never legitimate -- DFT against
    MLIP.

    Raises ValueError, deliberately: a silently wrong reaction energy is worse
    than a stopped workchain.
    """
    tags = set()
    for e in entries:
        src = {}
        src.update(getattr(e, "parameters", None) or {})
        src.update(getattr(e, "data", None) or {})
        for key in ("run_type", "functional", "model"):
            if src.get(key):
                tags.add(f"{key}={src[key]}")
                break
    if len(tags) > 1:
        raise ValueError(
            f"entries span {len(tags)} energy references ({sorted(tags)}). "
            f"A hull mixing DFT and MLIP energies gives a reaction energy "
            f"contaminated by a constant offset that does not cancel. Build "
            f"the hull from one source.")
    return tags.pop() if tags else None


def _norm_per_atom(comp):
    """Composition normalised to one atom, so energies compare per atom."""
    return Composition({el: amt / comp.num_atoms for el, amt in comp.items()})


def interface_reaction(c1, c2, entries, working_ion=None, mu_ion=None):
    """Decomposition energy of the c1|c2 junction, eV/atom (<= 0).

    Parameters
    ----------
    c1, c2 : Composition | str
        The two contacting phases (electrode and electrolyte).
    entries : list[ComputedEntry]
        Everything in the joint chemical system. MUST already include both
        phases and the elemental references, or the hull is wrong -- this is
        the single most common way to get a confidently meaningless number.
    working_ion : str, optional
        Element symbol. When given together with `mu_ion` the calculation runs
        on the grand potential diagram open to that element, i.e. AT AN APPLIED
        POTENTIAL rather than in the chemically closed limit.
    mu_ion : float, optional
        Chemical potential of the working ion, eV. Convention as elsewhere in
        the battery path: mu = 0 is the metal-anode (discharged) limit and
        increasingly negative values correspond to charging.

    Returns
    -------
    dict with `energy` (eV/atom), `reacts`, `severe`, `products`, `x_min`,
    `open_element`, `mu`.
    """
    c1, c2 = Composition(c1), Composition(c2)
    if not entries:
        raise ValueError("no entries supplied -- the hull cannot be built and "
                         "any reaction energy from it would be meaningless")
    assert_single_reference(entries)

    pd = PhaseDiagram(entries)
    if working_ion is None or mu_ion is None:
        rxn = InterfacialReactivity(c1, c2, pd, norm=True, use_hull_energy=True)
        open_el, mu = None, None
    else:
        open_el = Element(working_ion)
        if open_el not in pd.elements:
            raise ValueError(
                f"working ion {working_ion} is not in the chemical system "
                f"{[str(e) for e in pd.elements]}; a grand potential diagram "
                f"cannot be opened to an absent element")
        # GrandPotentialPhaseDiagram wants the ABSOLUTE chemical potential;
        # our mu is referenced to the elemental metal, so shift by it.
        mu_abs = mu_ion + pd.el_refs[open_el].energy_per_atom
        gpd = GrandPotentialPhaseDiagram(entries, {open_el: mu_abs})
        rxn = GrandPotentialInterfacialReactivity(
            c1, c2, gpd, pd_non_grand=pd,
            include_no_mixing_energy=True, norm=True, use_hull_energy=True)
        mu = mu_ion

    x_min, energy = rxn.minimum
    products = sorted({str(p) for p in rxn.products})
    return {"energy": float(energy),
            "reacts": bool(energy < REACTION_TOL),
            "severe": bool(energy < SEVERE_TOL),
            "products": products,
            "x_min": float(x_min),
            "open_element": working_ion if open_el is not None else None,
            "mu": mu}


def screen_pair(electrode, electrolyte, entries, working_ion, mu_grid=None):
    """Run `interface_reaction` across the potential window and take the worst.

    A junction that is benign at open circuit can decompose once charged, so a
    single closed-system number is not a verdict. The worst case over the
    window is what decides whether an atomistic interface is worth building.

    `mu_grid` defaults to the discharged limit (0) down to -4.0 eV, which spans
    the usable window of a Li/Na cell.
    """
    if mu_grid is None:
        mu_grid = [0.0, -1.0, -2.0, -3.0, -4.0]

    scan = []
    for mu in mu_grid:
        r = interface_reaction(electrode, electrolyte, entries,
                               working_ion=working_ion, mu_ion=mu)
        scan.append(r)

    worst = min(scan, key=lambda r: r["energy"])
    closed = interface_reaction(electrode, electrolyte, entries)
    return {"electrode": str(Composition(electrode).reduced_formula),
            "electrolyte": str(Composition(electrolyte).reduced_formula),
            "working_ion": working_ion,
            "closed_system": closed,
            "worst": worst,
            "scan": scan,
            # the gate stage 2 reads: build an interface only for pairs that
            # survive the whole window
            "build_interface": not worst["severe"]}
