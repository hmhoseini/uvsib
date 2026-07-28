"""
ORR (Oxygen Reduction Reaction) overpotential calculator.

Cathodic counterpart of OER. Same *OH-*OOH scaling relation (~3.2 eV)
caps the activity at η ≈ 0.3-0.4 V on Pt-group metals.

Supported pathways
------------------
4e_associative   : O2 + 4(H+ + e-) → 2 H2O via *O2 → *OOH → *O → *OH
4e_dissociative  : O2 → 2 *O, then 2*O → 2 *OH → 2 H2O
2e_to_h2o2       : O2 + 2(H+ + e-) → H2O2 via *O2 → *OOH (no O-O cleavage)

Gas-phase references (O2, H2, H2O, and H2O2 for the 2e- route) arrive inside
adsorption_energies, computed per-molecule from molecular_references/*.vasp.
"""

from uvsib.workchains.utils import load_zpe, che_overpotential

_ZPE = load_zpe("orr")

# Pathway definitions. See co2rr.py for convention notes: each dict is ONE
# elementary reaction (products +, reactants -); consumed (H+ + e-) -> 'H2':
# -1/2, released H2O / H2O2 -> +1, consumed O2 -> -1. Steps are summed
# individually, never differenced (utils.che_overpotential).
ORR_PATHWAYS = {
    "4e_associative": {
        "equilibrium_potential": 1.23,   # V vs RHE (O2/H2O 4e- equilibrium)
        "n_electrons": 4,
        "steps": [
            {},
            {'*O2':     +1, '*':       -1, 'O2': -1},                         # O2 + * → *O2 (chemical adsorption)
            {'*OOH':    +1, '*O2':     -1, 'H2': -1/2},                       # *O2 + (H+ + e-) → *OOH
            {'*O':      +1, 'H2O': +1, '*OOH': -1, 'H2': -1/2},               # *OOH + (H+ + e-) → *O + H2O
            {'*OH':     +1, '*O':      -1, 'H2': -1/2},                       # *O + (H+ + e-) → *OH
            {'*':       +1, 'H2O': +1, '*OH': -1, 'H2': -1/2},                # *OH + (H+ + e-) → H2O + *
        ],
    },
    "4e_dissociative": {
        "equilibrium_potential": 1.23,
        "n_electrons": 4,
        "steps": [
            {},
            {'*O':  +2, '*':   -2, 'O2': -1},                                 # O2 + 2* → 2 *O (2-site dissociation)
            {'*OH': +1, '*O':  -1, 'H2': -1/2},                               # *O + (H+ + e-) → *OH
            {'*':   +1, 'H2O': +1, '*OH': -1, 'H2': -1/2},                    # *OH + (H+ + e-) → H2O + *
        ],
    },
    "2e_to_h2o2": {
        "equilibrium_potential": 0.70,   # V vs RHE (O2/H2O2 2e- equilibrium)
        "n_electrons": 2,
        "steps": [
            {},
            {'*O2':     +1, '*':       -1, 'O2': -1},                         # O2 + * → *O2
            {'*OOH':    +1, '*O2':     -1, 'H2': -1/2},                       # *O2 + (H+ + e-) → *OOH
            {'*':       +1, 'H2O2': +1, '*OOH': -1, 'H2': -1/2},              # *OOH + (H+ + e-) → H2O2(aq) + *
        ],
    },
}


def calculate_orr_overpotential(adsorption_energies, pathway_name):
    """Calculate ORR overpotential using the CHE model.

    Parameters
    ----------
    adsorption_energies : dict
        DFT or ML energies of surface intermediates. Required keys depend
        on the pathway: '*O2', '*OOH', '*O', '*OH' for the 4e- routes;
        '*O2', '*OOH' only for the 2e- route.
    pathway_name : str
        One of: '4e_associative', '4e_dissociative', '2e_to_h2o2'.

    Returns
    -------
    overpotential : float
        Thermodynamic overpotential in V (positive = more difficult).
    dg_steps : list[float]
        ΔG per elementary step at U = 0 V vs RHE (eV).
    dg_cumulative : list[float]
        Cumulative free energies [0, ΔG1, ΔG1+ΔG2, ...] at U = 0 V (eV).

    Raises
    ------
    ValueError
        If pathway_name is not supported.
    KeyError
        If a required adsorbate or gas-phase reference key is missing from
        adsorption_energies (the '2e_to_h2o2' route additionally needs 'H2O2').
    """
    if pathway_name not in ORR_PATHWAYS:
        raise ValueError(
            f"Unsupported ORR pathway: '{pathway_name}'. "
            f"Supported: {list(ORR_PATHWAYS.keys())}"
        )

    # Gas-phase references (O2, H2, H2O, H2O2) are computed per-molecule from
    # molecular_references/*.vasp and arrive inside adsorption_energies
    # alongside the surface intermediates and the clean slab. ZPE is added
    # inside che_overpotential, so the stored energies are raw electronic.
    pathway = ORR_PATHWAYS[pathway_name]
    return che_overpotential(
        pathway["steps"], adsorption_energies, _ZPE,
        pathway["equilibrium_potential"], reduction=True,
    )
