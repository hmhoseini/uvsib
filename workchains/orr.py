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

import numpy as np
from uvsib.workchains.utils import load_zpe

_ZPE = load_zpe("orr")

# Pathway definitions. See co2rr.py for convention notes.
ORR_PATHWAYS = {
    "4e_associative": {
        "equilibrium_potential": 1.23,   # V vs RHE (O2/H2O 4e- equilibrium)
        "n_electrons": 4,
        "steps": [
            {},
            {'*O2_ads': +1, '*':       -1, 'O2':   -1},                       # O2 + * → *O2 (chemical adsorption)
            {'*OOH':    +1, '*O2_ads': -1, 'H2': 1/2},                        # *O2 + H+ + e- → *OOH
            {'*O':      +1, '*OOH':    -1, 'H2O':  -1, 'H2': 1/2},            # *OOH + H+ + e- → *O + H2O
            {'*OH':     +1, '*O':      -1, 'H2': 1/2},                        # *O + H+ + e- → *OH
            {'*':       +1, '*OH':     -1, 'H2O':  -1, 'H2': 1/2},            # *OH + H+ + e- → H2O + *
        ],
    },
    "4e_dissociative": {
        "equilibrium_potential": 1.23,
        "n_electrons": 4,
        "steps": [
            {},
            {'*O':  +2, '*':   -2, 'O2':   -1},                               # O2 + 2* → 2 *O (2-site dissociation)
            {'*OH': +1, '*O':  -1, 'H2': 1/2},                                # *O + H+ + e- → *OH
            {'*':   +1, '*OH': -1, 'H2O':  -1, 'H2': 1/2},                    # *OH + H+ + e- → H2O + *
        ],
    },
    "2e_to_h2o2": {
        "equilibrium_potential": 0.70,   # V vs RHE (O2/H2O2 2e- equilibrium)
        "n_electrons": 2,
        "steps": [
            {},
            {'*O2_ads': +1, '*':       -1, 'O2':   -1},                       # O2 + * → *O2
            {'*OOH':    +1, '*O2_ads': -1, 'H2': 1/2},                        # *O2 + H+ + e- → *OOH
            {'*':       +1, '*OOH':    -1, 'H2O2': -1, 'H2': 1/2},            # *OOH + H+ + e- → H2O2(aq) + *
        ],
    },
}


def calculate_orr_overpotential(adsorption_energies, pathway_name):
    """Calculate ORR overpotential using the CHE model.

    Parameters
    ----------
    adsorption_energies : dict
        DFT or ML energies of surface intermediates. Required keys depend
        on the pathway: '*O2_ads', '*OOH', '*O', '*OH' for the 4e- routes;
        '*O2_ads', '*OOH' only for the 2e- route.
    pathway_name : str
        One of: '4e_associative', '4e_dissociative', '2e_to_h2o2'.

    Returns
    -------
    overpotential : float
        Thermodynamic overpotential in V (positive = more difficult).
    dg_steps : list[float]
        ΔG per elementary step at U = 0 V vs RHE (eV).
    dg_cumulative : list[float]
        Cumulative free energies at U = equilibrium_potential (eV).

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
    # uniformly below via _ZPE, so the stored energies are raw electronic.
    local_energy = adsorption_energies

    pathway = ORR_PATHWAYS[pathway_name]
    reaction_path = pathway["steps"]
    equilibrium_potential = pathway["equilibrium_potential"]
    n_steps = len(reaction_path)

    dga_list = []
    for r in reaction_path:
        dgi = sum(
            (local_energy[q] + _ZPE[q]) * e
            for q, e in r.items()
        )
        dga_list.append(dgi)

    dga_list.append(0.0)
    dga = np.array(dga_list)

    dg_steps = (dga[1:] - dga[:-1]).tolist()
    overpotential = max(dg_steps) - equilibrium_potential

    charges = np.arange(n_steps + 1)
    dga -= equilibrium_potential * charges
    dg_cumulative = dga.tolist()

    return overpotential, dg_steps, dg_cumulative
