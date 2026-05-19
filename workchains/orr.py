"""
ORR (Oxygen Reduction Reaction) overpotential calculator.

Cathodic counterpart of OER. Same *OH-*OOH scaling relation (~3.2 eV)
caps the activity at η ≈ 0.3-0.4 V on Pt-group metals.

Supported pathways
------------------
4e_associative   : O2 + 4(H+ + e-) → 2 H2O via *O2 → *OOH → *O → *OH
4e_dissociative  : O2 → 2 *O, then 2*O → 2 *OH → 2 H2O
2e_to_h2o2       : O2 + 2(H+ + e-) → H2O2 via *O2 → *OOH (no O-O cleavage)

The 2e- pathway requires 'H2O2' in references.yaml; add it before use.
"""

import numpy as np
from uvsib.workchains.utils import load_references, load_zpe

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


def calculate_orr_overpotential(adsorption_energies, pathway_name, method, func):
    """Calculate ORR overpotential using the CHE model.

    Parameters
    ----------
    adsorption_energies : dict
        DFT or ML energies of surface intermediates. Required keys depend
        on the pathway: '*O2_ads', '*OOH', '*O', '*OH' for the 4e- routes;
        '*O2_ads', '*OOH' only for the 2e- route.
    pathway_name : str
        One of: '4e_associative', '4e_dissociative', '2e_to_h2o2'.
    method : str
        'dft' or ML model name (e.g. 'uPET', 'mace', 'mattersim').
    func : str
        Functional / reference set, e.g. 'r2SCAN'.

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
    NotImplementedError
        If the method/func combination has no defined references.
    KeyError
        If a required adsorbate / gas reference is missing
        (H2O2 must be added to references.yaml for '2e_to_h2o2').
    """
    if pathway_name not in ORR_PATHWAYS:
        raise ValueError(
            f"Unsupported ORR pathway: '{pathway_name}'. "
            f"Supported: {list(ORR_PATHWAYS.keys())}"
        )

    refs = load_references(method, func)
    local_energy = adsorption_energies.copy()
    local_energy.update(refs)

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
