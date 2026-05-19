"""
NRR (Nitrogen Reduction Reaction) overpotential calculator.

Electrochemical analog of Haber-Bosch. Strongly limited by HER
competition and by weak *N2 binding on most metals.

Supported pathways
------------------
distal       : *N2 → *NNH → *NNH2 → *N + NH3 → *NH → *NH2 → NH3
               (distal N hydrogenated and released first)
alternating  : *N2 → *NNH → *NHNH → *NHNH2 → *N2H4 → *NH2 + NH3 → NH3
               (H atoms alternate between the two N atoms)
dissociative : N2 → 2 *N → 2 *NH → 2 *NH2 → 2 NH3
               (single-site bookkeeping; needs very strong *N binding)
"""

import numpy as np
from uvsib.workchains.utils import load_references, load_zpe

_ZPE = load_zpe("nrr")

# Pathway definitions. See co2rr.py for convention notes.
NRR_PATHWAYS = {
    "distal": {
        "equilibrium_potential": 0.092,   # V vs RHE (N2/NH3 equilibrium at 1 M, 1 bar)
        "n_electrons": 6,
        "steps": [
            {},
            {'*N2_ads': +1, '*':       -1, 'N2':  -1},                       # N2 + * → *N2 (chemical adsorption)
            {'*NNH':    +1, '*N2_ads': -1, 'H2': 1/2},                       # *N2 + H+ + e- → *NNH
            {'*NNH2':   +1, '*NNH':    -1, 'H2': 1/2},                       # *NNH + H+ + e- → *NNH2
            {'*N':      +1, '*NNH2':   -1, 'NH3': -1, 'H2': 1/2},            # *NNH2 + H+ + e- → *N + NH3(g)
            {'*NH':     +1, '*N':      -1, 'H2': 1/2},                       # *N + H+ + e- → *NH
            {'*NH2':    +1, '*NH':     -1, 'H2': 1/2},                       # *NH + H+ + e- → *NH2
            {'*':       +1, '*NH2':    -1, 'NH3': -1, 'H2': 1/2},            # *NH2 + H+ + e- → NH3(g) + *
        ],
    },
    "alternating": {
        "equilibrium_potential": 0.092,
        "n_electrons": 6,
        "steps": [
            {},
            {'*N2_ads': +1, '*':        -1, 'N2':  -1},                      # N2 + * → *N2
            {'*NNH':    +1, '*N2_ads':  -1, 'H2': 1/2},                      # *N2 + H+ + e- → *NNH
            {'*NHNH':   +1, '*NNH':     -1, 'H2': 1/2},                      # *NNH + H+ + e- → *NHNH (H to near N)
            {'*NHNH2':  +1, '*NHNH':    -1, 'H2': 1/2},                      # *NHNH + H+ + e- → *NHNH2
            {'*N2H4':   +1, '*NHNH2':   -1, 'H2': 1/2},                      # *NHNH2 + H+ + e- → *N2H4 (hydrazine)
            {'*NH2':    +1, '*N2H4':    -1, 'NH3': -1, 'H2': 1/2},           # *N2H4 + H+ + e- → *NH2 + NH3(g)
            {'*NH3':    +1, '*NH2':     -1, 'H2': 1/2},                      # *NH2 + H+ + e- → *NH3
            {'*':       +1, '*NH3':     -1, 'NH3': -1},                      # *NH3 → NH3(g) + * (chemical desorption)
        ],
    },
    "dissociative": {
        "equilibrium_potential": 0.092,
        "n_electrons": 6,
        "steps": [
            {},
            {'*N':   +2, '*':    -2, 'N2':  -1},                             # N2 + 2* → 2 *N (chemical scission)
            {'*NH':  +1, '*N':   -1, 'H2': 1/2},                             # *N + H+ + e- → *NH
            {'*NH2': +1, '*NH':  -1, 'H2': 1/2},                             # *NH + H+ + e- → *NH2
            {'*':    +1, '*NH2': -1, 'NH3': -1, 'H2': 1/2},                  # *NH2 + H+ + e- → NH3(g) + *
        ],
    },
}


def calculate_nrr_overpotential(adsorption_energies, pathway_name, method, func):
    """Calculate NRR overpotential using the CHE model.

    Parameters
    ----------
    adsorption_energies : dict
        DFT or ML energies of surface intermediates. Required keys depend
        on the pathway: '*N2_ads', '*NNH', '*NNH2', '*N', '*NH', '*NH2'
        for 'distal'; add '*NHNH', '*NHNH2', '*N2H4', '*NH3' for
        'alternating'; just '*N', '*NH', '*NH2' for 'dissociative'.
    pathway_name : str
        One of: 'distal', 'alternating', 'dissociative'.
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
        If a required adsorbate key is missing from adsorption_energies.
    """
    if pathway_name not in NRR_PATHWAYS:
        raise ValueError(
            f"Unsupported NRR pathway: '{pathway_name}'. "
            f"Supported: {list(NRR_PATHWAYS.keys())}"
        )

    refs = load_references(method, func)
    local_energy = adsorption_energies.copy()
    local_energy.update(refs)

    pathway = NRR_PATHWAYS[pathway_name]
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
