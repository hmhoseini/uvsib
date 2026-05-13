"""
CO2RR (CO2 Reduction Reaction) overpotential calculator.
Supported pathways
------------------
co2_to_co     : CO2 → CO  (2e-)
co2_to_hcooh  : CO2 → HCOOH  (2e-)
co_to_ch4     : CO → CH4  (6e-, requires '*CO' in adsorption_energies)
co_to_ch3oh   : CO → CH3OH  (4e-, requires '*CO' in adsorption_energies)
co2_to_c2h4   : CO2 → C2H4  (8e-, C-C coupling via *OCCO)
"""

import numpy as np
from uvsib.workchains.utils import load_references, load_zpe

_ZPE = load_zpe("co2rr")

# Pathway definitions: each step is a dict of {species: stoichiometric_coefficient}.
# Positive = produced (energy added), negative = consumed (energy subtracted).
# The initial empty dict {} represents the clean surface reference state.
# n_electrons is informational only (not used in the CHE calculation).
CO2RR_PATHWAYS = {
    "co2_to_co": {
        "equilibrium_potential": 0.11,   # V vs RHE
        "n_electrons": 2,
        "steps": [
            {},
            {'*COOH': +1, '*':    -1, 'CO2':  -1, 'H2': 1/2},  # CO2 + H+ + e- → *COOH
            {'*CO':   +1, '*':    -1, 'H2O':  -1, 'H2': 1/2},  # *COOH + H+ + e- → *CO + H2O
        ],
    },
    "co2_to_hcooh": {
        "equilibrium_potential": -0.05,
        "n_electrons": 2,
        "steps": [
            {},
            {'*OCHO': +1, '*':    -1, 'CO2':   -1, 'H2': 1/2},  # CO2 + H+ + e- → *OCHO
            {'*':     +1, 'HCOOH':-1, '*OCHO': -1, 'H2': 1/2},  # *OCHO + H+ + e- → HCOOH(g)
        ],
    },
    "co_to_ch4": {
        "equilibrium_potential": 0.24,
        "n_electrons": 6,
        "steps": [
            {},
            {'*CHO':  +1, '*CO':   -1, 'H2': 1/2},               # *CO + H+ + e- → *CHO
            {'*CHOH': +1, '*CHO':  -1, 'H2': 1/2},               # *CHO + H+ + e- → *CHOH
            {'*CH':   +1, '*CHOH': -1, 'H2O': -1, 'H2': 1/2},   # *CHOH → *CH + H2O + H+ + e-
            {'*CH2':  +1, '*CH':   -1, 'H2': 1/2},               # *CH + H+ + e- → *CH2
            {'*CH3':  +1, '*CH2':  -1, 'H2': 1/2},               # *CH2 + H+ + e- → *CH3
            {'*':     +1, 'CH4':   -1, '*CH3': -1, 'H2': 1/2},   # *CH3 + H+ + e- → CH4(g)
        ],
    },
    "co_to_ch3oh": {
        "equilibrium_potential": 0.38,
        "n_electrons": 4,
        "steps": [
            {},
            {'*CHO':   +1, '*CO':    -1, 'H2': 1/2},
            {'*CHOH':  +1, '*CHO':   -1, 'H2': 1/2},
            {'*CH2OH': +1, '*CHOH':  -1, 'H2': 1/2},
            {'*':      +1, 'CH3OH':  -1, '*CH2OH': -1, 'H2': 1/2},
        ],
    },
    "co2_to_c2h4": {
        "equilibrium_potential": 0.34,
        "n_electrons": 8,
        "steps": [
            {},
            {'*COOH': +1, '*':     -1, 'CO2':   -1, 'H2': 1/2},
            {'*CO':   +1, '*':     -1, 'H2O':   -1, 'H2': 1/2},
            {'*OCCO': +1, '*CO':   -2},                            # 2*CO → *OCCO (C-C coupling)
            {'*CCHO': +1, '*OCCO': -1, 'H2': 1/2},
            {'*':     +1, 'C2H4':  -1, '*CCHO': -1, 'H2': 1/2},
        ],
    },
}


def calculate_co2rr_overpotential(adsorption_energies, pathway_name, method, func):
    """Calculate CO2RR overpotential using the CHE model.

    Parameters
    ----------
    adsorption_energies : dict
        DFT or ML energies of surface intermediates. Required keys depend
        on the pathway. For pathways starting from *CO ('co_to_ch4',
        'co_to_ch3oh'), '*CO' must be supplied by the caller.
    pathway_name : str
        One of: 'co2_to_co', 'co2_to_hcooh', 'co_to_ch4',
                'co_to_ch3oh', 'co2_to_c2h4'.
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
        Cumulative free energies including the final gas-phase reference
        state at U = equilibrium_potential (eV).

    Raises
    ------
    ValueError
        If pathway_name is not supported.
    NotImplementedError
        If the method/func combination has no defined references.
    KeyError
        If a required adsorbate key is missing from adsorption_energies.
    """
    if pathway_name not in CO2RR_PATHWAYS:
        raise ValueError(
            f"Unsupported CO2RR pathway: '{pathway_name}'. "
            f"Supported: {list(CO2RR_PATHWAYS.keys())}"
        )

    refs = load_references(method, func)
    local_energy = adsorption_energies.copy()
    local_energy.update(refs)

    pathway = CO2RR_PATHWAYS[pathway_name]
    reaction_path = pathway["steps"]
    equilibrium_potential = pathway["equilibrium_potential"]
    n_steps = len(reaction_path)

    # Accumulate absolute free energies at each state
    dga_list = []
    for r in reaction_path:
        dgi = sum(
            (local_energy[q] + _ZPE[q]) * e
            for q, e in r.items()
        )
        dga_list.append(dgi)

    # Final gas-phase reference state (product fully desorbed)
    dga_list.append(0.0)
    dga = np.array(dga_list)   # length = n_steps + 1

    # Elementary step free energies at U = 0 V vs RHE
    dg_steps = (dga[1:] - dga[:-1]).tolist()
    overpotential = max(dg_steps) - equilibrium_potential

    # Shift cumulative profile to equilibrium potential
    charges = np.arange(n_steps + 1)
    dga -= equilibrium_potential * charges
    dg_cumulative = dga.tolist()

    return overpotential, dg_steps, dg_cumulative
