"""
NOXRR (NOx Reduction Reaction) overpotential calculator.
Supported pathways
------------------
no_dissociative  : NO → N2  via dissociative N coupling
no_to_nh3_noh    : NO → NH3 via *NOH intermediate
no_to_nh3_nhoh   : NO → NH3 via *NHOH intermediate
no_to_n2o        : NO → N2O via *N2O2 coupling
no2_to_no        : NO2 → NO  (deoxygenation)
no3_to_nh3       : NO3- → NH3
no3_to_n2        : NO3- → N2
"""

import numpy as np
from uvsib.workchains.utils import load_references, load_zpe

_ZPE = load_zpe("noxrr")

# Pathway definitions. See co2rr.py for convention notes.
NOXRR_PATHWAYS = {
    "no_dissociative": {
        "equilibrium_potential": -0.10,  # V vs RHE
        "n_electrons": 0,                # non-electrochemical coupling steps
        "steps": [
            {},
            {'*N':    +1, '*':    -1, 'NO':    -1, 'O': +1},   # NO → *N + O(g)
            {'*N2O2': +1, '*N':   -2},                          # 2*N → *N2O2 coupling
            {'*':     +1, 'N2':   -1, '*N2O2': -1},            # *N2O2 → N2(g) + *
        ],
    },
    "no_to_nh3_noh": {
        "equilibrium_potential": 0.27,
        "n_electrons": 6,
        "steps": [
            {},
            {'*NOH':  +1, '*':    -1, 'NO':    -1, 'H2': 1/2},  # NO + H+ + e- → *NOH
            {'*N':    +1, '*NOH': -1, 'H2O':   -1, 'H2': 1/2},  # *NOH → *N + H2O + H+ + e-
            {'*NH':   +1, '*N':   -1, 'H2': 1/2},               # *N  + H+ + e- → *NH
            {'*NH2':  +1, '*NH':  -1, 'H2': 1/2},               # *NH + H+ + e- → *NH2
            {'*NH3':  +1, '*NH2': -1, 'H2': 1/2},               # *NH2 + H+ + e- → *NH3
            {'*':     +1, 'NH3':  -1, '*NH3':  -1},             # *NH3 → NH3(g) + *
        ],
    },
    "no_to_nh3_nhoh": {
        "equilibrium_potential": 0.27,
        "n_electrons": 6,
        "steps": [
            {},
            {'*NOH':  +1, '*':    -1, 'NO':    -1, 'H2': 1/2},
            {'*NHOH': +1, '*NOH': -1, 'H2': 1/2},               # *NOH + H+ + e- → *NHOH
            {'*NH2':  +1, '*NHOH':-1, 'H2O':   -1, 'H2': 1/2},  # *NHOH → *NH2 + H2O + H+ + e-
            {'*NH3':  +1, '*NH2': -1, 'H2': 1/2},
            {'*':     +1, 'NH3':  -1, '*NH3':  -1},
        ],
    },
    "no_to_n2o": {
        "equilibrium_potential": -0.03,
        "n_electrons": 0,
        "steps": [
            {},
            {'*N2O2': +1, '*NO':   -2},                          # 2*NO → *N2O2 coupling
            {'*N2O':  +1, '*':     -1, '*N2O2': -1, 'O': +1},   # *N2O2 → *N2O + O(g)
            {'*':     +1, 'N2O':   -1, '*N2O':  -1},            # *N2O → N2O(g) + *
        ],
    },
    "no2_to_no": {
        "equilibrium_potential": 0.80,
        "n_electrons": 2,
        "steps": [
            {},
            {'*NO':   +1, '*':    -1, 'NO2':   -1, 'O': +1},    # NO2 → *NO + O(g)
            {'*OH':   +1, '*O':   -1, 'H2': 1/2},               # *O + H+ + e- → *OH
            {'*':     +1, 'H2O':  -1, '*OH':   -1, 'H2': 1/2},  # *OH + H+ + e- → H2O + *
        ],
    },
    "no3_to_nh3": {
        "equilibrium_potential": 0.88,
        "n_electrons": 8,
        "steps": [
            {},
            {'*NO2':  +1, '*':    -1, 'NO3':   -1, 'H2O': -1, 'H2': 1/2},  # NO3- + H+ + e- → *NO2 + H2O
            {'*NO':   +1, '*NO2': -1, 'H2O':   -1, 'H2': 1/2},             # *NO2 + H+ + e- → *NO + H2O
            {'*NOH':  +1, '*NO':  -1, 'H2': 1/2},
            {'*N':    +1, '*NOH': -1, 'H2O':   -1, 'H2': 1/2},
            {'*NH':   +1, '*N':   -1, 'H2': 1/2},
            {'*NH2':  +1, '*NH':  -1, 'H2': 1/2},
            {'*NH3':  +1, '*NH2': -1, 'H2': 1/2},
            {'*':     +1, 'NH3':  -1, '*NH3':  -1},
        ],
    },
    "no3_to_n2": {
        "equilibrium_potential": 1.40,
        "n_electrons": 5,
        "steps": [
            {},
            {'*NO2':  +1, '*':    -1, 'NO3':   -1, 'H2O': -1, 'H2': 1/2},
            {'*NO':   +1, '*NO2': -1, 'H2O':   -1, 'H2': 1/2},
            {'*N':    +1, '*NO':  -1, 'O': +1},                  # *NO → *N + O(g)
            {'*N2O2': +1, '*N':   -2},                           # 2*N → *N2O2 coupling
            {'*':     +1, 'N2':   -1, '*N2O2': -1},
        ],
    },
}


def calculate_noxrr_overpotential(adsorption_energies, pathway_name, method, func):
    """Calculate NOXRR overpotential using the CHE model.

    Parameters
    ----------
    adsorption_energies : dict
        DFT or ML energies of surface intermediates. Required keys depend
        on the pathway (e.g. '*NO', '*N', '*NH3', etc.).
    pathway_name : str
        One of: 'no_dissociative', 'no_to_nh3_noh', 'no_to_nh3_nhoh',
                'no_to_n2o', 'no2_to_no', 'no3_to_nh3', 'no3_to_n2'.
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
        Returns empty list [] for non-electrochemical pathways
        (no_dissociative, no_to_n2o) where no potential-determining step applies.
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
    if pathway_name not in NOXRR_PATHWAYS:
        raise ValueError(
            f"Unsupported NOXRR pathway: '{pathway_name}'. "
            f"Supported: {list(NOXRR_PATHWAYS.keys())}"
        )

    refs = load_references(method, func)
    local_energy = adsorption_energies.copy()
    local_energy.update(refs)
    # Derive atomic O from O2 reference (O2 → 2O, half-reaction reference)
    local_energy['O'] = local_energy['O2'] / 2

    pathway = NOXRR_PATHWAYS[pathway_name]
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

    # Final gas-phase reference state
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
