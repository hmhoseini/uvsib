"""
CO2RR (CO2 Reduction Reaction) overpotential calculator.
Supported pathways
------------------
co2_to_co     : CO2 → CO  (2e-)
co2_to_hcooh  : CO2 → HCOOH  (2e-)
co_to_ch4     : CO → CH4  (6e-, requires '*CO' in adsorption_energies)
co_to_ch3oh   : CO → CH3OH  (4e-, requires '*CO' in adsorption_energies)
co2_to_ch4    : CO2 → CH4  (8e-, full chain = co2_to_co + co_to_ch4)
co2_to_ch3oh  : CO2 → CH3OH  (6e-, full chain = co2_to_co + co_to_ch3oh)
co2_to_c2h4   : CO2 → C2H4  (8e-, C-C coupling via *OCCO)
"""

from uvsib.workchains.utils import load_zpe, che_overpotential

_ZPE = load_zpe("co2rr")

# Pathway definitions: each step is a dict of {species: stoichiometric_coefficient}
# for that ONE elementary reaction. Products = positive, reactants = negative.
# CHE bookkeeping: one consumed (H+ + e-) -> 'H2': -1/2; a released gas product
# (H2O, CH4, ...) -> +1; a consumed gas reactant (CO2) -> -1. The leading empty
# {} marks the reference state and is ignored. Each step's free energy is summed
# on its own (see utils.che_overpotential) -- NOT differenced. n_electrons is
# informational only.
CO2RR_PATHWAYS = {
    "co2_to_co": {
        "equilibrium_potential": -0.11,  # V vs RHE (CO2/CO, standard dGf)
        "n_electrons": 2,
        "steps": [
            {},
            {'*COOH': +1, '*':   -1, 'CO2': -1, 'H2': -1/2},      # CO2 + (H+ + e-) → *COOH
            {'*CO':   +1, 'H2O': +1, '*COOH': -1, 'H2': -1/2},    # *COOH + (H+ + e-) → *CO + H2O
        ],
    },
    "co2_to_hcooh": {
        "equilibrium_potential": -0.12,  # V vs RHE (CO2/HCOOH)
        "n_electrons": 2,
        "steps": [
            {},
            {'*OCHO': +1, '*':     -1, 'CO2': -1, 'H2': -1/2},    # CO2 + (H+ + e-) → *OCHO
            {'*':     +1, 'HCOOH': +1, '*OCHO': -1, 'H2': -1/2},  # *OCHO + (H+ + e-) → HCOOH(g) + *
        ],
    },
    "co_to_ch4": {
        "equilibrium_potential": 0.26,   # V vs RHE (CO/CH4)
        "n_electrons": 6,
        "steps": [
            {},
            {'*CHO':  +1, '*CO':   -1, 'H2': -1/2},               # *CO + (H+ + e-) → *CHO
            {'*CHOH': +1, '*CHO':  -1, 'H2': -1/2},               # *CHO + (H+ + e-) → *CHOH
            {'*CH':   +1, 'H2O':   +1, '*CHOH': -1, 'H2': -1/2},  # *CHOH + (H+ + e-) → *CH + H2O
            {'*CH2':  +1, '*CH':   -1, 'H2': -1/2},               # *CH + (H+ + e-) → *CH2
            {'*CH3':  +1, '*CH2':  -1, 'H2': -1/2},               # *CH2 + (H+ + e-) → *CH3
            {'*':     +1, 'CH4':   +1, '*CH3': -1, 'H2': -1/2},   # *CH3 + (H+ + e-) → CH4(g) + *
        ],
    },
    "co_to_ch3oh": {
        "equilibrium_potential": 0.09,   # V vs RHE (CO/CH3OH)
        "n_electrons": 4,
        "steps": [
            {},
            {'*CHO':   +1, '*CO':    -1, 'H2': -1/2},
            {'*CHOH':  +1, '*CHO':   -1, 'H2': -1/2},
            {'*CH2OH': +1, '*CHOH':  -1, 'H2': -1/2},
            {'*':      +1, 'CH3OH':  +1, '*CH2OH': -1, 'H2': -1/2},  # *CH2OH + (H+ + e-) → CH3OH(g) + *
        ],
    },
    # Full CO2 -> CH4 chain: co2_to_co (CO2 -> *COOH -> *CO) spliced onto
    # co_to_ch4 (*CO -> ... -> CH4). Mirrors the 'co2_to_ch4' ReactionPathway
    # in codes/files/adsorbates.py. Starts from CO2(g), so '*CO' need NOT be
    # supplied by the caller (unlike co_to_ch4); the *CO state is produced
    # internally at step 2.
    # U_eq = standard equilibrium potential of CO2 -> CH4 (8 e-) from dGf.
    "co2_to_ch4": {
        "equilibrium_potential": 0.17,   # V vs RHE (CO2/CH4, standard dGf)
        "n_electrons": 8,
        "steps": [
            {},
            {'*COOH': +1, '*':   -1, 'CO2': -1, 'H2': -1/2},      # CO2 + (H+ + e-) → *COOH
            {'*CO':   +1, 'H2O': +1, '*COOH': -1, 'H2': -1/2},    # *COOH + (H+ + e-) → *CO + H2O
            {'*CHO':  +1, '*CO':   -1, 'H2': -1/2},               # *CO + (H+ + e-) → *CHO
            {'*CHOH': +1, '*CHO':  -1, 'H2': -1/2},               # *CHO + (H+ + e-) → *CHOH
            {'*CH':   +1, 'H2O':   +1, '*CHOH': -1, 'H2': -1/2},  # *CHOH + (H+ + e-) → *CH + H2O
            {'*CH2':  +1, '*CH':   -1, 'H2': -1/2},               # *CH + (H+ + e-) → *CH2
            {'*CH3':  +1, '*CH2':  -1, 'H2': -1/2},               # *CH2 + (H+ + e-) → *CH3
            {'*':     +1, 'CH4':   +1, '*CH3': -1, 'H2': -1/2},   # *CH3 + (H+ + e-) → CH4(g) + *
        ],
    },
    # Full CO2 -> CH3OH chain: co2_to_co spliced onto co_to_ch3oh. Mirrors the
    # 'co2_to_ch3oh' ReactionPathway in codes/files/adsorbates.py.
    # U_eq = standard equilibrium potential of CO2 -> CH3OH (6 e-) from dGf.
    "co2_to_ch3oh": {
        "equilibrium_potential": 0.03,   # V vs RHE (CO2/CH3OH, standard dGf)
        "n_electrons": 6,
        "steps": [
            {},
            {'*COOH':  +1, '*':   -1, 'CO2': -1, 'H2': -1/2},     # CO2 + (H+ + e-) → *COOH
            {'*CO':    +1, 'H2O': +1, '*COOH': -1, 'H2': -1/2},   # *COOH + (H+ + e-) → *CO + H2O
            {'*CHO':   +1, '*CO':    -1, 'H2': -1/2},
            {'*CHOH':  +1, '*CHO':   -1, 'H2': -1/2},
            {'*CH2OH': +1, '*CHOH':  -1, 'H2': -1/2},
            {'*':      +1, 'CH3OH':  +1, '*CH2OH': -1, 'H2': -1/2},  # *CH2OH + (H+ + e-) → CH3OH(g) + *
        ],
    },
    # NOTE: coarse C-C coupling route. It is now slab- and atom-balanced, but
    # the *OCCO -> *CCHO -> C2H4 tail collapses several (H+ + e-) transfers into
    # two lumped multi-electron steps (released O routed to H2O), so its
    # overpotential is QUALITATIVE only -- the lumped steps are not per-electron
    # resolved. Mirrors the "collapsed / chemical" bookkeeping in the runner.
    "co2_to_c2h4": {
        "equilibrium_potential": 0.08,   # V vs RHE (CO2/C2H4)
        "n_electrons": 8,
        "steps": [
            {},
            {'*COOH': +1, '*':   -1, 'CO2': -1, 'H2': -1/2},
            {'*CO':   +1, 'H2O': +1, '*COOH': -1, 'H2': -1/2},
            {'*OCCO': +1, '*': +1, '*CO': -2},                   # 2*CO → *OCCO + * (C-C coupling, frees a site)
            {'*CCHO': +1, 'H2O': +1, '*OCCO': -1, 'H2': -1.5},   # *OCCO + 3(H+ + e-) → *CCHO + H2O (lumped)
            {'*':     +1, 'C2H4': +1, 'H2O': +1, '*CCHO': -1, 'H2': -2.5},  # *CCHO + 5(H+ + e-) → C2H4(g) + H2O + * (lumped)
        ],
    },
}


def calculate_co2rr_overpotential(adsorption_energies, pathway_name):
    """Calculate CO2RR overpotential using the CHE model.

    Parameters
    ----------
    adsorption_energies : dict
        DFT or ML energies of surface intermediates. Required keys depend
        on the pathway. For pathways starting from *CO ('co_to_ch4',
        'co_to_ch3oh'), '*CO' must be supplied by the caller.
    pathway_name : str
        One of: 'co2_to_co', 'co2_to_hcooh', 'co_to_ch4',
                'co_to_ch3oh', 'co2_to_ch4', 'co2_to_ch3oh', 'co2_to_c2h4'.

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
        adsorption_energies.
    """
    if pathway_name not in CO2RR_PATHWAYS:
        raise ValueError(
            f"Unsupported CO2RR pathway: '{pathway_name}'. "
            f"Supported: {list(CO2RR_PATHWAYS.keys())}"
        )

    # Gas-phase references (CO2, H2, H2O, CH4, ...) are computed per-molecule
    # from molecular_references/*.vasp and arrive inside adsorption_energies
    # alongside the surface intermediates and the clean slab. ZPE is added
    # uniformly below via _ZPE, so the stored energies are raw electronic.
    pathway = CO2RR_PATHWAYS[pathway_name]
    return che_overpotential(
        pathway["steps"], adsorption_energies, _ZPE,
        pathway["equilibrium_potential"], reduction=True,
        n_electrons=pathway["n_electrons"], pin_total=True,
    )
