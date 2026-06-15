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

from uvsib.workchains.utils import load_zpe, che_overpotential

_ZPE = load_zpe("nrr")

# Pathway definitions. See co2rr.py for convention notes: each dict is ONE
# elementary reaction (products +, reactants -); consumed (H+ + e-) -> 'H2':
# -1/2, released NH3 -> +1, consumed N2 -> -1. Steps are summed individually,
# never differenced (utils.che_overpotential).
NRR_PATHWAYS = {
    "distal": {
        "equilibrium_potential": 0.092,   # V vs RHE (N2/NH3 equilibrium at 1 M, 1 bar)
        "n_electrons": 6,
        "steps": [
            {},
            {'*N2_ads': +1, '*':       -1, 'N2': -1},                        # N2 + * → *N2 (chemical adsorption)
            {'*NNH':    +1, '*N2_ads': -1, 'H2': -1/2},                      # *N2 + (H+ + e-) → *NNH
            {'*NNH2':   +1, '*NNH':    -1, 'H2': -1/2},                      # *NNH + (H+ + e-) → *NNH2
            {'*N':      +1, 'NH3': +1, '*NNH2': -1, 'H2': -1/2},             # *NNH2 + (H+ + e-) → *N + NH3(g)
            {'*NH':     +1, '*N':      -1, 'H2': -1/2},                      # *N + (H+ + e-) → *NH
            {'*NH2':    +1, '*NH':     -1, 'H2': -1/2},                      # *NH + (H+ + e-) → *NH2
            {'*':       +1, 'NH3': +1, '*NH2': -1, 'H2': -1/2},              # *NH2 + (H+ + e-) → NH3(g) + *
        ],
    },
    "alternating": {
        "equilibrium_potential": 0.092,
        "n_electrons": 6,
        "steps": [
            {},
            {'*N2_ads': +1, '*':        -1, 'N2': -1},                       # N2 + * → *N2
            {'*NNH':    +1, '*N2_ads':  -1, 'H2': -1/2},                     # *N2 + (H+ + e-) → *NNH
            {'*NHNH':   +1, '*NNH':     -1, 'H2': -1/2},                     # *NNH + (H+ + e-) → *NHNH (H to near N)
            {'*NHNH2':  +1, '*NHNH':    -1, 'H2': -1/2},                     # *NHNH + (H+ + e-) → *NHNH2
            {'*N2H4':   +1, '*NHNH2':   -1, 'H2': -1/2},                     # *NHNH2 + (H+ + e-) → *N2H4 (hydrazine)
            {'*NH2':    +1, 'NH3': +1, '*N2H4': -1, 'H2': -1/2},             # *N2H4 + (H+ + e-) → *NH2 + NH3(g)
            {'*NH3':    +1, '*NH2':     -1, 'H2': -1/2},                     # *NH2 + (H+ + e-) → *NH3
            {'*':       +1, 'NH3': +1, '*NH3': -1},                          # *NH3 → NH3(g) + * (chemical desorption)
        ],
    },
    "dissociative": {
        "equilibrium_potential": 0.092,
        "n_electrons": 6,
        "steps": [
            {},
            {'*N':   +2, '*':    -2, 'N2': -1},                              # N2 + 2* → 2 *N (chemical scission)
            {'*NH':  +1, '*N':   -1, 'H2': -1/2},                            # *N + (H+ + e-) → *NH
            {'*NH2': +1, '*NH':  -1, 'H2': -1/2},                            # *NH + (H+ + e-) → *NH2
            {'*':    +1, 'NH3': +1, '*NH2': -1, 'H2': -1/2},                 # *NH2 + (H+ + e-) → NH3(g) + *
        ],
    },
}


def calculate_nrr_overpotential(adsorption_energies, pathway_name):
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
        adsorption_energies (e.g. '*N2_ads', '*N', 'N2', 'NH3').
    """
    if pathway_name not in NRR_PATHWAYS:
        raise ValueError(
            f"Unsupported NRR pathway: '{pathway_name}'. "
            f"Supported: {list(NRR_PATHWAYS.keys())}"
        )

    # Gas-phase references (N2, NH3, H2, H2O) are computed per-molecule from
    # molecular_references/*.vasp and arrive inside adsorption_energies
    # alongside the surface intermediates and the clean slab. ZPE is added
    # inside che_overpotential, so the stored energies are raw electronic.
    pathway = NRR_PATHWAYS[pathway_name]
    return che_overpotential(
        pathway["steps"], adsorption_energies, _ZPE,
        pathway["equilibrium_potential"], reduction=True,
    )
