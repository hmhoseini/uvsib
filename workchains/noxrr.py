"""
NOXRR (NOx Reduction Reaction) overpotential calculator.
Supported pathways
------------------
no_dissociative  : NO → N2  via dissociative N coupling
no_to_nh3_noh    : NO → NH3 via *NOH (early N-O cleavage)
no_to_nh3_nhoh   : NO → NH3 via *NHOH → *NH2OH hydroxylamine (late N-O cleavage)
no_to_n2o        : NO → N2O via *N2O2 coupling
no2_to_no        : NO2 → NO  (deoxygenation)
no3_to_nh3       : NO3- → NH3
no3_to_n2        : NO3- → N2
"""

from uvsib.workchains.utils import load_zpe, che_overpotential

_ZPE = load_zpe("noxrr")

# Pathway definitions. See co2rr.py for convention notes: each dict is ONE
# elementary reaction (products +, reactants -); a consumed (H+ + e-) -> 'H2':
# -1/2, a released gas product -> +1, a consumed gas reactant -> -1. Released
# atomic O is referenced to 1/2 O2 ('O': +1, derived below from the O2 ref).
# Steps are summed individually, never differenced (utils.che_overpotential).
#
# NOTE: the non-electrochemical N-N coupling routes (no_dissociative, no_to_n2o,
# no3_to_n2) use sketch coupling steps (e.g. 2*N -> *N2O2) that are not
# atom-balanced in O; their signs are corrected here but treat their numbers as
# indicative only. The proton-coupled NH3 routes are quantitative.
NOXRR_PATHWAYS = {
    "no_dissociative": {
        "equilibrium_potential": -0.10,  # nominal (n_e=0, non-electrochemical; eta not a true overpotential)
        "n_electrons": 0,                # non-electrochemical coupling steps
        "steps": [
            {},
            {'*N':    +1, 'O':  +1, '*':  -1, 'NO': -1},        # * + NO → *N + O(g)  (O ref = 1/2 O2)
            {'N2':    +1, '*': +2, '*N': -2},                   # 2*N → N2(g) + 2*  (direct N-N coupling)
        ],
    },
    "no_to_nh3_noh": {
        "equilibrium_potential": 0.70,   # V vs RHE (NO/NH3, standard dGf)
        "n_electrons": 5,                # NO + 5(H+ + e-) → NH3 + H2O
        "steps": [
            {},
            {'*NOH':  +1, '*':   -1, 'NO': -1, 'H2': -1/2},     # NO + (H+ + e-) → *NOH
            {'*N':    +1, 'H2O': +1, '*NOH': -1, 'H2': -1/2},   # *NOH + (H+ + e-) → *N + H2O
            {'*NH':   +1, '*N':  -1, 'H2': -1/2},               # *N  + (H+ + e-) → *NH
            {'*NH2':  +1, '*NH': -1, 'H2': -1/2},               # *NH + (H+ + e-) → *NH2
            {'*NH3':  +1, '*NH2': -1, 'H2': -1/2},              # *NH2 + (H+ + e-) → *NH3
            {'*':     +1, 'NH3': +1, '*NH3': -1},               # *NH3 → NH3(g) + *
        ],
    },
    "no_to_nh3_nhoh": {
        "equilibrium_potential": 0.70,   # V vs RHE (NO/NH3, standard dGf)
        "n_electrons": 5,                # NO + 5(H+ + e-) → NH3 + H2O
        "steps": [
            {},
            {'*NOH':   +1, '*':   -1, 'NO': -1, 'H2': -1/2},    # NO + (H+ + e-) → *NOH
            {'*NHOH':  +1, '*NOH': -1, 'H2': -1/2},             # *NOH + (H+ + e-) → *NHOH
            {'*NH2OH': +1, '*NHOH': -1, 'H2': -1/2},            # *NHOH + (H+ + e-) → *NH2OH (hydroxylamine)
            {'*NH2':   +1, 'H2O': +1, '*NH2OH': -1, 'H2': -1/2}, # *NH2OH + (H+ + e-) → *NH2 + H2O (late N-O cleavage)
            {'*NH3':   +1, '*NH2': -1, 'H2': -1/2},             # *NH2 + (H+ + e-) → *NH3
            {'*':      +1, 'NH3': +1, '*NH3': -1},              # *NH3 → NH3(g) + * (chemical desorption)
        ],
    },
    "no_to_n2o": {
        "equilibrium_potential": -0.03,  # nominal (n_e=0, non-electrochemical; eta not a true overpotential)
        "n_electrons": 0,
        "steps": [
            {},
            {'*N2O2': +1, '*': +1, '*NO': -2},                  # 2*NO → *N2O2 + * (cis-dimer, frees a site)
            {'*N2O':  +1, 'O':  +1, '*N2O2': -1},               # *N2O2 → *N2O + O(g)  (O ref = 1/2 O2)
            {'*':     +1, 'N2O': +1, '*N2O': -1},               # *N2O → N2O(g) + *
        ],
    },
    "no2_to_no": {
        "equilibrium_potential": 1.05,   # V vs RHE (NO2/NO, standard dGf)
        "n_electrons": 2,
        "steps": [
            {},
            {'*NO': +1, 'O':  +1, '*': -1, 'NO2': -1},          # NO2 → *NO + O(g)
            {'*OH': +1, '*O':  -1, 'H2': -1/2},                 # *O + (H+ + e-) → *OH
            {'*':   +1, 'H2O': +1, '*OH': -1, 'H2': -1/2},      # *OH + (H+ + e-) → H2O + *
        ],
    },
    "no3_to_nh3": {
        # TODO(fix later): 0.88 is the NO3-/NH4+ standard potential, but this
        # pathway's last step releases NH3(g). For consistency with the product
        # (and the pin_total closure, G_total=-8*U_eq) recompute U_eq for the
        # NO3- -> NH3(g) couple (~0.80 V from standard dGf; ~0.69 V aqueous-NH3
        # per benchmarks/overpotential_refs/RECORDS.md) OR change the last step
        # to produce NH4+. Mismatch biases this pathway's pinned eta by ~0.1 V.
        "equilibrium_potential": 0.88,   # V vs RHE (NO3-/NH4+, standard) -- see TODO
        "n_electrons": 8,
        "steps": [
            {},
            {'*NO2': +1, 'H2O': +1, '*': -1, 'NO3': -1, 'H2': -1.0},  # NO3 + 2(H+ + e-) → *NO2 + H2O (2 e-)
            {'*NO':  +1, 'H2O': +1, '*NO2': -1, 'H2': -1.0},          # *NO2 + 2(H+ + e-) → *NO + H2O (2 e-)
            {'*NOH': +1, '*NO':  -1, 'H2': -1/2},
            {'*N':   +1, 'H2O': +1, '*NOH': -1, 'H2': -1/2},
            {'*NH':  +1, '*N':   -1, 'H2': -1/2},
            {'*NH2': +1, '*NH':  -1, 'H2': -1/2},
            {'*NH3': +1, '*NH2': -1, 'H2': -1/2},
            {'*':    +1, 'NH3': +1, '*NH3': -1},
        ],
    },
    "no3_to_n2": {
        "equilibrium_potential": 1.25,   # V vs RHE (NO3-/N2, standard dGf)
        "n_electrons": 5,
        "steps": [
            {},
            {'*NO2': +1, 'H2O': +1, '*': -1, 'NO3': -1, 'H2': -1.0},  # NO3 + 2(H+ + e-) → *NO2 + H2O (2 e-)
            {'*NO':  +1, 'H2O': +1, '*NO2': -1, 'H2': -1.0},          # *NO2 + 2(H+ + e-) → *NO + H2O (2 e-)
            {'*N':   +1, 'O':  +1, '*NO': -1},                  # *NO → *N + O(g)  (O ref = 1/2 O2)
            {'N2':   +1, '*': +2, '*N': -2},                    # 2*N → N2(g) + 2*  (direct N-N coupling)
        ],
    },
}


def calculate_noxrr_overpotential(adsorption_energies, pathway_name):
    """Calculate NOXRR overpotential using the CHE model.

    Parameters
    ----------
    adsorption_energies : dict
        DFT or ML energies of surface intermediates. Required keys depend
        on the pathway (e.g. '*NO', '*N', '*NH3', etc.).
    pathway_name : str
        One of: 'no_dissociative', 'no_to_nh3_noh', 'no_to_nh3_nhoh',
                'no_to_n2o', 'no2_to_no', 'no3_to_nh3', 'no3_to_n2'.

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
    if pathway_name not in NOXRR_PATHWAYS:
        raise ValueError(
            f"Unsupported NOXRR pathway: '{pathway_name}'. "
            f"Supported: {list(NOXRR_PATHWAYS.keys())}"
        )

    # Gas-phase references (NO, NO2, NO3, N2, N2O, NH3, O2, H2, H2O) are
    # computed per-molecule from molecular_references/*.vasp and arrive inside
    # adsorption_energies alongside the surface intermediates and clean slab.
    # ZPE is added inside che_overpotential, so the stored energies are raw
    # electronic.
    local_energy = adsorption_energies.copy()
    # Derive atomic O from the O2 reference (O2 → 2O half-reaction). Only the
    # O-releasing pathways need it, so guard the derivation when O2 is absent.
    if 'O2' in local_energy:
        local_energy['O'] = local_energy['O2'] / 2

    pathway = NOXRR_PATHWAYS[pathway_name]
    return che_overpotential(
        pathway["steps"], local_energy, _ZPE,
        pathway["equilibrium_potential"], reduction=True,
        n_electrons=pathway["n_electrons"], pin_total=True,
    )
