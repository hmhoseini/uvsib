"""
HER (Hydrogen Evolution Reaction) overpotential calculator.

Uses the computational hydrogen electrode (CHE) model. The single
intermediate is *H; the Sabatier volcano apex is |ΔG_H*| ≈ 0.

Supported pathways
------------------
volmer_tafel      : * + H+ + e- → *H,  then 2 *H → H2(g) + 2 *
volmer_heyrovsky  : * + H+ + e- → *H,  then *H + H+ + e- → H2(g) + *

Both pathways share the same thermodynamic profile (only *H matters in
CHE); they differ in the kinetic split between the two electron transfers.
"""

from uvsib.workchains.utils import load_zpe, che_overpotential

_ZPE = load_zpe("her")

# Pathway definitions. See co2rr.py for convention notes: each dict is ONE
# elementary reaction (products +, reactants -); a consumed (H+ + e-) -> 'H2':
# -1/2, a released H2 molecule -> +1. Steps are summed individually, never
# differenced (utils.che_overpotential). U_eq = 0 (HER is defined at RHE), so
# the overpotential reduces to |ΔG_H*| -- both routes share this thermodynamic
# profile (only *H matters in CHE). The desorption is written per *H so the
# 2-site Tafel and 1-e Heyrovsky give the same single-site PDS.
HER_PATHWAYS = {
    "volmer_tafel": {
        "equilibrium_potential": 0.00,   # V vs RHE (HER reference)
        "n_electrons": 2,
        "steps": [
            {},
            {'*H': +1, '*':  -1, 'H2': -1/2},                     # Volmer: * + (H+ + e-) → *H
            {'*':  +1, '*H': -1, 'H2': +1/2},                     # Tafel (per *H): *H → 1/2 H2(g) + *
        ],
    },
    "volmer_heyrovsky": {
        "equilibrium_potential": 0.00,
        "n_electrons": 2,
        "steps": [
            {},
            {'*H': +1, '*':  -1, 'H2': -1/2},                     # Volmer: * + (H+ + e-) → *H
            {'*':  +1, '*H': -1, 'H2': +1/2},                     # Heyrovsky: *H + (H+ + e-) → H2(g) + * (+1 gas - 1/2 e-)
        ],
    },
}


def calculate_her_overpotential(adsorption_energies, pathway_name):
    """Calculate HER overpotential using the CHE model.

    Parameters
    ----------
    adsorption_energies : dict
        DFT or ML energies of surface intermediates. Required key: '*H'.
    pathway_name : str
        One of: 'volmer_tafel', 'volmer_heyrovsky'.

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
        adsorption_energies (e.g. '*H' or 'H2').
    """
    if pathway_name not in HER_PATHWAYS:
        raise ValueError(
            f"Unsupported HER pathway: '{pathway_name}'. "
            f"Supported: {list(HER_PATHWAYS.keys())}"
        )

    # Gas-phase references (H2, ...) are computed per-molecule from
    # molecular_references/*.vasp and arrive inside adsorption_energies
    # alongside the surface intermediates and the clean slab. ZPE is added
    # inside che_overpotential, so the stored energies are raw electronic.
    pathway = HER_PATHWAYS[pathway_name]
    return che_overpotential(
        pathway["steps"], adsorption_energies, _ZPE,
        pathway["equilibrium_potential"], reduction=True,
    )
