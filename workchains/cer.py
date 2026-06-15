"""
CER (Chlorine Evolution Reaction) overpotential calculator.

Uses the CHE-equivalent framework with the Cl-/Cl2 couple as reference
(E° = 1.36 V vs SHE in 1 M HCl). Each electrochemical step transfers
(Cl- + e-), bookkept via Cl2: +1/2 in analogy to H2: +1/2 in OER/CO2RR.

Supported pathways
------------------
volmer_tafel     : * + Cl- → *Cl,   then 2 *Cl → Cl2(g) + 2 *
volmer_heyrovsky : * + Cl- → *Cl,   then *Cl + Cl- → Cl2(g) + *
krishtalik       : *O + Cl- → *OCl, then *OCl + Cl- → Cl2(g) + *O
                   (O-covered oxide route on RuO2(110), IrO2(110), Co3O4)

CER electrochemical steps consume Cl- (not H+); the Cl2 gas-phase reference
must be present in adsorption_energies (computed per-molecule from
molecular_references/cl2.vasp) for the calculation to succeed.
"""

from uvsib.workchains.utils import load_zpe, che_overpotential

_ZPE = load_zpe("cer")

# Chlorine evolution is an OXIDATION referenced to the Cl-/Cl2 couple. Each
# dict is ONE elementary reaction (products +, reactants -). One consumed
# (Cl- - e-) is referenced to 1/2 Cl2 just as (H+ + e-) is to 1/2 H2 in OER,
# so a step that consumes Cl- carries 'Cl2': -1/2 and one that releases a Cl2
# molecule carries +1/2 (net of the consumed Cl-). Steps are summed
# individually, never differenced (utils.che_overpotential).
#
# Overpotential is taken vs the Cl-/Cl2 reference itself, where the equilibrium
# offset is 0 (the 1.36 V below is only the Cl-/Cl2 potential vs SHE, used for
# reporting -- it is NOT subtracted in the limiting-potential overpotential,
# exactly as OER subtracts 1.23 V because RHE != the O2/H2O couple but CER's
# reference IS its own couple).
CER_U_SHE = 1.36  # V vs SHE, Cl-/Cl2 couple (documentation only)
CER_PATHWAYS = {
    "volmer_tafel": {
        "equilibrium_potential": 1.36,    # V vs SHE (Cl-/Cl2 couple)
        "n_electrons": 2,
        "steps": [
            {},
            {'*Cl': +1, '*':   -1, 'Cl2': -1/2},                            # Volmer: * + Cl- → *Cl (+ e-)
            {'*':   +1, '*Cl': -1, 'Cl2': +1/2},                            # Tafel (per *Cl): *Cl → 1/2 Cl2(g) + *
        ],
    },
    "volmer_heyrovsky": {
        "equilibrium_potential": 1.36,
        "n_electrons": 2,
        "steps": [
            {},
            {'*Cl': +1, '*':   -1, 'Cl2': -1/2},                            # Volmer: * + Cl- → *Cl (+ e-)
            {'*':   +1, '*Cl': -1, 'Cl2': +1/2},                            # Heyrovsky: *Cl + Cl- → Cl2(g) + * (+ e-)
        ],
    },
    "krishtalik": {
        "equilibrium_potential": 1.36,
        "n_electrons": 2,
        "steps": [
            {},
            {'*OCl': +1, '*O':   -1, 'Cl2': -1/2},                          # *O + Cl- → *OCl (+ e-)
            {'*O':   +1, '*OCl': -1, 'Cl2': +1/2},                          # *OCl + Cl- → Cl2(g) + *O (+ e-)
        ],
    },
}


def calculate_cer_overpotential(adsorption_energies, pathway_name):
    """Calculate CER overpotential using the CHE-equivalent model.

    Parameters
    ----------
    adsorption_energies : dict
        DFT or ML energies of surface intermediates. Required keys depend
        on the pathway: '*Cl' for 'volmer_tafel' / 'volmer_heyrovsky';
        '*OCl' and '*O' for 'krishtalik'.
    pathway_name : str
        One of: 'volmer_tafel', 'volmer_heyrovsky', 'krishtalik'.

    Returns
    -------
    overpotential : float
        Thermodynamic overpotential in V (positive = more difficult).
    dg_steps : list[float]
        ΔG per elementary step at U = 0 V vs the Cl-/Cl2 reference (eV).
    dg_cumulative : list[float]
        Cumulative free energies [0, ΔG1, ΔG1+ΔG2, ...] at U = 0 V (eV).

    Raises
    ------
    ValueError
        If pathway_name is not supported.
    KeyError
        If a required adsorbate or gas-phase reference key is missing from
        adsorption_energies (e.g. '*Cl', '*OCl', '*O', or 'Cl2').
    """
    if pathway_name not in CER_PATHWAYS:
        raise ValueError(
            f"Unsupported CER pathway: '{pathway_name}'. "
            f"Supported: {list(CER_PATHWAYS.keys())}"
        )

    # Gas-phase references (Cl2, ...) are computed per-molecule from
    # molecular_references/*.vasp and arrive inside adsorption_energies
    # alongside the surface intermediates and the clean slab. ZPE is added
    # inside che_overpotential, so the stored energies are raw electronic.
    # Oxidation, referenced to the Cl-/Cl2 couple -> equilibrium offset 0
    # (CER_U_SHE = 1.36 V is the SHE conversion, used only for reporting).
    pathway = CER_PATHWAYS[pathway_name]
    return che_overpotential(
        pathway["steps"], adsorption_energies, _ZPE,
        equilibrium_potential=0.0, reduction=False,
    )
