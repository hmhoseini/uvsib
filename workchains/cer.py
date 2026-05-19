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

CER electrochemical steps consume Cl- (not H+); the Cl2 reference must
exist in references.yaml under the chosen method/functional for the
calculation to succeed.
"""

import numpy as np
from uvsib.workchains.utils import load_references, load_zpe

_ZPE = load_zpe("cer")

# Pathway definitions. See co2rr.py for convention notes. Cl2: +1/2 plays
# the role of H2: +1/2 from the proton-coupled reactions, representing one
# (Cl- + e-) pair transferred against the Cl-/Cl2 reference couple.
CER_PATHWAYS = {
    "volmer_tafel": {
        "equilibrium_potential": 1.36,    # V vs SHE (Cl-/Cl2 couple)
        "n_electrons": 2,
        "steps": [
            {},
            {'*Cl': +1, '*':   -1, 'Cl2': 1/2},                              # Volmer: * + Cl- → *Cl + e-
            {'*':   +2, '*Cl': -2, 'Cl2': -1},                               # Tafel: 2 *Cl → Cl2(g) + 2 * (chemical, 2-site)
        ],
    },
    "volmer_heyrovsky": {
        "equilibrium_potential": 1.36,
        "n_electrons": 2,
        "steps": [
            {},
            {'*Cl': +1, '*':   -1, 'Cl2': 1/2},                              # Volmer: * + Cl- → *Cl + e-
            {'*':   +1, '*Cl': -1, 'Cl2': -1/2},                             # Heyrovsky: *Cl + Cl- → Cl2(g) + * + e-  (-1 gas + 1/2 electron)
        ],
    },
    "krishtalik": {
        "equilibrium_potential": 1.36,
        "n_electrons": 2,
        "steps": [
            {},
            {'*OCl': +1, '*O':   -1, 'Cl2': 1/2},                            # *O + Cl- → *OCl + e-
            {'*O':   +1, '*OCl': -1, 'Cl2': -1/2},                           # *OCl + Cl- → Cl2(g) + *O + e-
        ],
    },
}


def calculate_cer_overpotential(adsorption_energies, pathway_name, method, func):
    """Calculate CER overpotential using the CHE-equivalent model.

    Parameters
    ----------
    adsorption_energies : dict
        DFT or ML energies of surface intermediates. Required keys depend
        on the pathway: '*Cl' for 'volmer_tafel' / 'volmer_heyrovsky';
        '*OCl' and '*O' for 'krishtalik'.
    pathway_name : str
        One of: 'volmer_tafel', 'volmer_heyrovsky', 'krishtalik'.
    method : str
        'dft' or ML model name (e.g. 'uPET', 'mace', 'mattersim').
    func : str
        Functional / reference set, e.g. 'r2SCAN'.

    Returns
    -------
    overpotential : float
        Thermodynamic overpotential in V (positive = more difficult).
    dg_steps : list[float]
        ΔG per elementary step at U = 0 V vs Cl-/Cl2 reference (eV).
    dg_cumulative : list[float]
        Cumulative free energies at U = equilibrium_potential (eV).

    Raises
    ------
    ValueError
        If pathway_name is not supported.
    NotImplementedError
        If the method/func combination has no defined references.
    KeyError
        If a required adsorbate key is missing from adsorption_energies
        or 'Cl2' is missing from references.yaml.
    """
    if pathway_name not in CER_PATHWAYS:
        raise ValueError(
            f"Unsupported CER pathway: '{pathway_name}'. "
            f"Supported: {list(CER_PATHWAYS.keys())}"
        )

    refs = load_references(method, func)
    local_energy = adsorption_energies.copy()
    local_energy.update(refs)

    pathway = CER_PATHWAYS[pathway_name]
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
