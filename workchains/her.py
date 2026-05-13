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

import numpy as np
from uvsib.workchains.utils import load_references, load_zpe

_ZPE = load_zpe("her")

# Pathway definitions. See co2rr.py for convention notes.
HER_PATHWAYS = {
    "volmer_tafel": {
        "equilibrium_potential": 0.00,   # V vs RHE (HER reference)
        "n_electrons": 2,
        "steps": [
            {},
            {'*H': +1, '*':  -1, 'H2': 1/2},                       # Volmer: * + H+ + e- → *H
            {'*':  +2, '*H': -2, 'H2': -1},                        # Tafel: 2 *H → H2(g) + 2 * (chemical, 2-site)
        ],
    },
    "volmer_heyrovsky": {
        "equilibrium_potential": 0.00,
        "n_electrons": 2,
        "steps": [
            {},
            {'*H': +1, '*':  -1, 'H2': 1/2},                       # Volmer: * + H+ + e- → *H
            {'*':  +1, '*H': -1, 'H2': -1/2},                      # Heyrovsky: *H + H+ + e- → H2(g) + *  (-1 gas + 1/2 electron)
        ],
    },
}


def calculate_her_overpotential(adsorption_energies, pathway_name, method, func):
    """Calculate HER overpotential using the CHE model.

    Parameters
    ----------
    adsorption_energies : dict
        DFT or ML energies of surface intermediates. Required key: '*H'.
    pathway_name : str
        One of: 'volmer_tafel', 'volmer_heyrovsky'.
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
        If '*H' is missing from adsorption_energies.
    """
    if pathway_name not in HER_PATHWAYS:
        raise ValueError(
            f"Unsupported HER pathway: '{pathway_name}'. "
            f"Supported: {list(HER_PATHWAYS.keys())}"
        )

    refs = load_references(method, func)
    local_energy = adsorption_energies.copy()
    local_energy.update(refs)

    pathway = HER_PATHWAYS[pathway_name]
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
