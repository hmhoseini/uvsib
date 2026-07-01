"""
OER (Oxygen Evolution Reaction) overpotential calculator.

Uses the computational hydrogen electrode (CHE) model for the 4-electron
acidic mechanism: 2H2O → O2 + 4H+ + 4e-

Gas-phase references (H2, H2O) arrive inside adsorption_energies as raw
per-molecule electronic energies (computed from molecular_references/*.vasp);
their ZPE is added here from zpe_corrections.yaml, matching the convention
used by the other reaction calculators.
"""

from uvsib.workchains.utils import load_zpe

# ---------------------------------------------------------------------------
# Module-level constants (loaded once at import)
# ---------------------------------------------------------------------------

_ZPE = load_zpe("oer")

# Thermodynamic total free energy for O2 evolution (4 × 1.23 V)
G_OER_TOTAL = 4.92  # eV

# Equilibrium potential for OER
U_OER_EQ = 1.23  # V vs RHE


def calculate_oer_overpotential(adsorption_energies, pathway_name):
    """Calculate OER overpotential using the CHE model (4e acidic mechanism).

    Parameters
    ----------
    adsorption_energies : dict
        DFT or ML energies of surface intermediates. Required keys:
        '*', '*OH', '*O', '*OOH'.
    pathway_name : str
        Currently only '4e_mechanism' is supported. Parameter is accepted
        for API consistency with CO2RR/NOXRR functions.

    Returns
    -------
    overpotential : float
        Thermodynamic overpotential in V (positive = more difficult).
    dg_steps : list[float]
        [ΔG1, ΔG2, ΔG3, ΔG4] at U = 0 V vs RHE (eV).
    dg_cumulative : list[float]
        Cumulative free energies [0, ΔG1, ΔG1+ΔG2, ..., 4.92] at U = 0 V (eV).

    Raises
    ------
    KeyError
        If a required adsorbate or gas-phase reference key is missing from
        adsorption_energies (e.g. '*', '*OH', '*O', '*OOH', 'H2', 'H2O').
    """
    # Gas-phase references (H2, H2O) are computed per-molecule from
    # molecular_references/*.vasp and arrive inside adsorption_energies
    # alongside the surface intermediates and the clean slab.
    local_energy = adsorption_energies

    zpe = _ZPE
    DELTA_OH  = zpe['*OH']
    DELTA_O   = zpe['*O']
    DELTA_OOH = zpe['*OOH']

    # The stored gas energies are raw electronic; add their full ZPE here so
    # OER matches the other reactions, which add _ZPE per species. O2 is never
    # referenced directly -- step 4 uses the energy-conserving closure below
    # (G_OER_TOTAL), which avoids DFT's O2 overbinding error.
    half_H2 = 0.5 * (local_energy['H2'] + zpe['H2'])
    E_H2O   = local_energy['H2O'] + zpe['H2O']
    E_star  = local_energy['*']
    E_OH    = local_energy['*OH']
    E_O     = local_energy['*O']
    E_OOH   = local_energy['*OOH']

    # Adsorption energies relative to clean surface
    A = E_OH  - E_star   # ΔE(*OH)
    B = E_O   - E_star   # ΔE(*O)
    C = E_OOH - E_star   # ΔE(*OOH)

    # Elementary step free energies at U = 0 V vs RHE
    # Step 1: H2O + * → *OH + H+ + e-
    dG1 = A - E_H2O + half_H2 + DELTA_OH
    # Step 2: *OH → *O + H+ + e-
    dG2 = (B - A) + half_H2 + (DELTA_O - DELTA_OH)
    # Step 3: H2O + *O → *OOH + H+ + e-
    dG3 = (C - B) - E_H2O + half_H2 + (DELTA_OOH - DELTA_O)
    # Step 4: *OOH → O2 + * + H+ + e-  (energy-conserving closure)
    dG4 = G_OER_TOTAL - dG1 - dG2 - dG3

    dg_steps = [dG1, dG2, dG3, dG4]
    overpotential = max(dg_steps) - U_OER_EQ

    dg_cumulative = [
        0.0,
        dG1,
        dG1 + dG2,
        dG1 + dG2 + dG3,
        G_OER_TOTAL,
    ]

    return overpotential, dg_steps, dg_cumulative
