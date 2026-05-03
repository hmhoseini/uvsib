from __future__ import annotations
import json
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Callable
from ase.data import covalent_radii
from ase import Atoms
from ase.io import jsonio
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.constraints import FixAtoms
from pymatgen.core.periodic_table import DummySpecies
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.surface import Slab
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from scipy.spatial.distance import pdist, squareform

def _create_adsorbate_with_dummy(species: List[str],
                                 coords: List[List[float]],
                                 properties: Dict = {},
                                 height: float = 2) -> Molecule:
    """
    Create a Pymatgen Molecule with a DummySpecies binding atom.
    
    Parameters
    ----------
    species : list of str
        Species symbols
    coords : list of list of float
        Coordinates relative to binding site [in Angstroms].
    height : float, optional
        Length of X-Molecule bond (default: 2.0 Å).
    
    Returns
    -------
    Molecule
        Pymatgen Molecule with DummySpecies "X" at index 0 [0, 0, 0].
    """
    shift = np.array([0, 0, height])
    shifted_coords = [list(np.array(c) + shift) for c in coords]
    mol = Molecule(
        [DummySpecies("X")] + species,
        [[0.0, 0.0, 0.0]] + shifted_coords,
    )
    mol.properties = properties
    return mol

def has_reasonable_distances(atoms: Atoms, scale: float = 0.5) -> bool:
    """
    Check if interatomic distances are physically reasonable.
    Only checks distances involving H, C, N, or O atoms.
    Parameters
    ----------
    atoms : ASE Atoms object
    scale : float, optional
        Scaling factor for minimum allowed distance (default: 0.5).
        min_distance = scale * (covalent_radius_i + covalent_radius_j)
    Returns
    -------
    bool
        True if all checked distances are >= min_distance, False otherwise.
    """
    CHECK_ELEMENTS = {1, 6, 7, 8}
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    n = len(atoms)

    for i in range(n):
        Zi = numbers[i]
        for j in range(i + 1, n):
            Zj = numbers[j]

            # Skip if neither atom is a CHECK_ELEMENT
            if Zi not in CHECK_ELEMENTS and Zj not in CHECK_ELEMENTS:
                continue

            d = np.linalg.norm(positions[i] - positions[j])
            r_min = scale * (covalent_radii[Zi] + covalent_radii[Zj])

            if d < r_min:
                return False
    return True

def ase_to_pmg(atoms):
    """Convert an ASE Atoms object to a pymatgen Structure"""
    lattice = atoms.cell.array.tolist()
    symbols = atoms.get_chemical_symbols()
    frac_coords = atoms.get_scaled_positions().tolist()
    lattice_obj = Lattice(lattice)
    return Structure(lattice_obj,
                          symbols,
                          frac_coords,
                          coords_are_cartesian=False)

def pmg_to_ase(pmg_structure):
    """Convert a pymatgen Structure to an ASE Atoms object"""
    scaled_positions = pmg_structure.frac_coords
    symbols = [str(site.specie) for site in pmg_structure.sites]
    cell = pmg_structure.lattice.matrix

    atoms = Atoms(
        symbols=symbols,
        scaled_positions=scaled_positions,
        cell=cell,
        pbc=True
    )

    if "selective_dynamics" in pmg_structure.site_properties:
        sd = pmg_structure.site_properties["selective_dynamics"]
        mask = [not any(flags) for flags in sd]
        if any(mask):
            atoms.set_constraint(FixAtoms(mask=mask))
    return atoms

def average_minimum_distance_structure(structure):
    """"Compute the average nearest-neighbor distance"""
    positions = structure.cart_coords
    dist_matrix = squareform(pdist(positions))
    ma = np.ma.masked_equal(dist_matrix, 0.0, copy=False)
    minimums = np.min(ma, axis=0)
    return minimums.mean()

def get_adsorption_sites(slab_pmg: Structure,
                         positions: List[str] = ['ontop', 'bridge', 'hollow']) -> tuple:
    """
    Get adsorption sites using pymatgen's AdsorbateSiteFinder.
    
    Parameters
    ----------
    slab_pmg : pymatgen Structure
    positions : list of str, optional
        Site types to find: 'ontop', 'bridge', 'hollow', or combinations (default: all three).
    Returns
    -------
    tuple
        (sites_dict, asf) where:
        - sites_dict: Dictionary with site information for each site type
        - asf: AdsorbateSiteFinder instance for further operations
    """
    # Height criteria for selection of surface sites
    h = 1.7 * average_minimum_distance_structure(slab_pmg)
    asf = AdsorbateSiteFinder(slab_pmg, selective_dynamics=True, height=h)

    sites_dict = asf.find_adsorption_sites(
        distance=0,
        put_inside=True,
        symm_reduce=0.1,
        near_reduce=0.1,
        positions=positions,
        no_obtuse_hollow=True
    )

    return sites_dict, asf

# ==============================================================================
# Reaction pathway data model (shared by all reaction types)
# ==============================================================================
@dataclass
class ReactionStep:
    """One elementary step in a reaction pathway.

    Attributes:
        reactant: Name of the surface intermediate before this step.
        product: Name of the surface intermediate after this step.
        step_type: ``'electrochemical'`` (H⁺ + e⁻ transfer) or ``'chemical'`` (no charge).
        electrons: Number of electrons transferred (negative = gained).
        protons: Number of protons transferred.
        released: Species desorbed/released (e.g., ``['H2O']``, ``['CO']``).
        notes: Free-text annotation (e.g., rate-limiting, competing mechanisms).
    """
    reactant: str
    product: str
    step_type: str = "electrochemical"
    electrons: int = -1
    protons: int = 1
    released: List[str] = field(default_factory=list)
    notes: str = ""

@dataclass
class ReactionPathway:
    """A named reaction pathway with ordered elementary steps.

    Attributes:
        name: Unique identifier (used by :func:`get_pathway`).
        description: Human-readable summary.
        steps: Ordered list of :class:`ReactionStep` objects.
        selectivity_metals: Metals where this pathway dominates in DFT/expt.
        overall_reaction: Balanced overall equation string.
    """
    name: str
    description: str
    steps: List[ReactionStep]
    selectivity_metals: List[str] = field(default_factory=list)
    overall_reaction: str = ""

    @property
    def intermediates(self) -> List[str]:
        """All surface intermediates in pathway order (no duplicates)."""
        seen: List[str] = []
        for step in self.steps:
            if step.reactant not in seen:
                seen.append(step.reactant)
        last = self.steps[-1].product
        if last not in seen:
            seen.append(last)
        return seen

def generate_oer_adsorbates():
    """Return a list of Pymatgen Molecule objects for the OER reaction"""

    X = DummySpecies("X")

    adsorbates = []

    # *O
    mol = Molecule(
        [X, "O"],
        [(0, 0, 0), (0, 0, 2.0)],
    )
    mol.properties = {"adsorbate": "*O", "energy": 0}
    adsorbates.append(mol)

    # *OH
    mol = Molecule(
        [X, "O", "H"],
        [(0, 0, 0), (0, 0, 2.0), (0.1, 0.9, 2.9)],
    )
    mol.properties = {"adsorbate": "*OH", "energy": 0}
    adsorbates.append(mol)

    # *OOH
    mol = Molecule(
        [X, "O", "O", "H"],
        [(0, 0, 0), (0, 0, 2.0), (1.2, -0.2, 2.8), (0.8, -0.2, 3.7)],
    )
    mol.properties = {"adsorbate": "*OOH", "energy": 0}
    adsorbates.append(mol)

    return adsorbates

def generate_co2rr_adsorbates(pathway_name: str) -> tuple:
    """CO2 electroreduction reaction pathways on metal surfaces.

    Provides a literature-grounded library of CO2RR intermediates and reaction
    pathways (CHE model, Nørskov group and follow-up work) that can be placed on
    *any* surface slab.

    Pathways implemented
    --------------------
    - 'co2_to_co'    : CO2 → *COOH → *CO → CO(g)            (Au, Ag, Zn)
    - 'co2_to_hcooh' : CO2 → *OCHO → HCOOH(aq)              (formate, Pd, In)
    - 'co_to_ch4'    : *CO → *CHO → *CHOH → *CH2 → *CH3 → CH4(g)  (Cu)
    - 'co_to_ch3oh'  : *CO → *CHO → *CHOH → *CH2OH → CH3OH(g)     (Cu)
    - 'co2_to_ch4'   : CO2 → CH4 full pathway on Cu
    - 'co2_to_ch3oh' : CO2 → CH3OH full pathway on Cu
    - 'co2_to_c2h4'  : 2 *CO → *OCCO → … → C2H4(g)          (Cu C–C coupling)

    Each intermediate is a Pymatgen Molecule (X is binding atom at index 0)

    Returns
    -------
    tuple
        (pathway, adsorbates_dict) where pathway is a ReactionPathway and 
        adsorbates_dict maps intermediate names to Molecule objects.

    References
    ----------
    Peterson et al. *Energy Environ. Sci.* **3**, 1311 (2010).
    Kuhl et al. *J. Am. Chem. Soc.* **136**, 14107 (2014).
    Montoya et al. *ChemSusChem* **8**, 2180 (2015).
    Goodpaster et al. *J. Phys. Chem. Lett.* **7**, 1471 (2016).
    """

    # =========================================================================
    # Adsorbate geometry library (Pymatgen version)
    # Each molecule includes a DummySpecies "X" at [0, 0, 0]
    # =========================================================================

    def _co():
        """*CO — C-down (atop binding on most metals)."""
        return _create_adsorbate_with_dummy(
            ["C", "O"],
            [[0, 0, 0], [0, 0, 1.15]],
            properties = {"adsorbate": "*CO"}
        )

    def _cooh():
        """*COOH (carboxyl) — C-down, bidentate capable.
        Planar: C at origin, C=O pointing up, C-OH in-plane.
        """
        return _create_adsorbate_with_dummy(
            ["C", "O", "O", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 1.22],
                [1.10, 0.00, -0.60],
                [1.94, 0.00, -0.12],
            ],
            properties = {"adsorbate": "*COOH"}
        )

    def _ocho():
        """*OCHO (formate) — O-down bidentate or monodentate.
        Monodentate: O at origin; formate oriented upright.
        """
        return _create_adsorbate_with_dummy(
            ["O", "C", "O", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 1.30],
                [0.00, 1.10, 1.85],
                [0.00, -0.95, 1.98],
            ],
            properties = {"adsorbate": "*OCHO"}
        )

    def _co2_ads():
        """*CO2 — weakly adsorbed, bent (activated), O-C-O ~125°."""
        angle_rad = np.radians(125.0 / 2)
        d = 1.20
        return _create_adsorbate_with_dummy(
            ["C", "O", "O"],
            [
                [0.00, 0.00, 0.00],
                [d * np.sin(angle_rad), 0, d * np.cos(angle_rad)],
                [-d * np.sin(angle_rad), 0, d * np.cos(angle_rad)],
            ],
            properties = {"adsorbate": "*CO2"}
        )

    def _cho():
        """*CHO (formyl) — C-down."""
        return _create_adsorbate_with_dummy(
            ["C", "H", "O"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 1.09, 0.63],
                [1.05, 0.00, 0.85],
            ],
            properties = {"adsorbate": "*CHO"}
        )

    def _choh():
        """*CHOH — C-down."""
        return _create_adsorbate_with_dummy(
            ["C", "H", "O", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 1.09, 0.70],
                [1.23, 0.00, 0.65],
                [1.85, 0.00, 1.38],
            ],
            properties = {"adsorbate": "*CHOH"}
        )

    def _ch2o():
        """*CH2O (formaldehyde adsorbed) — C-down, η1 mode."""
        return _create_adsorbate_with_dummy(
            ["C", "H", "H", "O"],
            [
                [0.00, 0.00, 0.00],
                [0.94, 0.00, 0.59],
                [-0.94, 0.00, 0.59],
                [0.00, 1.10, 0.60],
            ],
            properties = {"adsorbate": "*CH2O"}
        )

    def _ch2oh():
        """*CH2OH — C-down."""
        return _create_adsorbate_with_dummy(
            ["C", "H", "H", "O", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.95, 0.55, 0.62],
                [-0.95, 0.55, 0.62],
                [0.00, -1.22, 0.62],
                [0.00, -1.90, 1.35],
            ],
            properties = {"adsorbate": "*CH2OH"}
        )

    def _ch():
        """*CH — C-down (hollow site preferred)."""
        return _create_adsorbate_with_dummy(
            ["C", "H"],
            [[0, 0, 0], [0, 0, 1.09]],
            properties = {"adsorbate": "*CH"}
        )

    def _ch2():
        """*CH2 — C-down."""
        return _create_adsorbate_with_dummy(
            ["C", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.94, 0.00, 0.70],
                [-0.94, 0.00, 0.70],
            ],
            properties = {"adsorbate": "*CH2"}
        )

    def _ch3():
        """*CH3 — C-down (atop)."""
        return _create_adsorbate_with_dummy(
            ["C", "H", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 1.03, 0.70],
                [0.89, -0.51, 0.70],
                [-0.89, -0.51, 0.70],
            ],
            properties = {"adsorbate": "*CH3"}
        )

    def _oh():
        """*OH — O-down."""
        return _create_adsorbate_with_dummy(
            ["O", "H"],
            [[0, 0, 0], [0, 0, 0.97]],
            properties = {"adsorbate": "*OH"}
        )

    def _h():
        """*H — single hydrogen."""
        return _create_adsorbate_with_dummy(
            ["H"],
            [[0, 0, 0]],
            properties = {"adsorbate": "*H"}
        )

    def _occo():
        """*OCCO (oxalyl, CO dimer) — first O binds surface."""
        return _create_adsorbate_with_dummy(
            ["O", "C", "C", "O"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 1.25],
                [0.00, 1.35, 1.25],
                [0.00, 1.35, 2.50],
            ],
            properties = {"adsorbate": "*OCCO"}
        )

    def _ccho():
        """*CCHO — C-down, C2 species."""
        return _create_adsorbate_with_dummy(
            ["C", "C", "O", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 1.35, 0.65],
                [0.00, 2.40, 1.30],
                [0.00, 1.35, 1.76],
            ],
            properties = {"adsorbate": "*CCHO"}
        )

    def _c2h4():
        """C2H4 (ethylene) — di-σ mode, C-C bridging; C1 at origin (mol_index=0)."""
        return _create_adsorbate_with_dummy(
            ["C", "C", "H", "H", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [1.34, 0.00, 0.00],
                [-0.56, 0.92, 0.60],
                [-0.56, -0.92, 0.60],
                [1.90, 0.92, 0.60],
                [1.90, -0.92, 0.60],
            ],
            properties = {"adsorbate": "C2H4"}
        )

    # Registry: name → factory function
    _ADSORBATE_REGISTRY: Dict[str, Callable] = {
        "CO2_ads": _co2_ads,
        "COOH": _cooh,
        "OCHO": _ocho,
        "CO": _co,
        "CHO": _cho,
        "CHOH": _choh,
        "CH2O": _ch2o,
        "CH2OH": _ch2oh,
        "CH": _ch,
        "CH2": _ch2,
        "CH3": _ch3,
        "OH": _oh,
        "H": _h,
        "OCCO": _occo,
        "CCHO": _ccho,
        "C2H4_ads": _c2h4,
    }

    def get_adsorbate(name: str) -> Molecule:
        """Return a fresh copy of the named adsorbate geometry.
        Args:
            name: Key from the adsorbate registry.
        Returns:
            Pymatgen Molecule with X at index 0
        Raises:
            KeyError: If *name* is not in the registry.
        """
        if name not in _ADSORBATE_REGISTRY:
            raise KeyError(
                f"Unknown adsorbate '{name}'. "
                f"Available: {sorted(_ADSORBATE_REGISTRY)}"
            )
        return _ADSORBATE_REGISTRY[name]()

    # =========================================================================
    # Reaction pathway data
    # =========================================================================
    _PATHWAYS: dict[str, ReactionPathway] = {}

    def _reg(pw: ReactionPathway) -> ReactionPathway:
        _PATHWAYS[pw.name] = pw
        return pw

    _reg(ReactionPathway(
        name="co2_to_co",
        description="CO2 reduction to CO via carboxyl intermediate",
        overall_reaction="CO2 + H+ + e- → CO + H2O",
        selectivity_metals=["Au", "Ag", "Zn"],
        steps=[
            ReactionStep(
                reactant="CO2_ads", product="COOH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="potential-limiting on most metals",
            ),
            ReactionStep(
                reactant="COOH", product="CO",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
                notes="CO desorbs on weak-binding metals (Au, Ag)",
            ),
        ],
    ))

    _reg(ReactionPathway(
        name="co2_to_hcooh",
        description="CO2 reduction to formic acid via formate intermediate",
        overall_reaction="CO2 + 2(H+ + e-) → HCOOH",
        selectivity_metals=["Pd", "In", "Sn"],
        steps=[
            ReactionStep(
                reactant="CO2_ads", product="OCHO",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="O-bound formate; competing with COOH on Pd/In",
            ),
            ReactionStep(
                reactant="OCHO", product="CO2_ads",  # product released
                step_type="electrochemical", electrons=-1, protons=1,
                released=["HCOOH"],
                notes="HCOOH desorbs; surface returns to clean",
            ),
        ],
    ))

    _reg(ReactionPathway(
        name="co_to_ch4",
        description="Further reduction of *CO to methane (Langmuir–Hinshelwood)",
        overall_reaction="*CO + 6(H+ + e-) → CH4 + H2O",
        selectivity_metals=["Cu"],
        steps=[
            ReactionStep(
                reactant="CO", product="CHO",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="potential-limiting on Cu",
            ),
            ReactionStep(
                reactant="CHO", product="CHOH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CHOH", product="CH",
                step_type="chemical", electrons=0, protons=0,
                released=["H2O"],
                notes="water elimination; CH occupies hollow site",
            ),
            ReactionStep(
                reactant="CH", product="CH2",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CH2", product="CH3",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CH3", product="CO",  # product = bare surface marker
                step_type="electrochemical", electrons=-1, protons=1,
                released=["CH4"],
            ),
        ],
    ))

    _reg(ReactionPathway(
        name="co_to_ch3oh",
        description="Further reduction of *CO to methanol",
        overall_reaction="*CO + 4(H+ + e-) → CH3OH",
        selectivity_metals=["Cu", "Mo"],
        steps=[
            ReactionStep(
                reactant="CO", product="CHO",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CHO", product="CHOH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CHOH", product="CH2OH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CH2OH", product="CO",  # bare surface
                step_type="electrochemical", electrons=-1, protons=1,
                released=["CH3OH"],
            ),
        ],
    ))

    _reg(ReactionPathway(
        name="co2_to_ch4",
        description="Full CO2 → CH4 pathway on Cu",
        overall_reaction="CO2 + 8(H+ + e-) → CH4 + 2 H2O",
        selectivity_metals=["Cu"],
        steps=[
            ReactionStep(
                reactant="CO2_ads", product="COOH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="COOH", product="CO",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
            ),
            ReactionStep(
                reactant="CO", product="CHO",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="potential-limiting on Cu",
            ),
            ReactionStep(
                reactant="CHO", product="CHOH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CHOH", product="CH",
                step_type="chemical", electrons=0, protons=0,
                released=["H2O"],
            ),
            ReactionStep(
                reactant="CH", product="CH2",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CH2", product="CH3",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CH3", product="CO",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["CH4"],
            ),
        ],
    ))

    _reg(ReactionPathway(
        name="co2_to_ch3oh",
        description="Full CO2 → CH3OH pathway on Cu/Mo",
        overall_reaction="CO2 + 6(H+ + e-) → CH3OH + H2O",
        selectivity_metals=["Cu", "Mo"],
        steps=[
            ReactionStep(
                reactant="CO2_ads", product="COOH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="COOH", product="CO",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
            ),
            ReactionStep(
                reactant="CO", product="CHO",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CHO", product="CHOH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CHOH", product="CH2OH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CH2OH", product="CO",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["CH3OH"],
            ),
        ],
    ))

    _reg(ReactionPathway(
        name="co2_to_c2h4",
        description="CO2 → C2H4 via CO dimerisation on Cu",
        overall_reaction="2 CO2 + 12(H+ + e-) → C2H4 + 4 H2O",
        selectivity_metals=["Cu"],
        steps=[
            ReactionStep(
                reactant="CO2_ads", product="COOH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="COOH", product="CO",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
            ),
            ReactionStep(
                reactant="CO", product="OCCO",
                step_type="chemical", electrons=0, protons=0,
                notes="C–C coupling of two *CO; rate-determining on Cu(100)",
            ),
            ReactionStep(
                reactant="OCCO", product="CCHO",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CCHO", product="C2H4_ads",
                step_type="chemical", electrons=0, protons=0,
                notes="multiple H+/e- transfers collapsed; treated as chemical for pathway bookkeeping",
            ),
            ReactionStep(
                reactant="C2H4_ads", product="CO",
                step_type="chemical", electrons=0, protons=0,
                released=["C2H4"],
            ),
        ],
    ))

    def get_co2rr_pathway(name: str) -> ReactionPathway:
        """Return the named CO2RR :class:`ReactionPathway`.
        Args:
            name: One of the CO2RR pathway keys.
        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in _PATHWAYS:
            raise KeyError(
                f"Unknown CO2RR pathway '{name}'. "
                f"Available: {sorted(_PATHWAYS)}"
            )
        return _PATHWAYS[name]

    skip_if_not_registered = {"CO"} #TODO not clear

    pathway = get_co2rr_pathway(pathway_name)
    adsorbates = {}

    for name in pathway.intermediates:
        # Skip "bare surface" markers that aren't registered adsorbates
        if name in skip_if_not_registered and name not in _ADSORBATE_REGISTRY:
            continue
        if name in _ADSORBATE_REGISTRY:
            adsorbates[name] = get_adsorbate(name)

    return pathway, adsorbates

def generate_noxrr_adsorbates(pathway_name: str) -> tuple:
    """NOx electroreduction reaction pathways on metal surfaces.

    Provides a library of NOx reduction intermediates and reaction pathways
    (CHE model) that can be placed on *any* surface slab.

    NOx covered: NO, NO₂, NO₃⁻ as starting species.

    Pathways implemented
    --------------------
    - 'no_dissociative' : *NO → *N + *O → N₂(g)              (Ru, Rh, Ir)
    - 'no_to_nh3_noh'   : *NO → *NOH → *N → *NH₂ → NH₃       (Cu, Fe)
    - 'no_to_nh3_nhoh'  : *NO → *NOH → *NHOH → *NH₂OH → NH₃  (Cu, hydroxylamine route)
    - 'no_to_n2o'       : 2*NO → *ONNO → N₂O + *O            (Pt, Pd automotive)
    - 'no2_to_no'       : *NO₂ → *NO + *O                     (prereduction step)
    - 'no3_to_nh3'      : *NO₃ → *NO₂ → *NO → … → NH₃        (eNO3RR, Cu)
    - 'no3_to_n2'       : *NO₃ → *NO₂ → *NO → *N → N₂        (eNO3RR, Ru)

    Returns
    -------
    tuple
        (pathway, adsorbates_dict) where pathway is a ReactionPathway and 
        adsorbates_dict maps intermediate names to Molecule objects.

    References
    ----------
    Gao et al. *Nat. Chem.* **9**, 547 (2017).
    Liu et al. *Nat. Commun.* **12**, 5797 (2021).
    Wang et al. *J. Am. Chem. Soc.* **142**, 5702 (2020).
    van 't Veer et al. *J. Phys. Chem. C* **124**, 22 (2020).
    Pérez-Ramírez & López *Nat. Catal.* **2**, 971 (2019).
    """

    # =========================================================================
    # Adsorbate geometry library (Pymatgen version with DummySpecies)
    # =========================================================================

    def _no():
        """*NO — N-down (preferred on most transition metals)."""
        return _create_adsorbate_with_dummy(
            ["N", "O"],
            [[0, 0, 0], [0, 0, 1.15]],
            properties = {"adsorbate": "*NO"}
        )

    def _no2():
        """*NO₂ — N-down, bent (O–N–O ≈ 115°)."""
        half = np.radians(115.0 / 2)
        d = 1.20
        return _create_adsorbate_with_dummy(
            ["N", "O", "O"],
            [
                [0.00, 0.00, 0.00],
                [d * np.sin(half), 0, d * np.cos(half)],
                [-d * np.sin(half), 0, d * np.cos(half)],
            ],
            properties = {"adsorbate": "*NO2"}
        )

    def _no3():
        """*NO₃ — N-down, planar nitrate (D₃ₕ, N–O = 1.24 Å) slightly tilted up."""
        d = 1.24
        h = 0.40
        return _create_adsorbate_with_dummy(
            ["N", "O", "O", "O"],
            [
                [0.00, 0.00, 0.00],
                [0.00, d, h],
                [d * np.sin(np.radians(120)), -d * 0.5, h],
                [-d * np.sin(np.radians(120)), -d * 0.5, h],
            ],
            properties = {"adsorbate": "*NO3"}
        )

    def _noh():
        """*NOH — N-down, O–H bond (first H on O)."""
        return _create_adsorbate_with_dummy(
            ["N", "O", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 1.20],
                [0.90, 0.00, 1.70],
            ],
            properties = {"adsorbate": "*NOH"}
        )

    def _hno():
        """*HNO — N-down, N–H bond (first H on N)."""
        return _create_adsorbate_with_dummy(
            ["N", "H", "O"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 1.01, 0.42],
                [1.10, 0.00, 0.75],
            ],
            properties = {"adsorbate": "*HNO"}
        )

    def _n2o2():
        """*ONNO (cis-hyponitrite dimer, N-down) — O=N–N=O bridge species."""
        return _create_adsorbate_with_dummy(
            ["N", "O", "N", "O"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 1.18],
                [1.30, 0.00, 0.00],
                [1.30, 0.00, 1.18],
            ],
            properties = {"adsorbate": "*ONNO"}
        )

    def _n2o():
        """*N₂O — N-down, linear (N≡N–O, terminal N binds)."""
        return _create_adsorbate_with_dummy(
            ["N", "N", "O"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 1.13],
                [0.00, 0.00, 2.27],
            ],
            properties = {"adsorbate": "*N2O"}
        )

    def _n_ads():
        """*N — atomic nitrogen (hollow site preferred)."""
        return _create_adsorbate_with_dummy(
            ["N"],
            [[0, 0, 0]],
            properties = {"adsorbate": "*N"}
        )

    def _nh():
        """*NH — N-down."""
        return _create_adsorbate_with_dummy(
            ["N", "H"],
            [[0, 0, 0], [0, 0, 1.01]],
            properties = {"adsorbate": "*NH"}
        )

    def _nh2():
        """*NH₂ — N-down."""
        return _create_adsorbate_with_dummy(
            ["N", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.82, 0.00, 0.56],
                [-0.82, 0.00, 0.56],
            ],
            properties = {"adsorbate": "*NH2"}
        )

    def _nh3():
        """*NH₃ — N-down (tetrahedral)."""
        return _create_adsorbate_with_dummy(
            ["N", "H", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.94, 0.34],
                [0.82, -0.47, 0.34],
                [-0.82, -0.47, 0.34],
            ],
            properties = {"adsorbate": "*NH3"}
        )

    def _nhoh():
        """*NHOH — N-down, both N–H and O–H bonds present."""
        return _create_adsorbate_with_dummy(
            ["N", "H", "O", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.95, 0.45],
                [1.25, 0.00, 0.65],
                [1.85, 0.00, 1.40],
            ],
            properties = {"adsorbate": "*NHOH"}
        )

    def _nh2oh():
        """*NH₂OH (hydroxylamine) — N-down."""
        return _create_adsorbate_with_dummy(
            ["N", "H", "H", "O", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.95, 0.45],
                [-0.95, -0.35, 0.45],
                [1.22, 0.00, 0.75],
                [1.82, 0.00, 1.50],
            ],
            properties = {"adsorbate": "*NH2OH"}
        )

    def _o_ads():
        """*O — atomic oxygen."""
        return _create_adsorbate_with_dummy(
            ["O"],
            [[0, 0, 0]],
            properties = {"adsorbate": "*O"}
        )

    def _oh():
        """*OH — O-down."""    
        return _create_adsorbate_with_dummy(
            ["O", "H"],
            [[0, 0, 0], [0, 0, 0.97]],
            properties = {"adsorbate": "*OH"}
        )

    def _h2o():
        """*H₂O — O-down (physisorbed; typically desorbs above 200 K)."""
        return _create_adsorbate_with_dummy(
            ["O", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.76, 0.00, 0.59],
                [-0.76, 0.00, 0.59],
            ],
            properties = {"adsorbate": "*H2O"}
        )

    # Registry
    _ADSORBATE_REGISTRY: Dict[str, Callable] = {
        "NO": _no,
        "NO2": _no2,
        "NO3": _no3,
        "NOH": _noh,
        "HNO": _hno,
        "N2O2": _n2o2,
        "N2O": _n2o,
        "N": _n_ads,
        "NH": _nh,
        "NH2": _nh2,
        "NH3": _nh3,
        "NHOH": _nhoh,
        "NH2OH": _nh2oh,
        "O": _o_ads,
        "OH": _oh,
        "H2O": _h2o,
    }

    def get_noxrr_adsorbate(name: str) -> Molecule:
        """Return a fresh copy of the named NOx-reduction adsorbate.

        Args:
            name: Key from the adsorbate registry.
        Returns:
            Pymatgen Molecule with X at index 0
        Raises:
            KeyError: If *name* is not in the registry.
        """
        if name not in _ADSORBATE_REGISTRY:
            raise KeyError(
                f"Unknown NOx adsorbate '{name}'. "
                f"Available: {sorted(_ADSORBATE_REGISTRY)}"
            )
        return _ADSORBATE_REGISTRY[name]()

    # =========================================================================
    # Reaction pathway data (using global ReactionPathway and ReactionStep)
    # =========================================================================
    _PATHWAYS: dict[str, ReactionPathway] = {}

    def _reg(pw: ReactionPathway) -> ReactionPathway:
        _PATHWAYS[pw.name] = pw
        return pw

    # 1 ── Dissociative NO reduction → N₂ (Ru, Rh, Ir catalytic)
    _reg(ReactionPathway(
        name="no_dissociative",
        description="Dissociative NO reduction to N₂ via N coupling",
        overall_reaction="2 NO → N₂ + 2 O*",
        selectivity_metals=["Ru", "Rh", "Ir", "Pd"],
        steps=[
            ReactionStep(
                reactant="NO", product="N",
                step_type="chemical", electrons=0, protons=0,
                released=["O"],
                notes="N–O bond scission; rate-limiting on Rh/Pd",
            ),
            ReactionStep(
                reactant="N", product="N2O2",
                step_type="chemical", electrons=0, protons=0,
                notes="N + N Langmuir–Hinshelwood coupling (2 *N sites)",
            ),
            ReactionStep(
                reactant="N2O2", product="N",  # bare surface marker
                step_type="chemical", electrons=0, protons=0,
                released=["N2"],
                notes="N₂ desorption",
            ),
        ],
    ))

    # 2 ── Electrochemical NO → NH₃ via NOH route (N–O bond breaks early)
    _reg(ReactionPathway(
        name="no_to_nh3_noh",
        description="Electrochemical NO reduction to NH₃ via *NOH → *N pathway",
        overall_reaction="NO + 5(H+ + e-) → NH3 + H2O",
        selectivity_metals=["Cu", "Fe", "Co"],
        steps=[
            ReactionStep(
                reactant="NO", product="NOH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="O protonation; competing with *HNO on many metals",
            ),
            ReactionStep(
                reactant="NOH", product="N",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
                notes="N–O cleavage after O protonation",
            ),
            ReactionStep(
                reactant="N", product="NH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NH", product="NH2",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NH2", product="NH3",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NH3", product="N",  # bare surface marker
                step_type="chemical", electrons=0, protons=0,
                released=["NH3"],
                notes="NH₃ desorption",
            ),
        ],
    ))

    # 3 ── Electrochemical NO → NH₃ via hydroxylamine route (*NHOH → *NH₂OH)
    _reg(ReactionPathway(
        name="no_to_nh3_nhoh",
        description="Electrochemical NO reduction to NH₃ via hydroxylamine (*NH₂OH)",
        overall_reaction="NO + 5(H+ + e-) → NH3 + H2O",
        selectivity_metals=["Cu", "Pt", "Ag"],
        steps=[
            ReactionStep(
                reactant="NO", product="NOH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="O protonation; first step common to both routes",
            ),
            ReactionStep(
                reactant="NOH", product="NHOH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="N protonation giving N,O-dihydroxyl; potential-limiting on Cu",
            ),
            ReactionStep(
                reactant="NHOH", product="NH2OH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="N protonation to hydroxylamine",
            ),
            ReactionStep(
                reactant="NH2OH", product="NH2",
                step_type="chemical", electrons=0, protons=0,
                released=["H2O"],
                notes="N–O bond cleavage / H₂O elimination",
            ),
            ReactionStep(
                reactant="NH2", product="NH3",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NH3", product="N",  # bare surface marker
                step_type="chemical", electrons=0, protons=0,
                released=["NH3"],
                notes="NH₃ desorption",
            ),
        ],
    ))

    # 4 ── Catalytic NO → N₂O (automotive, Pt/Pd/Rh three-way catalyst)
    _reg(ReactionPathway(
        name="no_to_n2o",
        description="Catalytic NO dimerisation to N₂O (Langmuir–Hinshelwood)",
        overall_reaction="2 NO → N₂O + O*",
        selectivity_metals=["Pt", "Pd", "Rh"],
        steps=[
            ReactionStep(
                reactant="NO", product="N2O2",
                step_type="chemical", electrons=0, protons=0,
                notes="2*NO → *ONNO dimer; kinetically facile on Pt(111)",
            ),
            ReactionStep(
                reactant="N2O2", product="N2O",
                step_type="chemical", electrons=0, protons=0,
                released=["O"],
                notes="asymmetric N–O cleavage of dimer leaving *N₂O + *O",
            ),
            ReactionStep(
                reactant="N2O", product="NO",  # bare surface marker
                step_type="chemical", electrons=0, protons=0,
                released=["N2O"],
                notes="N₂O desorption (or further reduction on Rh)",
            ),
        ],
    ))

    # 5 ── NO₂ → NO (prereduction / first step of NO₂ reduction)
    _reg(ReactionPathway(
        name="no2_to_no",
        description="NO₂ reduction to NO (dissociative O removal)",
        overall_reaction="NO₂ + 2(H+ + e-) → NO + H₂O",
        selectivity_metals=["Pt", "Cu", "Ru", "Fe"],
        steps=[
            ReactionStep(
                reactant="NO2", product="NO",
                step_type="chemical", electrons=0, protons=0,
                released=["O"],
                notes="N–O scission; *O then removed by H (next steps) or stays",
            ),
            ReactionStep(
                reactant="O", product="OH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="*O hydrogenation",
            ),
            ReactionStep(
                reactant="OH", product="NO",  # bare surface marker
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
                notes="*OH hydrogenation → H₂O desorption",
            ),
        ],
    ))

    # 6 ── eNO3RR: NO₃⁻ → NH₃ (full electrochemical, Cu selectivity)
    _reg(ReactionPathway(
        name="no3_to_nh3",
        description="Electrochemical NO₃⁻ reduction to NH₃ (eNO3RR) via NOH route",
        overall_reaction="NO3- + 9(H+ + e-) → NH3 + 3 H2O",
        selectivity_metals=["Cu", "Fe", "Co", "Ru"],
        steps=[
            ReactionStep(
                reactant="NO3", product="NO2",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
                notes="first reduction; often fast on Cu",
            ),
            ReactionStep(
                reactant="NO2", product="NO",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
            ),
            ReactionStep(
                reactant="NO", product="NOH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="potential-limiting step on many metals",
            ),
            ReactionStep(
                reactant="NOH", product="N",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
            ),
            ReactionStep(
                reactant="N", product="NH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NH", product="NH2",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NH2", product="NH3",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NH3", product="NO3",  # bare surface marker
                step_type="chemical", electrons=0, protons=0,
                released=["NH3"],
            ),
        ],
    ))

    # 7 ── eNO3RR: NO₃⁻ → N₂ (dissociative, Ru selectivity)
    _reg(ReactionPathway(
        name="no3_to_n2",
        description="Electrochemical NO₃⁻ reduction to N₂ (dissociative pathway, Ru)",
        overall_reaction="2 NO3- + 12(H+ + e-) → N2 + 6 H2O",
        selectivity_metals=["Ru", "Rh", "Ir"],
        steps=[
            ReactionStep(
                reactant="NO3", product="NO2",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
            ),
            ReactionStep(
                reactant="NO2", product="NO",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
            ),
            ReactionStep(
                reactant="NO", product="N",
                step_type="chemical", electrons=0, protons=0,
                released=["O"],
                notes="N–O bond scission; rate-limiting on Ru",
            ),
            ReactionStep(
                reactant="N", product="N2O2",
                step_type="chemical", electrons=0, protons=0,
                notes="N + N coupling (2 sites)",
            ),
            ReactionStep(
                reactant="N2O2", product="NO3",  # bare surface marker
                step_type="chemical", electrons=0, protons=0,
                released=["N2"],
            ),
        ],
    ))

    def get_noxrr_pathway(name: str) -> ReactionPathway:
        """Return the named NOxRR :class:`ReactionPathway`.
        Args:
            name: One of the NOxRR pathway keys.
        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in _PATHWAYS:
            raise KeyError(
                f"Unknown NOxRR pathway '{name}'. "
                f"Available: {sorted(_PATHWAYS)}"
            )
        return _PATHWAYS[name]

    pathway = get_noxrr_pathway(pathway_name)
    adsorbates = {}

    for name in pathway.intermediates:
        if name in _ADSORBATE_REGISTRY:
            adsorbates[name] = get_noxrr_adsorbate(name)

    return pathway, adsorbates

def get_multipliers(slab_pmg):
    a = slab_pmg.lattice.a
    b = slab_pmg.lattice.b
    if a > b:
        return [(1, 1, 1), (1, 2, 1), (1, 3, 1), (2, 2, 1)]
    return [(1, 1, 1), (2, 1, 1), (3, 1, 1), (2, 2, 1)]

def generate_adsorbed_structures(reaction: str, pathway_name: str = "") -> list:
    """Generate surface + adsorbate structures for specified reaction.
    
    Parameters
    ----------
    reaction : str
        Reaction type: 'OER', 'CO2RR', or 'NOXRR'.
    pathway_name : str, optional
        Pathway name (required for CO2RR and NOXRR).
    
    Returns
    -------
    list
        List of adsorption sets, where each set is a list of ASE Atoms objects
        with site_type, ads_coord, and adsorbate info in the Atoms.info dict.
    
    Raises
    ------
    ValueError
        If reaction type is unknown or pathway_name missing for certain reactions.
    FileNotFoundError
        If input_structures.json is not found.
    """
    with open('input_structures.json', 'r') as f:
        data = json.load(f)

    slab_pmg = Slab.from_dict(data[0])
    slab_pmg.sort()

    # Generate adsorbates based on reaction type and pathway
    if reaction == "OER":
        adsorbates = generate_oer_adsorbates()
    elif reaction == "CO2RR":
        _, adsorbates = generate_co2rr_adsorbates(pathway_name)
        adsorbates = list(adsorbates.values())
    elif reaction == "NOXRR":
        _, adsorbates = generate_noxrr_adsorbates(pathway_name)
        adsorbates = list(adsorbates.values())
    else:
        raise ValueError(f"Unknown reaction: {reaction}")

    # Get adsorption sites from slab
    sites_dict, asf = get_adsorption_sites(slab_pmg)

    # Build slab + adsorbate
    adsorption_sets = []
    site_types = ['ontop', 'bridge', 'hollow']
    multipliers = get_multipliers(slab_pmg)

    for repeat in multipliers:
        clean_slab = pmg_to_ase(asf.slab * (repeat))
        clean_slab.info['adsorbate'] = "*"
        for site_type in site_types:
            sites = sites_dict.get(site_type, [])
            for ads_coord in sites:
                adsorb_set = {}
                adsorb_set["site_type"] = site_type
                adsorb_set["ads_coord"] = ads_coord
                adsorb_set["repeat"] = repeat
                adsorb_set["structures"] = []
                for ads in adsorbates:
                    ads_slab = asf.add_adsorbate(ads, ads_coord, repeat=repeat, translate=False, reorient=True)
                    ads_slab.remove_species("X")

                    ase_struct = pmg_to_ase(ads_slab)
                    ase_struct.info['adsorbate'] = ads.properties['adsorbate']

                    if not has_reasonable_distances(ase_struct):
                        break
                    adsorb_set["structures"].append(ase_struct)

                # Only add complete sets where all adsorbates passed validation
                if len(adsorb_set["structures"]) == len(adsorbates):
                    adsorption_sets.append(adsorb_set)

    return clean_slab, adsorption_sets

def run_relaxation(ml_model: str, calc, fmax: float, max_steps: int,
                   reaction: str, pathway: str) -> dict:
    """Run geometry relaxation on adsorbate structures.
    
    Parameters
    ----------
    ml_model : str
        ML model name (used for output key naming).
    calc : Calculator
        ASE calculator object (MACE, UPET, MatterSim).
    fmax : float
        Force convergence criterion (eV/Å).
    max_steps : int
        Maximum relaxation steps.
    reaction : str
        Reaction type: 'OER', 'CO2RR', or 'NOXRR'.
    pathway : str
        Pathway name (for CO2RR/NOXRR) or empty string (for OER).
    """
    relaxed_sets = []
    num_failed = 0
    model_key = f'{ml_model.lower()}_energy'

    clean_slab, adsorption_sets = generate_adsorbed_structures(reaction, pathway)
    clean_slab.calc = calc
    relax = BFGSLineSearch(clean_slab, maxstep=0.1, logfile='opt.log')
    try:
        relax.run(fmax=fmax, steps=max_steps)
    except Exception as e:
        print(f"Erroe: Relaxation failed for the clean surface: {e}")
        exit()
    clean_slab.info[model_key] = clean_slab.get_potential_energy()

    for adsorb_set in adsorption_sets:
        relaxed_set = {}
        relaxed_set["site_type"] = adsorb_set["site_type"]
        relaxed_set["ads_coord"] = adsorb_set["ads_coord"].tolist()
        relaxed_set["repeat"] = adsorb_set["repeat"]
        relaxed_set["structures"] = []
        for adsorbed in adsorb_set["structures"]:
            adsorbed.calc = calc
            relax = BFGSLineSearch(adsorbed, maxstep=0.1, logfile='opt.log')
            try:
                relax.run(fmax=fmax, steps=max_steps)
            except Exception as e:
                print(f"Warning: Relaxation failed for {adsorbed.info.get('adsorbate', 'unknown')}: {e}")
                num_failed += 1
                break

            if not relax.converged:
                num_failed += 1
                break

            adsorbed.info[model_key] = adsorbed.get_potential_energy()
            relaxed_set["structures"].append(jsonio.encode(adsorbed))

        # Only add complete relaxed sets
        if len(relaxed_set["structures"]) == len(adsorb_set["structures"]):
            relaxed_set["structures"].append(jsonio.encode(clean_slab))
            relaxed_sets.append(relaxed_set)

    # Write output files
    output = {'structures': relaxed_sets}
    with open('output.json', 'w') as f:
        json.dump(output, f)

    with open('total.txt', 'w') as f:
        f.write(str(len(adsorption_sets)))

    with open('failed.txt', 'w') as f:
        f.write(str(num_failed))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ML_model", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--fmax", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--slab_energy", type=float)
    parser.add_argument("--reaction", type=str)
    parser.add_argument("--pathway", type=str)
    args = parser.parse_args()

    if "MACE" in args.ML_model:
        from mace.calculators import MACECalculator
        calc = MACECalculator(
                model_paths=args.model_path,
                device=args.device
        )
    elif "PET" in args.ML_model:
        from upet.calculator import UPETCalculator
        calc = UPETCalculator(
                model=args.model,
                device=args.device
        )
    elif "MatterSim" in args.ML_model:
        from mattersim.forcefield import MatterSimCalculator
        calc = MatterSimCalculator(
                load_path=args.model_path,
                device=args.device
        )
    else:
        raise ValueError(
            f"Unknown ML_model '{args.ML_model}'. "
            "Expected one of: MACE, PET, MatterSim."
        )

    run_relaxation(ml_model=args.ML_model, calc=calc, fmax=args.fmax, max_steps=args.max_steps,
                   reaction=args.reaction, pathway=args.pathway)
