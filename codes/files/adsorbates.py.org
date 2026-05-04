from __future__ import annotations

import json
import argparse
import numpy as np
from ase.data import covalent_radii
from ase import Atoms
from ase.io import jsonio
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.constraints import FixAtoms
from cluskit import Cluster, utils
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import Slab
from scipy.spatial import distance_matrix, Delaunay


CHECK_ELEMENTS = {1, 6, 7, 8}  # H, C, N, O (atomic numbers)

def has_reasonable_distances(atoms, scale=0.5):
    """
    Check interatomic distances only if at least one atom is H, C, N, or O.
    """
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    n = len(atoms)

    for i in range(n):
        Zi = numbers[i]
        for j in range(i + 1, n):
            Zj = numbers[j]

            if Zi not in CHECK_ELEMENTS and Zj not in CHECK_ELEMENTS:
                continue

            d = np.linalg.norm(positions[i] - positions[j])
            r_min = scale * (covalent_radii[Zi] + covalent_radii[Zj])

            if d < r_min:
                return False
    return True

def simple_delaunator(xyz, rcut):
    """
    Simplified surface detector for slabs or nanoclusters.

    Parameters
    ----------
    xyz : (N, 3) array
        Cartesian atomic positions.
    rcut : float
        Atoms within `rcut` A of z_max are considered surface atoms.
        Maximum distance between two surface atoms to be considered connected.
    """
    xyz = np.asarray(xyz)
    edge_cut = rcut

    # top surface atoms
    zmax = np.max(xyz[:, 2])
    ids_surface_atoms = np.where(xyz[:, 2] >= (zmax - rcut))[0]
    surf_xyz = xyz[ids_surface_atoms]

    normals_surface_atoms = np.tile([0, 0, 1.0], (len(surf_xyz), 1))

    # surface edges
    dist = distance_matrix(surf_xyz, surf_xyz)
    edge_pairs = np.array(np.where((dist > 0.0) & (dist < edge_cut))).T
    edge_pairs = edge_pairs[edge_pairs[:, 0] < edge_pairs[:, 1]]  # remove duplicates?

    centers_surface_edges = (
        (surf_xyz[edge_pairs[:, 0]] + surf_xyz[edge_pairs[:, 1]]) / 2.0
        if len(edge_pairs) > 0 else np.empty((0, 3))
    )

    normals_surface_edges = np.zeros_like(centers_surface_edges)
    if len(edge_pairs) > 0:
        z_axis = np.array([0, 0, 1.0])
        edge_vecs = surf_xyz[edge_pairs[:, 1]] - surf_xyz[edge_pairs[:, 0]]
        for i, v in enumerate(edge_vecs):
            n = np.cross(np.cross(z_axis, v), v)
            if n[2] < 0:  # ensure consistent orientation
                n = -n
            norm = np.linalg.norm(n)
            normals_surface_edges[i] = n / norm if norm > 1e-10 else [0, 0, 0]

    # surface triangles via 2D Delaunay in xy-plane
    tri_ids = np.empty((0, 3), int)
    centers_surface_triangles = np.empty((0, 3))
    normals_surface_triangles = np.empty((0, 3))

    if len(surf_xyz) >= 3:
        try:
            tri = Delaunay(surf_xyz[:, :2])
            tri_ids = ids_surface_atoms[tri.simplices]
            centers_surface_triangles = surf_xyz[tri.simplices].mean(axis=1)
            normals_surface_triangles = np.tile([0, 0, 1.0], (len(tri.simplices), 1))
        except Exception:
            pass  # ignore coplanar cases

    return {
        "ids_surface_atoms": ids_surface_atoms,
        "positions_surface_atoms": surf_xyz,
        "normals_surface_atoms": normals_surface_atoms,
        "ids_surface_edges": ids_surface_atoms[edge_pairs] if len(edge_pairs) else np.empty((0, 2), int),
        "centers_surface_edges": centers_surface_edges,
        "normals_surface_edges": normals_surface_edges,
        "ids_surface_triangles": tri_ids,
        "centers_surface_triangles": centers_surface_triangles,
        "normals_surface_triangles": normals_surface_triangles}


class MyCluster(Cluster):
    """ClusKit Cluster"""
    def get_surface_atoms(self, mask=False):
        if self.has('surface'):
            ids_surface_atoms = self.arrays['surface']
        else:
            summary = simple_delaunator(self.get_positions(), self.max_bondlength)
            ids_surface_atoms = summary["ids_surface_atoms"]
            self.arrays['surface'] = ids_surface_atoms

            # store sites
            self.site_surface_atom_ids[1] = ids_surface_atoms
            self.zero_site_positions[1] = summary["positions_surface_atoms"]
            self.adsorption_vectors[1] = summary["normals_surface_atoms"]

            self.site_surface_atom_ids[2] = summary["ids_surface_edges"]
            self.zero_site_positions[2] = summary["centers_surface_edges"]
            self.adsorption_vectors[2] = summary["normals_surface_edges"]

            self.site_surface_atom_ids[3] = summary["ids_surface_triangles"]
            self.zero_site_positions[3] = summary["centers_surface_triangles"]
            self.adsorption_vectors[3] = summary["normals_surface_triangles"]

        if mask:
            surface_mask = np.zeros(len(self), dtype=bool)
            surface_mask[ids_surface_atoms] = True
            return surface_mask
        return ids_surface_atoms

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
    return Atoms(
        symbols=symbols,
        scaled_positions=scaled_positions,
        cell=cell,
        pbc=True
    )

def generate_oer_adsorbates():
    """Return a list of ASE Atoms objects for the OER reaction"""
    return [
        Atoms(symbols='X', positions=[(0, 0, 0)],
              info={'adsorbate': '*', 'energy': 0}),
        Atoms(symbols="XO", positions=[(0, 0, 0), (0, 0, 2.0)],
              info={"adsorbate": "*O", "energy": 0}),
        Atoms(symbols="XOH", positions=[(0, 0, 0), (0, 0, 2.0), (0.1, 0.9, 2.9)],
              info={"adsorbate": "*OH", "energy": 0}),
        Atoms(symbols="XOOH", positions=[(0, 0, 0), (0, 0, 2.0), (1.2, -0.2, 2.8), (0.8, -0.2, 3.7)],
              info={"adsorbate": "*OOH", "energy": 0})
    ]

def generate_adsorbed_structures(reaction, pathway):
    """Generate surface + adsorbate structures"""
    site_map = {1: 'top', 2: 'bridge', 3: 'hollow'}
    with open('input_structures.json', 'r') as f:
        data = json.load(f)
    slab_data = Slab.from_dict(data[0])
    slab_data.sort()
    clean_slab = pmg_to_ase(slab_data)
    slab = MyCluster(clean_slab)
    clean_slab.set_constraint(FixAtoms(slab.get_nonsurface_atoms()))
    adsorption_sets = []

    if reaction == 'OER':
        adsorbates = generate_oer_adsorbates()
        for site_type in [1, 2, 3]:
            slab.get_sites(sitetype=site_type)
            slab.get_sites_descriptor(sitetype=site_type)
            unique_list = slab.get_unique_sites(sitetype=site_type, idx=[])
            zero_sites = slab.zero_site_positions[site_type]
            ads_vectors = slab.adsorption_vectors[site_type]
            for unique_site in unique_list:
                adsorb_set = []
                for ad in adsorbates:
                    clean = clean_slab.copy()
                    if ad.info['adsorbate'] == '*':
                        clean.info['site'] = site_map[site_type]
                        clean.info['adsorbate_collection'] = unique_site
                        clean.info['adsorbate'] = '*'
                        adsorb_set.append(clean)
                    else:
                        adsorbed = clean + utils.place_molecule_on_site(ad.copy(), zero_sites[unique_site], ads_vectors[unique_site])
                        if not has_reasonable_distances(adsorbed):
                            break
                        adsorbed.info['site'] = site_map[site_type]
                        adsorbed.info['adsorbate_collection'] = unique_site
                        adsorbed.info['adsorbate'] = ad.info['adsorbate']
                        adsorb_set.append(adsorbed)
                if len(adsorb_set) < len(adsorbates):
                    print('Adsorb set {} failed to build, discarding'.format(unique_site))
                    continue
                adsorption_sets.append(adsorb_set)


    elif reaction == 'CO2RR':
        """CO2 electroreduction reaction pathways on metal surfaces.

        Provides a literature-grounded library of CO2RR intermediates and reaction
        pathways (CHE model, Nørskov group and follow-up work) that can be placed on
        *any* ASE Atoms surface slab.

        Pathways implemented
        --------------------
        - ``'co2_to_co'``    : CO2 → *COOH → *CO → CO(g)            (Au, Ag, Zn)
        - ``'co2_to_hcooh'`` : CO2 → *OCHO → HCOOH(aq)              (formate, Pd, In)
        - ``'co_to_ch4'``    : *CO → *CHO → *CHOH → *CH2 → *CH3 → CH4(g)  (Cu)
        - ``'co_to_ch3oh'``  : *CO → *CHO → *CHOH → *CH2OH → CH3OH(g)     (Cu)
        - ``'co2_to_ch4'``   : CO2 → CH4 full pathway on Cu
        - ``'co2_to_ch3oh'`` : CO2 → CH3OH full pathway on Cu
        - ``'co2_to_c2h4'``  : 2 *CO → *OCCO → … → C2H4(g)          (Cu C–C coupling)

        Each intermediate is an ASE ``Atoms`` object (binding atom at index 0) that
        can be placed on a slab via :func:`place_intermediate`.

        References
        ----------
        Peterson et al. *Energy Environ. Sci.* **3**, 1311 (2010).
        Kuhl et al. *J. Am. Chem. Soc.* **136**, 14107 (2014).
        Montoya et al. *ChemSusChem* **8**, 2180 (2015).
        Goodpaster et al. *J. Phys. Chem. Lett.* **7**, 1471 (2016).
        """

        from ase import Atoms
        from ase.build import add_adsorbate
        from dataclasses import dataclass, field

        # ---------------------------------------------------------------------------
        # Adsorbate geometry library
        # All molecules have the surface-binding atom at index 0.
        # Bond lengths are equilibrium gas-phase values; DFT relaxation will adjust.
        # ---------------------------------------------------------------------------
        def _co() -> Atoms:
            """*CO — C-down (atop binding on most metals)."""
            return Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.15]])

        def _cooh() -> Atoms:
            """*COOH (carboxyl) — C-down, bidentate capable.
            Planar: C at origin, C=O pointing up, C-OH in-plane.
            """
            return Atoms(
                "COOH",
                positions=[
                    [0.00, 0.00, 0.00],  # C  (binds surface)
                    [0.00, 0.00, 1.22],  # O  (double-bond oxygen)
                    [1.10, 0.00, -0.60],  # O  (hydroxyl oxygen)
                    [1.94, 0.00, -0.12],  # H
                ],
            )

        def _ocho() -> Atoms:
            """*OCHO (formate) — O-down bidentate or monodentate.
            Monodentate: O at origin; formate oriented upright.
            """
            return Atoms(
                "OCHO",
                positions=[
                    [0.00, 0.00, 0.00],  # O  (binds surface)
                    [0.00, 0.00, 1.30],  # C
                    [0.00, 1.10, 1.85],  # O  (second O of formate)
                    [0.00, -0.95, 1.98],  # H
                ],
            )

        def _co2_ads() -> Atoms:
            """*CO2 — weakly adsorbed, bent (activated), O-C-O ~125°."""
            angle_rad = np.radians(125.0 / 2)
            d = 1.20  # C-O bond
            return Atoms(
                "OCO",
                positions=[
                    [0.00, 0.00, 0.00],  # C (binds)
                    [d * np.sin(angle_rad), 0, d * np.cos(angle_rad)],  # O1
                    [-d * np.sin(angle_rad), 0, d * np.cos(angle_rad)],  # O2
                ],
            )

        def _cho() -> Atoms:
            """*CHO (formyl) — C-down."""
            return Atoms(
                "CHO",
                positions=[
                    [0.00, 0.00, 0.00],  # C  (binds)
                    [0.00, 1.09, 0.63],  # H
                    [1.05, 0.00, 0.85],  # O
                ],
            )

        def _choh() -> Atoms:
            """*CHOH — C-down."""
            return Atoms(
                "CHOH",
                positions=[
                    [0.00, 0.00, 0.00],  # C  (binds)
                    [0.00, 1.09, 0.70],  # H  (on C)
                    [1.23, 0.00, 0.65],  # O
                    [1.85, 0.00, 1.38],  # H  (on O)
                ],
            )

        def _ch2o() -> Atoms:
            """*CH2O (formaldehyde adsorbed) — C-down, η1 mode."""
            return Atoms(
                "CH2O",
                positions=[
                    [0.00, 0.00, 0.00],  # C  (binds)
                    [0.94, 0.00, 0.59],  # H
                    [-0.94, 0.00, 0.59],  # H
                    [0.00, 1.10, 0.60],  # O
                ],
            )

        def _ch2oh() -> Atoms:
            """*CH2OH — C-down."""
            return Atoms(
                "CH2OH",
                positions=[
                    [0.00, 0.00, 0.00],  # C  (binds)
                    [0.95, 0.55, 0.62],  # H
                    [-0.95, 0.55, 0.62],  # H
                    [0.00, -1.22, 0.62],  # O
                    [0.00, -1.90, 1.35],  # H  (on O)
                ],
            )

        def _ch() -> Atoms:
            """*CH — C-down (hollow site preferred)."""
            return Atoms("CH", positions=[[0, 0, 0], [0, 0, 1.09]])

        def _ch2() -> Atoms:
            """*CH2 — C-down."""
            return Atoms(
                "CH2",
                positions=[
                    [0.00, 0.00, 0.00],
                    [0.94, 0.00, 0.70],
                    [-0.94, 0.00, 0.70],
                ],
            )

        def _ch3() -> Atoms:
            """*CH3 — C-down (atop)."""
            return Atoms(
                "CH3",
                positions=[
                    [0.00, 0.00, 0.00],
                    [0.00, 1.03, 0.70],
                    [0.89, -0.51, 0.70],
                    [-0.89, -0.51, 0.70],
                ],
            )

        def _oh() -> Atoms:
            """*OH — O-down."""
            return Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.97]])

        def _h() -> Atoms:
            """*H — single hydrogen."""
            return Atoms("H", positions=[[0, 0, 0]])

        def _occo() -> Atoms:
            """*OCCO (oxalyl, CO dimer) — first O binds surface."""
            return Atoms(
                "OCCO",
                positions=[
                    [0.00, 0.00, 0.00],  # O1 (binds)
                    [0.00, 0.00, 1.25],  # C1
                    [0.00, 1.35, 1.25],  # C2
                    [0.00, 1.35, 2.50],  # O2
                ],
            )

        def _ccho() -> Atoms:
            """*CCHO — C-down, C2 species."""
            return Atoms(
                "CCHO",
                positions=[
                    [0.00, 0.00, 0.00],  # C1 (binds)
                    [0.00, 1.35, 0.65],  # C2
                    [0.00, 2.40, 1.30],  # O
                    [0.00, 1.35, 1.76],  # H
                ],
            )

        def _c2h4() -> Atoms:
            """C2H4 (ethylene) — di-σ mode, C-C bridging; C1 at origin (mol_index=0)."""
            return Atoms(
                "C2H4",
                positions=[
                    [0.00, 0.00, 0.00],  # C1 (binds, at origin)
                    [1.34, 0.00, 0.00],  # C2 (binds, 2nd site)
                    [-0.56, 0.92, 0.60],  # H
                    [-0.56, -0.92, 0.60],  # H
                    [1.90, 0.92, 0.60],  # H
                    [1.90, -0.92, 0.60],  # H
                ],
            )

        # Registry: name → factory function
        _ADSORBATE_REGISTRY: dict[str, callable] = {
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

        def get_adsorbate(name: str) -> Atoms:
            """Return a fresh copy of the named adsorbate geometry.

            Args:
                name: Key from the adsorbate registry.  Call
                      :func:`list_adsorbates` to see all available names.

            Returns:
                :class:`~ase.Atoms` with the surface-binding atom at index 0.

            Raises:
                KeyError: If *name* is not in the registry.
            """
            if name not in _ADSORBATE_REGISTRY:
                raise KeyError(
                    f"Unknown adsorbate '{name}'. "
                    f"Available: {sorted(_ADSORBATE_REGISTRY)}"
                )
            return _ADSORBATE_REGISTRY[name]()

        def list_adsorbates() -> list[str]:
            """Return the sorted list of all registered adsorbate names."""
            return sorted(_ADSORBATE_REGISTRY)

        # ---------------------------------------------------------------------------
        # Reaction pathway data model
        # ---------------------------------------------------------------------------
        @dataclass
        class ReactionStep:
            """One elementary step in a CO2RR pathway.

            Attributes:
                reactant: Name of the surface intermediate before this step.
                product: Name of the surface intermediate after this step.
                step_type: ``'electrochemical'`` (H⁺ + e⁻ transfer) or
                           ``'chemical'`` (no charge transfer).
                electrons: Number of electrons transferred (negative = gained).
                protons: Number of protons transferred.
                released: Species desorbed/released, e.g. ``['H2O']`` or ``['CO']``.
                notes: Free-text annotation (e.g. potential-limiting step).
            """
            reactant: str
            product: str
            step_type: str = "electrochemical"
            electrons: int = -1
            protons: int = 1
            released: list[str] = field(default_factory=list)
            notes: str = ""

        @dataclass
        class ReactionPathway:
            """A named CO2RR pathway with ordered elementary steps.

            Attributes:
                name: Unique identifier (used by :func:`get_pathway`).
                description: Human-readable summary.
                steps: Ordered list of :class:`ReactionStep` objects.
                selectivity_metals: Metals where this pathway dominates in DFT/expt.
                overall_reaction: Balanced overall equation string.
            """
            name: str
            description: str
            steps: list[ReactionStep]
            selectivity_metals: list[str] = field(default_factory=list)
            overall_reaction: str = ""

            @property
            def intermediates(self) -> list[str]:
                """All surface intermediates in pathway order (no duplicates)."""
                seen: list[str] = []
                for step in self.steps:
                    if step.reactant not in seen:
                        seen.append(step.reactant)
                last = self.steps[-1].product
                if last not in seen:
                    seen.append(last)
                return seen

        # ---------------------------------------------------------------------------
        # Pathway definitions
        # ---------------------------------------------------------------------------
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

        # ---------------------------------------------------------------------------
        # Public API — pathway access
        # ---------------------------------------------------------------------------
        def get_pathway(name: str) -> ReactionPathway:
            """Return the named :class:`ReactionPathway`.

            Args:
                name: One of the pathway keys.  Call :func:`list_pathways` to see all.

            Raises:
                KeyError: If *name* is not registered.
            """
            if name not in _PATHWAYS:
                raise KeyError(
                    f"Unknown pathway '{name}'. "
                    f"Available: {list_pathways()}"
                )
            return _PATHWAYS[name]

        def list_pathways() -> list[str]:
            """Return the sorted list of all registered pathway names."""
            return sorted(_PATHWAYS)

        # ---------------------------------------------------------------------------
        # Placement on arbitrary surfaces
        # ---------------------------------------------------------------------------
        def _find_surface_atom_z(slab: Atoms, tol: float = 1.5) -> float:
            """Return z-coordinate of the topmost surface layer."""
            z_max = slab.get_positions()[:, 2].max()
            return z_max

        def place_intermediate(
                slab: Atoms,
                intermediate_name: str,
                site: str | tuple = "ontop",
                height: float = 2.0,
                offset: tuple[float, float] = (0.0, 0.0),
                mol_index: int = 0,
        ) -> Atoms:
            """Place a CO2RR intermediate on a surface slab.

            Args:
                slab: ASE Atoms surface slab (must have a cell).
                intermediate_name: Name of the adsorbate (see :func:`list_adsorbates`).
                site: Adsorption site — ``'ontop'``, ``'bridge'``, ``'hollow'``, or a
                      ``(x, y)`` tuple of fractional coordinates.
                height: Height in Å above the topmost surface atom.
                offset: ``(dx, dy)`` fractional offset from the site centre.
                mol_index: Index of the binding atom in the adsorbate Atoms object
                           (default 0, matching all built-in geometries here).

            Returns:
                A new :class:`~ase.Atoms` object: slab + adsorbate.
            """
            mol = get_adsorbate(intermediate_name)
            slab_with_ads = slab.copy()

            if isinstance(site, tuple) and len(site) == 2 and not isinstance(site[0], str):
                # Fractional coordinates → Cartesian x, y
                cell = slab.get_cell()
                xy = site[0] * cell[0, :2] + site[1] * cell[1, :2]
                position = (float(xy[0] + offset[0] * cell[0, 0]),
                            float(xy[1] + offset[1] * cell[1, 1]))
                add_adsorbate(slab_with_ads, mol, height=height, position=position, mol_index=mol_index)
            else:
                add_adsorbate(slab_with_ads, mol, height=height, position=site, offset=offset, mol_index=mol_index)

            return slab_with_ads

        def place_pathway(
                slab: Atoms,
                pathway_name: str,
                height: float = 2.0,
                site: str = "ontop",
        ) -> dict[str, Atoms]:
            """Generate one slab structure per intermediate in a CO2RR pathway.

            Each structure is the clean slab with exactly one intermediate adsorbed at
            the specified site.  Useful for building an input set for DFT single-points
            or NEB calculations.

            Args:
                slab: Clean surface slab (ASE Atoms, with cell).
                pathway_name: Name of the pathway (see :func:`list_pathways`).
                height: Adsorption height above the surface in Å.
                site: Adsorption site for all intermediates — ``'ontop'``,
                      ``'bridge'``, or ``'hollow'``.

            Returns:
                ``{intermediate_name: Atoms}`` ordered dict mapping each intermediate
                to its corresponding slab+adsorbate structure.  The ``'CO'`` bare-
                surface marker used to close some pathways is omitted.
            """
            pathway = get_pathway(pathway_name)
            structures: dict[str, Atoms] = {}
            skip = {"CO"}  # used as "bare surface" sentinel in some pathways' last step

            for name in pathway.intermediates:
                if name in skip and name not in _ADSORBATE_REGISTRY:
                    continue
                if name not in _ADSORBATE_REGISTRY:
                    continue
                structures[name] = place_intermediate(slab, name,
                                                      site=site, height=height)
            return structures

        for site in ['ontop', 'bridge', 'hollow']:
            adsorption_sets.append(place_pathway(slab=clean_slab, pathway_name=pathway, site=site))


    elif reaction == 'NOXRR':
        """NOx electroreduction reaction pathways on metal surfaces.

        Provides a library of NOx reduction intermediates and reaction pathways
        (CHE model) that can be placed on *any* ASE Atoms surface slab.

        NOx covered: NO, NO₂, NO₃⁻ as starting species.

        Pathways implemented
        --------------------
        - ``'no_dissociative'``  : *NO → *N + *O → N₂(g)              (Ru, Rh, Ir)
        - ``'no_to_nh3_noh'``   : *NO → *NOH → *N → *NH₂ → NH₃       (Cu, Fe)
        - ``'no_to_nh3_nhoh'``  : *NO → *NOH → *NHOH → *NH₂OH → NH₃  (Cu, hydroxylamine route)
        - ``'no_to_n2o'``        : 2*NO → *ONNO → N₂O + *O            (Pt, Pd automotive)
        - ``'no2_to_no'``        : *NO₂ → *NO + *O                     (prereduction step)
        - ``'no3_to_nh3'``       : *NO₃ → *NO₂ → *NO → … → NH₃        (eNO3RR, Cu)
        - ``'no3_to_n2'``        : *NO₃ → *NO₂ → *NO → *N → N₂        (eNO3RR, Ru)

        References
        ----------
        Gao et al. *Nat. Chem.* **9**, 547 (2017).
        Liu et al. *Nat. Commun.* **12**, 5797 (2021).
        Wang et al. *J. Am. Chem. Soc.* **142**, 5702 (2020).
        van 't Veer et al. *J. Phys. Chem. C* **124**, 22 (2020).
        Pérez-Ramírez & López *Nat. Catal.* **2**, 971 (2019).
        """

        from ase import Atoms
        from ase.build import add_adsorbate
        from dataclasses import dataclass, field

        # ---------------------------------------------------------------------------
        # Adsorbate geometry library (binding atom at index 0)
        # ---------------------------------------------------------------------------
        def _no() -> Atoms:
            """*NO — N-down (preferred on most transition metals)."""
            return Atoms("NO", positions=[[0, 0, 0], [0, 0, 1.15]])

        def _no2() -> Atoms:
            """*NO₂ — N-down, bent (O–N–O ≈ 115°)."""
            half = np.radians(115.0 / 2)
            d = 1.20  # N–O bond
            return Atoms(
                "NO2",
                positions=[
                    [0.00, 0.00, 0.00],
                    [d * np.sin(half), 0, d * np.cos(half)],
                    [-d * np.sin(half), 0, d * np.cos(half)],
                ],
            )

        def _no3() -> Atoms:
            """*NO₃ — N-down, planar nitrate (D₃ₕ, N–O = 1.24 Å) slightly tilted up."""
            d = 1.24
            h = 0.40  # slight tilt upward for O atoms
            return Atoms(
                "NO3",
                positions=[
                    [0.00, 0.00, 0.00],  # N (binds)
                    [0.00, d, h],  # O1
                    [d * np.sin(np.radians(120)), -d * 0.5, h],  # O2
                    [-d * np.sin(np.radians(120)), -d * 0.5, h],  # O3
                ],
            )

        def _noh() -> Atoms:
            """*NOH — N-down, O–H bond (first H on O)."""
            return Atoms(
                "NOH",
                positions=[
                    [0.00, 0.00, 0.00],  # N (binds)
                    [0.00, 0.00, 1.20],  # O
                    [0.90, 0.00, 1.70],  # H (on O)
                ],
            )

        def _hno() -> Atoms:
            """*HNO — N-down, N–H bond (first H on N)."""
            return Atoms(
                "HNO",
                positions=[
                    [0.00, 0.00, 0.00],  # N (binds)
                    [0.00, 1.01, 0.42],  # H (on N)
                    [1.10, 0.00, 0.75],  # O
                ],
            )

        def _n2o2() -> Atoms:
            """*ONNO (cis-hyponitrite dimer, N-down) — O=N–N=O bridge species.
            First N binds; O–N–N–O roughly linear, slightly tilted.
            """
            return Atoms(
                "N2O2",
                positions=[
                    [0.00, 0.00, 0.00],  # N1 (binds)
                    [0.00, 0.00, 1.18],  # O1
                    [1.30, 0.00, 0.00],  # N2
                    [1.30, 0.00, 1.18],  # O2
                ],
            )

        def _n2o() -> Atoms:
            """*N₂O — N-down, linear (N≡N–O, terminal N binds)."""
            return Atoms(
                "N2O",
                positions=[
                    [0.00, 0.00, 0.00],  # N1 (binds, terminal)
                    [0.00, 0.00, 1.13],  # N2
                    [0.00, 0.00, 2.27],  # O
                ],
            )

        def _n_ads() -> Atoms:
            """*N — atomic nitrogen (hollow site preferred)."""
            return Atoms("N", positions=[[0, 0, 0]])

        def _nh() -> Atoms:
            """*NH — N-down."""
            return Atoms("NH", positions=[[0, 0, 0], [0, 0, 1.01]])

        def _nh2() -> Atoms:
            """*NH₂ — N-down."""
            return Atoms(
                "NH2",
                positions=[
                    [0.00, 0.00, 0.00],
                    [0.82, 0.00, 0.56],
                    [-0.82, 0.00, 0.56],
                ],
            )

        def _nh3() -> Atoms:
            """*NH₃ — N-down (tetrahedral)."""
            return Atoms(
                "NH3",
                positions=[
                    [0.00, 0.00, 0.00],
                    [0.00, 0.94, 0.34],
                    [0.82, -0.47, 0.34],
                    [-0.82, -0.47, 0.34],
                ],
            )

        def _nhoh() -> Atoms:
            """*NHOH — N-down, both N–H and O–H bonds present."""
            return Atoms(
                "NHOH",
                positions=[
                    [0.00, 0.00, 0.00],  # N (binds)
                    [0.00, 0.95, 0.45],  # H (on N)
                    [1.25, 0.00, 0.65],  # O
                    [1.85, 0.00, 1.40],  # H (on O)
                ],
            )

        def _nh2oh() -> Atoms:
            """*NH₂OH (hydroxylamine) — N-down."""
            return Atoms(
                "NH2OH",
                positions=[
                    [0.00, 0.00, 0.00],  # N (binds)
                    [0.00, 0.95, 0.45],  # H1 (on N)
                    [-0.95, -0.35, 0.45],  # H2 (on N)
                    [1.22, 0.00, 0.75],  # O
                    [1.82, 0.00, 1.50],  # H (on O)
                ],
            )

        def _o_ads() -> Atoms:
            """*O — atomic oxygen."""
            return Atoms("O", positions=[[0, 0, 0]])

        def _oh() -> Atoms:
            """*OH — O-down."""
            return Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.97]])

        def _h2o() -> Atoms:
            """*H₂O — O-down (physisorbed; typically desorbs above 200 K)."""
            return Atoms(
                "H2O",
                positions=[
                    [0.00, 0.00, 0.00],  # O (binds)
                    [0.76, 0.00, 0.59],  # H
                    [-0.76, 0.00, 0.59],  # H
                ],
            )

        # Registry
        _ADSORBATE_REGISTRY: dict[str, callable] = {
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

        def get_adsorbate(name: str) -> Atoms:
            """Return a fresh copy of the named NOx-reduction adsorbate.

            Args:
                name: Key from the adsorbate registry.  Call
                      :func:`list_adsorbates` to see all available names.

            Returns:
                :class:`~ase.Atoms` with the surface-binding atom at index 0.

            Raises:
                KeyError: If *name* is not in the registry.
            """
            if name not in _ADSORBATE_REGISTRY:
                raise KeyError(
                    f"Unknown adsorbate '{name}'. "
                    f"Available: {sorted(_ADSORBATE_REGISTRY)}"
                )
            return _ADSORBATE_REGISTRY[name]()

        def list_adsorbates() -> list[str]:
            """Return the sorted list of all registered adsorbate names."""
            return sorted(_ADSORBATE_REGISTRY)

        # ---------------------------------------------------------------------------
        # Pathway data model (identical interface to co2_reduction)
        # ---------------------------------------------------------------------------
        @dataclass
        class ReactionStep:
            """One elementary step in a NOx reduction pathway.

            Attributes:
                reactant: Name of the surface intermediate before this step.
                product: Name of the surface intermediate after this step.
                step_type: ``'electrochemical'`` (H⁺ + e⁻) or ``'chemical'``.
                electrons: Number of electrons transferred (negative = gained).
                protons: Number of protons transferred.
                released: Species desorbed/released (e.g. ``['H2O']``, ``['N2']``).
                notes: Free-text annotation.
            """
            reactant: str
            product: str
            step_type: str = "electrochemical"
            electrons: int = -1
            protons: int = 1
            released: list[str] = field(default_factory=list)
            notes: str = ""

        @dataclass
        class ReactionPathway:
            """A named NOx reduction pathway.

            Attributes:
                name: Unique identifier.
                description: Human-readable summary.
                steps: Ordered elementary steps.
                selectivity_metals: Metals where this pathway dominates.
                overall_reaction: Balanced overall equation string.
            """
            name: str
            description: str
            steps: list[ReactionStep]
            selectivity_metals: list[str] = field(default_factory=list)
            overall_reaction: str = ""

            @property
            def intermediates(self) -> list[str]:
                """All surface intermediates in pathway order (no duplicates)."""
                seen: list[str] = []
                for step in self.steps:
                    if step.reactant not in seen:
                        seen.append(step.reactant)
                last = self.steps[-1].product
                if last not in seen:
                    seen.append(last)
                return seen

        # ---------------------------------------------------------------------------
        # Pathway definitions
        # ---------------------------------------------------------------------------
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

        # ---------------------------------------------------------------------------
        # Public API — pathway access
        # ---------------------------------------------------------------------------
        def get_pathway(name: str) -> ReactionPathway:
            """Return the named :class:`ReactionPathway`.

            Raises:
                KeyError: If *name* is not registered.
            """
            if name not in _PATHWAYS:
                raise KeyError(
                    f"Unknown pathway '{name}'. "
                    f"Available: {list_pathways()}"
                )
            return _PATHWAYS[name]

        def list_pathways() -> list[str]:
            """Return the sorted list of all registered pathway names."""
            return sorted(_PATHWAYS)

        # ---------------------------------------------------------------------------
        # Placement on arbitrary surfaces (same interface as co2_reduction)
        # ---------------------------------------------------------------------------
        def place_intermediate(
                slab: Atoms,
                intermediate_name: str,
                site: str | tuple = "ontop",
                height: float = 2.0,
                offset: tuple[float, float] = (0.0, 0.0),
                mol_index: int = 0,
        ) -> Atoms:
            """Place a NOx-reduction intermediate on a surface slab.

            Args:
                slab: ASE Atoms surface slab (must have a cell).
                intermediate_name: Name of the adsorbate (see :func:`list_adsorbates`).
                site: Adsorption site — ``'ontop'``, ``'bridge'``, or a
                      ``(x, y)`` tuple of fractional coordinates.
                height: Height in Å above the topmost surface atom.
                offset: ``(dx, dy)`` fractional offset from the site centre.
                mol_index: Index of the binding atom in the adsorbate (default 0).

            Returns:
                A new :class:`~ase.Atoms` object: slab + adsorbate.
            """
            mol = get_adsorbate(intermediate_name)
            slab_with_ads = slab.copy()

            if isinstance(site, tuple) and not isinstance(site[0], str):
                cell = slab.get_cell()
                xy = site[0] * cell[0, :2] + site[1] * cell[1, :2]
                position = (float(xy[0] + offset[0] * cell[0, 0]),
                            float(xy[1] + offset[1] * cell[1, 1]))
                add_adsorbate(slab_with_ads, mol, height=height, position=position, mol_index=mol_index)
            else:
                add_adsorbate(slab_with_ads, mol, height=height, position=site, offset=offset, mol_index=mol_index)

            return slab_with_ads

        def place_pathway(
                slab: Atoms,
                pathway_name: str,
                height: float = 2.0,
                site: str = "ontop",
        ) -> dict[str, Atoms]:
            """Generate one slab+adsorbate structure per intermediate in a NOx pathway.

            Args:
                slab: Clean surface slab.
                pathway_name: Name of the pathway (see :func:`list_pathways`).
                height: Adsorption height above the surface in Å.
                site: Adsorption site for all intermediates.

            Returns:
                ``{intermediate_name: Atoms}`` for each registered intermediate in the
                pathway.  Bare-surface sentinel intermediates are skipped.
            """
            pathway = get_pathway(pathway_name)
            structures: dict[str, Atoms] = {}

            for name in pathway.intermediates:
                if name not in _ADSORBATE_REGISTRY:
                    continue
                structures[name] = place_intermediate(slab, name, site=site, height=height)
            return structures

        for site in ['ontop', 'bridge', 'hollow']:
            adsorption_sets.append(place_pathway(slab=clean_slab, pathway_name=pathway, site=site))

    return adsorption_sets

def run_relaxation(ML_model, calc, fmax, max_steps, reaction, pathway):
    adsorption_sets = generate_adsorbed_structures(reaction=reaction, pathway=pathway)
    relaxed_adsorption_sets = []
    num_failed = 0
    for adsorb_set in adsorption_sets:
        relaxed_adsorb_set = []
        for adsorbed in adsorb_set:
            adsorbed.calc = calc
            relax = BFGSLineSearch(adsorbed, maxstep=0.1, logfile='opt.log')
            relax.run(fmax=fmax, steps=max_steps)
            if not relax.converged:
                num_failed += 1
                break
            adsorbed.info['{}_energy'.format(str(ML_model).lower())] = adsorbed.get_potential_energy()
            relaxed_adsorb_set.append(jsonio.encode(adsorbed))
        if len(relaxed_adsorb_set) < len(adsorb_set):
            continue
        relaxed_adsorption_sets.append(relaxed_adsorb_set)

    output = dict({'structures': relaxed_adsorption_sets})

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

    run_relaxation(ML_model=args.ML_model, calc=calc, fmax=args.fmax, max_steps=args.max_steps,
                   reaction=args.reaction, pathway=args.pathway)
