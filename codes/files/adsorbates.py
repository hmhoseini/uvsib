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

def generate_adsorbates(reaction):
    """Return a list of ASE Atoms objects for a given reaction"""
    if reaction == "OER":
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
    if reaction == "CO2RR":
        return []

    raise NotImplementedError('Reaction {} not implemented'.format(reaction))

def generate_adsorbed_structures(reaction):
    """Generate surface + adsorbate structures"""
    site_map = {1: 'top', 2: 'bridge', 3: 'hollow'}
    with open('input_structures.json', 'r') as f:
        data = json.load(f)
    slab_data = Slab.from_dict(data[0])
    slab_data.sort()
    clean_slab = pmg_to_ase(slab_data)
    slab = MyCluster(clean_slab)
    clean_slab.set_constraint(FixAtoms(slab.get_nonsurface_atoms()))
    adsorbates = generate_adsorbates(reaction)
    adsorption_sets = []
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
    return adsorption_sets

def run_relaxation(ML_model, calc, fmax, max_steps, reaction):
    adsorption_sets = generate_adsorbed_structures(reaction)
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

    run_relaxation(ML_model=args.ML_model, calc=calc, fmax=args.fmax, max_steps=args.max_steps, reaction=args.reaction)
