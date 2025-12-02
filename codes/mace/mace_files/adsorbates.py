import json
import argparse
from collections import defaultdict
import numpy as np
from ase import Atoms
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.constraints import FixAtoms
from cluskit import Cluster, utils
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import Slab
from scipy.spatial import distance_matrix, Delaunay
from mace.calculators import MACECalculator

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
    edge_pairs = edge_pairs[edge_pairs[:, 0] < edge_pairs[:, 1]]  # remove duplicates

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
    """
    Convert an ASE Atoms object to a pymatgen Structure dictionary
    """
    lattice = atoms.cell.array.tolist()
    symbols = atoms.get_chemical_symbols()
    frac_coords = atoms.get_scaled_positions().tolist()
    lattice_obj = Lattice(lattice)
    return Structure(lattice_obj,
                          symbols,
                          frac_coords,
                          coords_are_cartesian=False)

def pmg_to_ase(pmg_structure):
    """
    Convert a pymatgen Structure to an ASE Atoms object.
    """
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
    """Return a list of ASE Atoms objects for a given adsorbate label"""
    if reaction == "WS":
        return [
            Atoms("XH", positions=[(0, 0, 0), (0, 0, 1.0)],
                  info={"adsorbate": "H", "energy": -3.05}), #TODO calculate the energy for the isolated adsorbate
            Atoms("XO", positions=[(0, 0, 0), (0, 0, 3.1)],
                  info={"adsorbate": "O", "energy": -1.65}),
            Atoms("XOH", positions=[(0, 0, 0), (0, 0, 3.1), (0.1, 0.1, 4.1)],
                  info={"adsorbate": "OH", "energy": 7.55}),
        ]
    return []

def run_add_adsorbates(model_path, device, fmax, max_steps, reaction):
    """Run"""
    with open('input_structures.json', 'r') as f:
        data = json.load(f)
    slab_data = Slab.from_dict(data[0])
    clean_slab = pmg_to_ase(slab_data)
    clean_slab_energy = slab_data.energy
    slab = MyCluster(clean_slab)
    fixed_ids = slab.get_nonsurface_atoms()
    clean_slab.set_constraint(FixAtoms(indices=fixed_ids))

    adsorbates = generate_adsorbates(reaction)
    mace_calc = MACECalculator(model_paths=model_path, device=device)

    built_faces = []
    for ad in adsorbates:
        for site_type in [1, 2, 3]:
            zero_sites = slab.zero_site_positions[site_type]
            ads_vectors = slab.adsorption_vectors[site_type]
            if len(zero_sites) == 0:
                continue
            for site_pos, site_vec in zip(zero_sites, ads_vectors):
                adsorbed = clean_slab + utils.place_molecule_on_site(ad, site_pos, site_vec)
                built_faces.append([ad, adsorbed])

    relaxed_faces = []
    num_failed = 0
    tmp_data = defaultdict(list)
    for ad, atoms in built_faces:
        atoms.calc = mace_calc
        relax = BFGSLineSearch(atoms, maxstep=0.1, logfile='opt.log')
        try:
            relax.run(fmax=fmax, steps=max_steps)
            relaxed_structure = ase_to_pmg(atoms).as_dict()
            ad_energy = float(atoms.get_potential_energy()) - clean_slab_energy - ad.info["energy"]
            tmp_data[ad.info["adsorbate"]].append([relaxed_structure, ad_energy])
        except Exception:
            num_failed += 1
            continue
    to_dump = {}
    percentage_to_select = 0.5
    for ad in tmp_data.keys():
        tmp_data[ad].sort(key=lambda x: x[1])
        n_selected =  max(1, int(len(tmp_data[ad]) * percentage_to_select))
        to_dump[ad] =  tmp_data[ad][:n_selected]

    with open('output.json', 'w') as f:
        json.dump(to_dump, f)

    with open('total.txt', 'w') as f:
        f.write(str(len(relaxed_faces)))
    with open('failed.txt', 'w') as f:
        f.write(str(num_failed))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--fmax", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--reaction", type=str)
    args = parser.parse_args()

    run_add_adsorbates(args.model_path, args.device, args.fmax, args.max_steps, args.reaction)
