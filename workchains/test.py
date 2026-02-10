import os
import yaml
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from aiida.orm import Str, List, Dict, load_code, StructureData
from aiida.engine import WorkChain
from aiida.plugins import WorkflowFactory
from uvsib.db.tables import DBSurface
from uvsib.db.utils import get_structure_uuid_surface_id, query_by_columns
from uvsib.codes.vasp.workchains import construct_vasp_builder
from uvsib.workflows import settings
from uvsib.workchains.utils import get_code, get_model_device
from scipy.spatial import Delaunay, distance_matrix
from cluskit import Cluster, utils
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import Slab

from pymatgen.io.vasp import Poscar
from ase.io import write

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
            normals_surface_edges[i] = n / norm if norm > 1e-10 else [0, 0, 0.0]

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


chemical_formula = "InSe"
structure_surface_rows = get_structure_uuid_surface_id(chemical_formula)
adsorption_sets = dict()

def generate_adsorbed_structures():
    """ do the work """
    pos_c = 0
    site_map = {1: 'top', 2: 'bridge', 3: 'hollow'}
    adsorb_set = list()

    for row in structure_surface_rows:
        structure_uuid = row[0]
        surface_id = row[1]
        slab_row = query_by_columns(DBSurface, {'id': surface_id})[0]
        uuid_str = str(structure_uuid)
        adsorption_sets[uuid_str] = list()
        slab_data = Slab.from_dict(slab_row.slab)
        slab_data.sort()
        print(slab_data.miller_index)
        #print(Poscar(slab_data))
        clean_slab = AseAtomsAdaptor().get_atoms(slab_data)
        slab = MyCluster(clean_slab)
        fixed_ids = slab.get_nonsurface_atoms()
        clean_slab.set_constraint(FixAtoms(indices=fixed_ids))
        adsorbates = generate_adsorbates("OER")
        for site_type in [1, 2, 3]:
            #print(f" site type: {site_type}")
            slab = MyCluster(clean_slab)
            all_sites = slab.get_sites(sitetype=site_type)
            if len(slab.zero_site_positions[site_type]) == 0: #TODO HM
                continue
            sites_descriptors = slab.get_sites_descriptor(sitetype=site_type)
            unique_list = slab.get_unique_sites(sitetype=site_type, idx=[])
            zero_sites = slab.zero_site_positions[site_type]
            #print(f"zero sites: {zero_sites}")
            #print(unique_list)
            ads_vectors = slab.adsorption_vectors[site_type]
            for unique_idx in unique_list:
                for ad in adsorbates:
                    pos_c = pos_c + 1
                    clean = clean_slab.copy()
                    clean.info['site'] = site_map[site_type]
                    clean.info['adsorbate_collection'] = unique_idx
                    if ad.info['adsorbate'] == '*':
                        clean.info['site'] = site_map[site_type]
                        clean.info['adsorbate'] = '*'
                        adsorb_set.append(clean)
                    else:
                        #print(zero_sites[unique_idx])
                        vec = ads_vectors[unique_idx]
                        if np.linalg.norm(vec) < 1e-8: #TODO HM
                            continue
                        adsorbed = clean + utils.place_molecule_on_site(ad, zero_sites[unique_idx], ads_vectors[unique_idx])
#                        write("POSCAR"+str(pos_c)+".vasp", adsorbed, format="vasp")
                        adsorbed.info['site'] = site_map[site_type]
                        adsorbed.info['adsorbate'] = ad.info['adsorbate']

                        adsorb_set.append(adsorbed)
    return adsorb_set

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

    elif reaction == "CO2RR":
        return []

    else:  # not implemented case
        raise NotImplementedError('Reaction {} not implemented'.format(reaction))

adsorb_set = generate_adsorbed_structures()
