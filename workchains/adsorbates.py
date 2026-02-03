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


def read_yaml(file_path):
    """Read yaml file"""
    with open(file_path, 'r', encoding='utf8') as fhandle:
        data = yaml.safe_load(fhandle)
    return data


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


class AdsorbatesWorkChain(WorkChain):
    """Adsorbates WorkChain"""
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("chemical_formula", valid_type=Str)
        spec.input("ML_model", valid_type=Str)
        spec.input("reaction", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.generate_adsorbed_structures,
            cls.inspect_adsorbs,
            cls.store_results,
            cls.final_report
        )

        spec.exit_code(300,
            "ERROR_CALCULATION_FAILED",
            message="The calculation did not finish successfully"
        )
        spec.exit_code(
            301,
            "ERROR_NO_STRUCTURES_FOUND",
            message="No structures were found for the given formula."
        )

    def setup(self):
        """Setup and report"""
        self.report('Running {} Adsorbates WorkChain for {}'.format(self.inputs.chemical_formula.value,
                                                                    self.inputs.reaction.value))
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.ML_model = self.inputs.ML_model.value
        self.ctx.reaction = self.inputs.reaction.value
        self.ctx.structure_surface_rows = get_structure_uuid_surface_id(self.ctx.chemical_formula)
        if not self.ctx.structure_surface_rows:
            return self.exit_codes.ERROR_NO_STRUCTURES_FOUND
        self.ctx.protocol = read_yaml(os.path.join(settings.vasp_files_path, 'protocol.yaml'))
        self.ctx.potentials = read_yaml(os.path.join(settings.vasp_files_path, 'potentials.yaml'))
        self.ctx.vasp_code = load_code(settings.configs['codes']['VASP']['code_string'])
        self.ctx.adsorption_sets = dict()
        self.ctx.r2scan_adsorbates = dict()
        self.ctx.r2scan_adsorbates[self.ctx.reaction] = dict()

        return None

    def generate_adsorbed_structures(self):
        """ do the work """
        site_map = {1: 'top', 2: 'bridge', 3: 'hollow'}
        for row in self.ctx.structure_surface_rows:
            structure_uuid = row[0]
            surface_id = row[1]
            slab_row = query_by_columns(DBSurface, {'id': surface_id})[0]
            uuid_str = str(structure_uuid)
            self.ctx.adsorption_sets[uuid_str] = list()
            slab_data = Slab.from_dict(slab_row.structure)
            clean_slab = AseAtomsAdaptor().get_atoms(slab_data)
            slab = MyCluster(clean_slab)
            fixed_ids = slab.get_nonsurface_atoms()
            clean_slab.set_constraint(FixAtoms(indices=fixed_ids))
            adsorbates = self.generate_adsorbates(self.ctx.reaction)
            for site_type in [1, 2, 3]:
                slab = MyCluster(clean_slab)
                all_sites = slab.get_sites(sitetype=site_type)
                sites_descriptors = slab.get_sites_descriptor(sitetype=site_type)
                unique_list = slab.get_unique_sites(sitetype=site_type, idx=[])
                zero_sites = slab.zero_site_positions[site_type]
                ads_vectors = slab.adsorption_vectors[site_type]
                for idx, unique_site in enumerate(unique_list):
                    adsorb_set = list()
                    for ad in adsorbates:
                        clean = clean_slab.copy()
                        clean.info['site'] = site_map[site_type]
                        clean.info['adsorbate_collection'] = idx
                        if ad.info['adsorbate'] == '*':
                            clean.info['site'] = site_map[site_type]
                            clean.info['adsorbate'] = '*'
                            adsorb_set.append(clean)
                        else:
                            adsorbed = clean + utils.place_molecule_on_site(ad, zero_sites[idx], ads_vectors[idx])
                            adsorbed.info['site'] = site_map[site_type]
                            adsorbed.info['adsorbate'] = ad.info['adsorbate']
                            adsorb_set.append(adsorbed)
                    self.ctx.adsorption_sets[uuid_str].append(adsorb_set)

    def inspect_adsorbs(self):
        """Inspect Adsorbates WorkChain"""
        for row in self.ctx.structure_surface_rows:
            structure_uuid = row[0]
            uuid_str = str(structure_uuid)
            for set in self.ctx.adsorption_sets[uuid_str]:
                if len(set) != 4:
                    self.report('Complete set for adsorption set index {} failed to build, discarding '.format(uuid_str))
            if len(self.ctx.adsorption_sets) == 0:
                return self.exit_codes.ERROR_NO_STRUCTURES_FOUND

    def run_r2scan(self):
        """Run r2SCAN geometry optimization"""
        for row in self.ctx.structure_surface_rows:
            uuid_str = str(row[0])
            for idx, adsorption_set in enumerate(self.ctx.adsorption_sets[uuid_str]):
                for atoms in adsorption_set:
                    adsorbate = atoms.info['adsorbate']
                    structure = AseAtomsAdaptor().get_structure(atoms)
                    builder = construct_vasp_builder(StructureData(pymatgen=structure), self.ctx.protocol['r2SCAN'],
                                                                   self.ctx.potentials, self.ctx.vasp_code)
                    future = self.submit(builder)
                    self.to_context(**{f'final_adsorption_r2scan_{uuid_str}_{idx}_{adsorbate}': future})
                # break  # debug janK

    def inspect_r2scan(self):
        """Inspect r2SCAN adsorbates WorkChain"""
        failed_jobs = 0
        adsorbate_collection = dict()
        for row in self.ctx.structure_surface_rows:
            uuid_str = str(row[0])
            adsorbate_collection[uuid_str] = list()
            for idx in range(len(self.ctx.adsorption_sets[uuid_str])):
                if self.ctx.reaction == 'OER':
                    energy_set = dict()
                    structure_set = dict()
                    for adsorbate in ['*', '*O', '*OH', '*OOH']:
                        ad_chain = self.ctx[f'final_adsorption_r2scan_{uuid_str}_{idx}_{adsorbate}']
                        if not ad_chain.is_finished_ok:
                            failed_jobs += 1
                            self.report('Adsorb computation for {} failed in {}'.format(adsorbate, uuid_str))
                            continue
                        vasp_outputs = ad_chain.called[-1].outputs
                        energy_set[adsorbate] = vasp_outputs.misc['total_energies']['energy_extrapolated']
                        structure_set[adsorbate] = vasp_outputs.structure

                    dG = self.calculate_oer_overpotential(energy_set)[0]
                    adsorbate_collection[uuid_str].append(list([dG, energy_set, structure_set]))
                    self.report('set number {} in {} with OER dG: {}'.format(idx, uuid_str, dG))
                else:
                    raise NotImplementedError('Reaction not implemented')

            self.ctx.r2scan_adsorbates[self.ctx.reaction][uuid_str] = adsorbate_collection[uuid_str]

        if failed_jobs > 0:
            self.report('some jobs failed, check if needed')

    def store_results(self):
        """Store results"""
        assert 1 == 2
        # return

    def final_report(self):
        """Final report"""
        self.report(f"AdsorbatesWorkChain for {self.ctx.chemical_formula} finished successfully")

    @staticmethod
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

    @staticmethod
    def calculate_oer_overpotential(adsorption_energies):
        """Calculate overpotential for given reaction energy set"""
        local_energy = adsorption_energies.copy()
        local_energy['H2'] = -7.018265666883999     # includes zpe corrections for VASP data
        local_energy['H2O'] = -14.226717097410363   # includes zpe corrections for VASP data

        charges = list([0, 1, 2, 3, 4])
        oer_zpe = dict({'*': 0, '*O': 0.05,'*OH': 0.35, '*OOH': 0.4, 'H2': 0, 'H2O': 0})  # specific to OER pathway
        reaction_path = list([{},
                              {'*OH': +1, '*': -1, 'H2O': -1, 'H2': 1 / 2},
                              {'*O': +1, '*': -1, 'H2O': -1, 'H2': 1},
                              {'*OOH': +1, '*': -1, 'H2O': -2, 'H2': 3 / 2}])
        dga = np.array([])
        for r in reaction_path:
            dgi = 0
            for q, e in r.items():
                dgi += local_energy[q] * e + oer_zpe[q]
            dga = np.append(dga, dgi)
        dga = np.append(dga, 4.92)
        dg_rel_0_pot = dga[1:] - dga[:-1]
        overpotential = max(dg_rel_0_pot) - 1.23
        dga -= 1.23 * np.array(charges)  # assume equilibrium
        return overpotential, dga

    @staticmethod
    def _construct_adsorbate_builder(ml_structure, ML_model, reaction):
        """
        Builder for generating surface and surface optimization
        """
        structure = [ml_structure]
        Workflow = WorkflowFactory(ML_model.lower())
        builder = Workflow.get_builder()
        builder.input_structures = List(structure)
        builder.code = get_code(ML_model)
        _, model_path, device = get_model_device(ML_model)
        relax_key = "adsorbates"
        job_info = {
            "job_type": "adsorbates",
            "model_path": model_path,
            "device": device,
            "fmax": settings.inputs[relax_key]["fmax"],
            "max_steps": settings.inputs[relax_key]["max_steps"],
            "reaction": reaction,
        }
        builder.job_info = Dict(job_info)
        return builder
