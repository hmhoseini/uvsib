import os
import yaml
import numpy as np
from ase.io import jsonio
from aiida.engine import WorkChain
from aiida.plugins import WorkflowFactory
from aiida.orm import Str, List, Dict, load_code, StructureData
from uvsib.db.tables import DBSurface
from uvsib.db.utils import add_surface_adsorbate, add_surface_ml_adsorbate
from uvsib.codes.vasp.workchains import construct_vasp_builder
from uvsib.codes.utils import ase_to_pmg
from uvsib.db.utils import get_structure_uuid_surface_id, query_by_columns
from uvsib.workchains.utils import get_code, get_model_device
from uvsib.workflows import settings
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure


def read_yaml(file_path):
    """Read yaml file"""
    with open(file_path, "r", encoding="utf8") as fhandle:
        data = yaml.safe_load(fhandle)
    return data

class AdsorbatesWorkChain(WorkChain):
    """Adsorbates WorkChain"""
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("chemical_formula", valid_type=Str)
        spec.input("ML_model", valid_type=Str)
        spec.input("reaction", valid_type=Dict)

        spec.outline(
            cls.setup,
            cls.run_adsorbs,
            cls.inspect_adsorbs,
            cls.store_results_ml,
            # cls.scan_relax,
            # cls.inspect_relax,
            # cls.store_scan_results,
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
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.ML_model = self.inputs.ML_model.value
        self.ctx.reaction = self.inputs.reaction.value
        self.ctx.structure_surface_rows = get_structure_uuid_surface_id(self.ctx.chemical_formula)
        if not self.ctx.structure_surface_rows:
            return self.exit_codes.ERROR_NO_STRUCTURES_FOUND
        self.ctx.ml_results = dict()
        self.ctx.candidates = dict()
        self.ctx.relaxation_results = dict()
        self.ctx.adsorption_sets = dict()
        self.ctx.protocol = read_yaml(os.path.join(settings.vasp_files_path, "protocol.yaml"))
        self.ctx.potential_family = settings.configs["codes"]["VASP"]["potential_family"]
        potential_mapping = read_yaml(os.path.join(settings.vasp_files_path, "potential_mapping.yaml"))
        self.ctx.potential_mapping = potential_mapping["potential_mapping"]
        self.ctx.vasp_code = load_code(settings.configs["codes"]["VASP"]["code_string"])

        self.report("Running Adsorbates WorkChain for {}".format(self.ctx.chemical_formula))

    def run_adsorbs(self):
        """Run Adsorbates WorkChain"""
        for structure_uuid, surface_id in self.ctx.structure_surface_rows:
            slab_row = query_by_columns(DBSurface, {"id":surface_id})[0]
            uuid_str = str(structure_uuid)
            builder = self._construct_adsorbate_builder(slab_row.slab, self.ctx.ML_model, self.ctx.reaction)
            future = self.submit(builder)
            self.to_context(**{f"ads_{uuid_str}_{surface_id}": future})

    def inspect_adsorbs(self):
        """Inspect Adsorbates WorkChain"""
        for structure_uuid, surface_id in self.ctx.structure_surface_rows:
            uuid_str = str(structure_uuid)
            ads_wch = self.ctx[f"ads_{uuid_str}_{surface_id}"]
            if not ads_wch.is_finished_ok:
                continue
            output_dict = ads_wch.called[-1].outputs.output_dict
            self.ctx.ml_results[f"{uuid_str}_{surface_id}"] = output_dict["structures"]

    def store_results_ml(self):
        for parent_key, adsorption_sets in self.ctx.ml_results.items():
            uuid_str, surface_id = parent_key.split("_", 2)
            idx = None
            site = None
            energy_set = dict()
            for adsorb_set in adsorption_sets:
                for ads_json in adsorb_set:
                    adsorbed = jsonio.decode(ads_json)
                    energy_set[adsorbed.info['adsorbate']] = adsorbed.info['{}_energy'.format(str(self.ctx.ML_model).lower())]
                    site = adsorbed.info["site"]
                    idx = adsorbed.info["adsorbate_collection"]
                eta, dG = self.calculate_oer_overpotential(energy_set)
                if eta < 1:
                    self.ctx.candidates[parent_key] = adsorb_set
                add_surface_ml_adsorbate(existing_uuid=uuid_str, surf_id=surface_id, comp=self.ctx.chemical_formula,
                                         reac=self.ctx.reaction, s_m=site, u_idx=idx, e=eta, dg=dG, ad_set=adsorb_set)

    def scan_relax(self):
        """Run r2SCAN geometry optimization"""
        for parent_key, adsorption_set in self.ctx.candidates.items():
            for adsorb_json in adsorption_set:
                adsorb = jsonio.decode(adsorb_json)
                unique_idx = adsorb.info["adsorbate_collection"]
                site = adsorb.info["site"]
                ad = adsorb.info["adsorbate"]
                structure = ase_to_pmg(adsorb)
                struct = StructureData(pymatgen=structure)
                struct.base.attributes.set("site_properties", structure.site_properties)
                builder = construct_vasp_builder(struct, self.ctx.protocol["r2SCAN_adsorbates"],
                                                 self.ctx.potential_family, self.ctx.potential_mapping,
                                                 self.ctx.vasp_code)
                future = self.submit(builder)
                self.to_context(**{f"scan_relax_{parent_key}_{site}_{unique_idx}_{ad}": future})

    def inspect_relax(self):
        """Inspect r2SCAN geometry optimization"""
        failed_jobs = 0
        for parent_key, adsorption_set in self.ctx.candidates.items():
            for adsorb_json in adsorption_set:
                adsorbed = jsonio.decode(adsorb_json)
                unique_idx = adsorbed.info["adsorbate_collection"]
                site = adsorbed.info["site"]
                ad = adsorbed.info["adsorbate"]
                wch = self.ctx[f"scan_relax_{parent_key}_{site}_{unique_idx}_{ad}"]
                if not wch.is_finished_ok:
                    failed_jobs += 1
                    break
                outputs = wch.called[-1].outputs
                structure = outputs.structure.get_pymatgen()
                energy = outputs.misc["total_energies"]["energy_extrapolated"]
                self.ctx.relaxation_results[f"{parent_key}_{site}_{unique_idx}_{ad}"] = [structure, energy]
        if failed_jobs:
            self.report(f"{failed_jobs} r2SCAN relaxations failed")

    def store_scan_results(self):
        energy_sets = dict()
        for parent_key, entry in self.ctx.relaxation_results.items():
            uuid_str = parent_key.split("_")[0]
            surface_id = parent_key.split("_")[1]
            site = parent_key.split("_")[2]
            idx = parent_key.split("_")[3]
            adsorb = parent_key.split("_")[4]
            if f"{uuid_str}_{surface_id}_{site}_{idx}" not in self.ctx.adsorption_sets:
                energy_sets[f"{uuid_str}_{surface_id}_{site}_{idx}"] = dict()
                self.ctx.adsorption_sets[f"{uuid_str}_{surface_id}_{site}_{idx}"] = dict()

            energy_sets[f"{uuid_str}_{surface_id}_{site}_{idx}"].update({adsorb: entry[1]})
            self.ctx.adsorption_sets[f"{uuid_str}_{surface_id}_{site}_{idx}"].update(
                {adsorb: dict({'structure_dict': entry[0].as_dict(), 'dft_energy': entry[1]})})

        for key in energy_sets:
            if len(energy_sets[key]) != 4:
                print('set with key {} has missing computations, skipped'.format(key))
                continue

            uuid_str = key.split("_")[0]
            surface_id = key.split("_")[1]
            site = key.split("_")[2]
            idx = key.split("_")[3]

            eta, dG = self.calculate_oer_overpotential(energy_sets[key])
            add_surface_adsorbate(existing_uuid=uuid_str, surf_id=surface_id, comp=self.ctx.chemical_formula,
                                  reac=self.ctx.reaction, s_m=site, u_idx=idx, e=eta, dg=dG, ad_set=self.ctx.adsorption_sets[key])

    def final_report(self):
        """Final report"""
        if len(self.ctx.adsorption_sets) == 0:
            return
        else:
            self.report('AdsorbatesWorkChain for {} finished successfully, {} reaction sets computed (4 each for OER).'
                        .format(self.ctx.chemical_formula, len(self.ctx.adsorption_sets)))

    @staticmethod
    def calculate_oer_overpotential(adsorption_energies):
        """Calculate overpotential for given reaction energy set"""
        local_energy = adsorption_energies.copy()
        local_energy['H2'] = -7.02570471              # includes zpe corrections for VASP r2SCAN
        local_energy['H2O'] = -15.41801614            # includes zpe corrections for VASP r2SCAN

        charges = list([0, 1, 2, 3, 4])
        oer_zpe = dict({'*': 0,
                        '*O': 0.05,
                        '*OH': 0.35,
                        '*OOH': 0.4,
                        'H2': 0,
                        'H2O': 0})  # specific to OER pathway

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
        return overpotential, dga.tolist()

    @staticmethod
    def _construct_adsorbate_builder(slab, ML_model, reaction):
        """
        Builder for generating surface and surface optimiziation with MatterSim or MACE
        """
        structure = [slab]
        slab_energy = slab["energy"]
        Workflow = WorkflowFactory(ML_model.lower())
        builder = Workflow.get_builder()
        builder.input_structures = List(structure)
        builder.code = get_code(ML_model)
        model, model_path, device = get_model_device(ML_model)
        my_reaction = None
        my_path = None
        for entry in reaction:
            my_reaction = entry
            my_path = reaction[entry]
            break  # TODO implement multi pathways here?
        relax_key = "adsorbates"
        job_info = {
            "job_type": "adsorbates",
            "ML_model": ML_model,
            "device": device,
            "slab_energy": slab_energy,
            "fmax": settings.inputs[relax_key]["fmax"],
            "max_steps": settings.inputs[relax_key]["max_steps"],
            "reaction": my_reaction,
            "pathway": my_path
        }
        if ML_model in ["uPET"]:
            job_info.update({"model_name": model})
        else:
            job_info.update({"model_path": model_path})

        builder.job_info = Dict(job_info)

        return builder
