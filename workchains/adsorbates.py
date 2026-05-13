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
from uvsib.workchains.oer import calculate_oer_overpotential
from uvsib.workchains.co2rr import calculate_co2rr_overpotential
from uvsib.workchains.noxrr import calculate_noxrr_overpotential
from uvsib.workflows import settings


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
        spec.input("reaction", valid_type=Str)
        spec.input("reaction_path", valid_type=Str)

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
        self.ctx.reaction_path = self.inputs.reaction_path.value
        self.ctx.structure_surface_rows = get_structure_uuid_surface_id(self.ctx.chemical_formula)
        if not self.ctx.structure_surface_rows:
            return self.exit_codes.ERROR_NO_STRUCTURES_FOUND
        self.ctx.ml_results = {}
        self.ctx.candidates = {}
        self.ctx.relaxation_results = {}
        self.ctx.adsorption_sets = {}
        self.ctx.protocol = read_yaml(os.path.join(settings.vasp_files_path, "protocol.yaml"))
        self.ctx.potential_family = settings.configs["codes"]["VASP"]["potential_family"]
        potential_mapping = read_yaml(os.path.join(settings.vasp_files_path, "potential_mapping.yaml"))
        self.ctx.potential_mapping = potential_mapping["potential_mapping"]
        self.ctx.vasp_code = load_code(settings.configs["codes"]["VASP"]["code_string"])

        self.report(f"Running Adsorbates WorkChain for {self.ctx.chemical_formula}. Reaction: {self.ctx.reaction}, reaction_path: {self.ctx.reaction_path}")

    def run_adsorbs(self):
        """Run Adsorbates WorkChain"""
        for index, (structure_uuid, surface_id) in enumerate(self.ctx.structure_surface_rows):
            slab_row = query_by_columns(DBSurface, {"id":surface_id})[0]
            uuid_str = str(structure_uuid)
            builder = self._construct_adsorbate_builder(slab_row.slab, self.ctx.ML_model, self.ctx.reaction, self.ctx.reaction_path)
            future = self.submit(builder)
            self.to_context(**{f"ads_{uuid_str}_{surface_id}": future})
            if index > 10:
                return

    def inspect_adsorbs(self):
        """Inspect Adsorbates WorkChain"""
        failed = []

        for structure_uuid, surface_id in self.ctx.structure_surface_rows:
            uuid_str = str(structure_uuid)
            ads_wch = self.ctx[f"ads_{uuid_str}_{surface_id}"]

            if not ads_wch.is_finished_ok:
                failed.append(f"{uuid_str}_{surface_id} (exit status: {ads_wch.exit_status})")
                continue

            output_dict = ads_wch.outputs.output_dict
            self.ctx.ml_results[f"{uuid_str}_{surface_id}"] = output_dict["structures"]

        if failed:
            self.report(f"WARNING: {len(failed)} sub-workflow(s) failed.")

        if len(failed) == len(self.ctx.structure_surface_rows):
            self.report("ERROR: all sub-workflows failed - no results to process.")
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def store_results_ml(self):
        """Store ML results """
        reaction_map = {
            "OER": (calculate_oer_overpotential, 2.0),
            "CO2RR": (calculate_co2rr_overpotential, 1.5),
            "NOXRR": (calculate_noxrr_overpotential, 1.5),
        }

        if self.ctx.reaction not in reaction_map:
            self.report(f"The reaction {self.ctx.reaction} is not known")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        self.report(f"Storing {self.ctx.reaction} results.")

        calc_method, eta_threshold = reaction_map[self.ctx.reaction]

        slab_cache = {}
        for parent_key, adsorption_sets in self.ctx.ml_results.items():
            uuid_str, surface_id = parent_key.split("_")
            if surface_id not in slab_cache:
                slab_cache[surface_id] = query_by_columns(DBSurface, {"id": surface_id})[0]
            slab_row = slab_cache[surface_id]
            miller_index = slab_row.slab["miller_index"]
            energy_set = {}
            for adsorb_set in adsorption_sets:
                energy_set = {}
                site_type = adsorb_set["site_type"]
                ads_coord = adsorb_set["ads_coord"]
                repeat = adsorb_set["repeat"]
                for ads_json in adsorb_set["structures"]:
                    adsorbed = jsonio.decode(ads_json)
                    energy_set[adsorbed.info["adsorbate"]] = adsorbed.info['{}_energy'.format(str(self.ctx.ML_model).lower())]

                eta, dG_steps, dG_cumulative = calc_method(
                    energy_set, self.ctx.reaction_path, self.ctx.ML_model, settings.ML_FUNC
                )

                if eta > eta_threshold:
                    continue

                self.ctx.candidates[parent_key] = adsorb_set
                add_surface_ml_adsorbate(existing_uuid=uuid_str, surf_id=surface_id, surface_miller_index= miller_index,
                                         comp=self.ctx.chemical_formula,
                                         react=self.ctx.reaction, react_path=self.ctx.reaction_path,
                                         site_type=site_type, ads_coord=ads_coord, repeat=repeat,
                                         e=eta, dG_steps=dG_steps, dG_cumulative=dG_cumulative,
                                         ad_set=adsorb_set)

#   def scan_relax(self):
#       """Run r2SCAN geometry optimization"""
#       for parent_key, adsorption_set in self.ctx.candidates.items():
#           for adsorb_json in adsorption_set:
#               adsorb = jsonio.decode(adsorb_json)
#               unique_idx = adsorb.info["adsorbate_collection"]
#               site = adsorb.info["site"]
#               ad = adsorb.info["adsorbate"]
#               structure = ase_to_pmg(adsorb)
#               struct = StructureData(pymatgen=structure)
#               struct.base.attributes.set("site_properties", structure.site_properties)
#               builder = construct_vasp_builder(struct, self.ctx.protocol["r2SCAN_adsorbates"],
#                                                self.ctx.potential_family, self.ctx.potential_mapping,
#                                                self.ctx.vasp_code)
#               future = self.submit(builder)
#               self.to_context(**{f"scan_relax_{parent_key}_{site}_{unique_idx}_{ad}": future})

#   def inspect_relax(self):
#       """Inspect r2SCAN geometry optimization"""
#       failed_jobs = 0
#       for parent_key, adsorption_set in self.ctx.candidates.items():
#           for adsorb_json in adsorption_set:
#               adsorbed = jsonio.decode(adsorb_json)
#               unique_idx = adsorbed.info["adsorbate_collection"]
#               site = adsorbed.info["site"]
#               ad = adsorbed.info["adsorbate"]
#               wch = self.ctx[f"scan_relax_{parent_key}_{site}_{unique_idx}_{ad}"]
#               if not wch.is_finished_ok:
#                   failed_jobs += 1
#                   break
#               outputs = wch.called[-1].outputs
#               structure = outputs.structure.get_pymatgen()
#               energy = outputs.misc["total_energies"]["energy_extrapolated"]
#               self.ctx.relaxation_results[f"{parent_key}_{site}_{unique_idx}_{ad}"] = [structure, energy]
#       if failed_jobs:
#           self.report(f"{failed_jobs} r2SCAN relaxations failed")

#   def store_scan_results(self):
#       energy_sets = dict()
#       for parent_key, entry in self.ctx.relaxation_results.items():
#           uuid_str = parent_key.split("_")[0]
#           surface_id = parent_key.split("_")[1]
#           site = parent_key.split("_")[2]
#           idx = parent_key.split("_")[3]
#           adsorb = parent_key.split("_")[4]
#           if f"{uuid_str}_{surface_id}_{site}_{idx}" not in self.ctx.adsorption_sets:
#               energy_sets[f"{uuid_str}_{surface_id}_{site}_{idx}"] = dict()
#               self.ctx.adsorption_sets[f"{uuid_str}_{surface_id}_{site}_{idx}"] = dict()

#           energy_sets[f"{uuid_str}_{surface_id}_{site}_{idx}"].update({adsorb: entry[1]})
#           self.ctx.adsorption_sets[f"{uuid_str}_{surface_id}_{site}_{idx}"].update(
#               {adsorb: dict({'structure_dict': entry[0].as_dict(), 'dft_energy': entry[1]})})

#           #TODO: Check if we have expected number of intermediates (varies by reaction)
#           uuid_str = key.split("_")[0]
#           surface_id = key.split("_")[1]
#           site = key.split("_")[2]
#           idx = key.split("_")[3]

#           # Calculate overpotential based on reaction type
#           if self.ctx.reaction == "OER":
#               eta, dG = self.calculate_oer_overpotential(energy_sets[key])
#           elif self.ctx.reaction == "CO2RR":
#               eta, dG = self.calculate_co2rr_overpotential(energy_sets[key], self.ctx.reaction_path)
#           elif self.ctx.reaction == "NOXRR":
#               eta, dG = self.calculate_noxrr_overpotential(energy_sets[key], self.ctx.reaction_path)

#           add_surface_adsorbate(existing_uuid=uuid_str, surf_id=surface_id, comp=self.ctx.chemical_formula,
#                                 react=self.ctx.reaction, react_path=self.ctx.reaction_path, site_type=site, ads_coord=idx, e=eta, dg=dG, ad_set=self.ctx.adsorption_sets[key])
    def final_report(self):
        """Final report"""
        if not self.ctx.candidates:
            self.report(f"AdsorbatesWorkChain for {self.ctx.chemical_formula}: no candidates below eta threshold.")
            return self.exit_codes.ERROE_CALCULATION_FAILED
        self.report(f"AdsorbatesWorkChain for {self.ctx.chemical_formula} finished successfully.")

    def _construct_adsorbate_builder(self, slab, ML_model, reaction, pathway):
        """
        Builder for generating surface and surface optimiziation with MatterSim or MACE
        """
        structure = [slab]
        slab_energy = slab["energy"]
        Workflow = WorkflowFactory(ML_model.lower())
        builder = Workflow.get_builder()
        builder.input_structures = List(structure)
        if self.ctx.reaction == "OER":
            builder.local_label = Str("Adsorbates: OER on {}".format(self.ctx.chemical_formula))
        else:
            builder.local_label = Str("Adsorbates: {} on {}".format(self.ctx.reaction_path, self.ctx.chemical_formula))
        builder.code = get_code(ML_model)

        model, model_path, device = get_model_device(ML_model)
        relax_key = "adsorbates"
        job_info = {
            "job_type": "adsorbates",
            "ML_model": ML_model,
            "device": device,
            "slab_energy": slab_energy,
            "fmax": settings.inputs[relax_key]["fmax"],
            "max_steps": settings.inputs[relax_key]["max_steps"],
            "reaction": reaction,
            "pathway": pathway
        }
        if ML_model in ["uPET"]:
            job_info.update({"model_name": model})
        else:
            job_info.update({"model_path": model_path})

        builder.job_info = Dict(job_info)

        return builder
