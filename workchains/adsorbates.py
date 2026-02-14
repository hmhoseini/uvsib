import os
import yaml
import numpy as np
from collections import defaultdict
from ase.io import jsonio
from aiida.engine import WorkChain
from aiida.plugins import WorkflowFactory
from aiida.orm import Str, List, Dict, load_code, StructureData
from uvsib.db.tables import DBSurface
from uvsib.db.utils import add_surface_adsorbate
from uvsib.codes.vasp.workchains import construct_vasp_builder
from uvsib.codes.utils import ase_to_pmg
from uvsib.db.utils import get_structure_uuid_surface_id, query_by_columns
from uvsib.workchains.utils import get_code, get_model_device
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

        spec.outline(
            cls.setup,
            cls.run_adsorbs,
            cls.inspect_adsorbs,
            cls.store_results_ml,
#            cls.run_scan,
#            cls.inspect_scan,
#            cls.store_results,
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
        self.report("Running Adsorbates WorkChain")
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.ML_model = self.inputs.ML_model.value
        self.ctx.reaction = self.inputs.reaction.value
        self.ctx.structure_surface_rows = get_structure_uuid_surface_id(self.ctx.chemical_formula)
        if not self.ctx.structure_surface_rows:
            return self.exit_codes.ERROR_NO_STRUCTURES_FOUND
        self.ctx.ml_results = {}
        self.ctx.scan_results = defaultdict(list)
        self.ctx.protocol = read_yaml(
                os.path.join(settings.vasp_files_path, "protocol.yaml")
        )
        self.ctx.potential_family = settings.configs["codes"]["VASP"]["potential_family"]
        potential_mapping = read_yaml(os.path.join(settings.vasp_files_path, "potential_mapping.yaml"))
        self.ctx.potential_mapping = potential_mapping["potential_mapping"]
        self.ctx.vasp_code = load_code(
                settings.configs["codes"]["VASP"]["code_string"]
        )

    def run_adsorbs(self):
        """Run Adsorbates WorkChain"""
        for structure_uuid, surface_id in self.ctx.structure_surface_rows:
            slab_row = query_by_columns(DBSurface, {"id":surface_id})[0]
            uuid_str = str(structure_uuid)
            builder = self._construct_adsorbate_builder(slab_row.slab,
                                                        self.ctx.ML_model,
                                                        self.ctx.reaction)
            future = self.submit(builder)
            self.to_context(**{f"ads_{uuid_str}_{surface_id}": future})

    def inspect_adsorbs(self):
        """Inspect Adsorbates WorkChain"""
        for structure_uuid, surface_id in self.ctx.structure_surface_rows:
            uuid_str = str(structure_uuid)
            key = f"ads_{uuid_str}_{surface_id}"
            ads_wch = self.ctx[key]
            if not ads_wch.is_finished_ok:
                continue
            output_dict = ads_wch.called[-1].outputs.output_dict
            self.ctx.ml_results[f"{uuid_str}_{surface_id}"] = output_dict["structures"]

    def store_results_ml(self):
        for parent_key, adsorption_sets in self.ctx.ml_results.items():
            uuid_str, surface_id = parent_key.split("_", 2)
            energy_set = {}
            for adsorb_set in adsorption_sets:
                for ads_json in adsorb_set:
                    adsorbed = jsonio.decode(ads_json)
                    energy_set[adsorbed.info["adsorbate"]] = adsorbed.info["energy"]
                    site = adsorbed.info["site"]
                    idx = adsorbed.info["adsorbate_collection"]

                eta, dG = self.calculate_oer_overpotential(energy_set)

                add_surface_adsorbate(
                    uuid_str,
                    surface_id,
                    self.ctx.chemical_formula,
                    self.ctx.reaction,
                    site,
                    idx,
                    eta,
                    dG,
                    adsorb_set,
                )

    def run_scan(self):
        """Run r2SCAN geometry optimization"""
        for parent_key, adsorption_sets in self.ctx.ml_results.items():
            run_clean_surface = True
            for adsorb_set in adsorption_sets:
                for ads_json in adsorb_set:
                    adsorbed = jsonio.decode(ads_json)
                    if adsorbed.info["adsorbate"] == "*":
                        if not run_clean_surface:
                            continue
                        run_clean_surface = False
                    unique_idx = adsorbed.info["adsorbate_collection"]
                    site = adsorbed.info["site"]
                    ad = adsorbed.info["adsorbate"]
                    pmg_structure = ase_to_pmg(adsorbed)
                    struct = StructureData(pymatgen=pmg_structure)
                    struct.base.attributes.set("site_properties",
                                                pmg_structure.site_properties
                    )
                    builder = construct_vasp_builder(
                        struct,
                        self.ctx.protocol["r2SCAN_slab"],
                        self.ctx.potential_family,
                        self.ctx.potential_mapping,
                        self.ctx.vasp_code
                    )
                    future = self.submit(builder)
                    self.to_context(**{f"scan_{parent_key}_{site}_{unique_idx}_{ad}": future})

    def inspect_scan(self):
        """Inspect r2SCAN geometry optimization"""
        failed_jobs = 0
        for parent_key, adsorption_sets in self.ctx.ml_results.items():
            for adsorb_set in adsorption_sets:
                ad_set = []
                for ads_json in adsorb_set:
                    adsorbed = jsonio.decode(ads_json)
                    unique_idx = adsorbed.info["adsorbate_collection"]
                    site = adsorbed.info["site"]
                    ad = adsorbed.info["adsorbate"]
                    if adsorbed.info["adsorbate"] == "*":
                        ad_set.append([{}, "*", adsorbed.info["clean_slab_energy"]]) #TODO bare surface energy is calculated only by ML
                        continue
                    scan_wch = self.ctx[f"scan_{parent_key}_{site}_{unique_idx}_{ad}"]
                    if not scan_wch.is_finished_ok:
                        failed_jobs += 1
                        break
                    outputs = scan_wch.outputs
                    structure = outputs.structure.get_pymatgen()
                    energy = outputs.misc["total_energies"]["energy_extrapolated"]
                    ad_set.append([structure.as_dict(), ad, energy])
                self.ctx.scan_results[f"{parent_key}_{site}_{unique_idx}"].append(ad_set)

        if failed_jobs:
            self.report(f"{failed_jobs} r2SCAN jobs failed")

    def store_results(self):
        for key, values in self.ctx.scan_results.items():
            energy_set = {}
            uuid_str, surface_id, site, idx = key.split("_", 3)
            for ad_set in values:
                for v in ad_set:
                    energy_set[v[1]] = v[2]
            eta, dG = self.calculate_oer_overpotential(energy_set)

            add_surface_adsorbate(
                uuid_str,
                surface_id,
                self.ctx.chemical_formula,
                self.ctx.reaction,
                site,
                idx,
                eta,
                dG,
                values,
            )

    def final_report(self):
        """Final report"""
        self.report(f"AdsorbatesWorkChain for {self.ctx.chemical_formula} finished successfully")

    @staticmethod
    def calculate_oer_overpotential(adsorption_energies):
        """Calculate overpotential for given reaction energy set"""
        local_energy = adsorption_energies.copy()
        local_energy['H2'] = -7.018265666883999     # includes zpe corrections for VASP data
        local_energy['H2O'] = -14.226717097410363   # includes zpe corrections for VASP data

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

        relax_key = "adsorbates"

        job_info = {
            "job_type": "adsorbates",
            "ML_model": ML_model,
            "device": device,
            "slab_energy": slab_energy,
            "fmax": settings.inputs[relax_key]["fmax"],
            "max_steps": settings.inputs[relax_key]["max_steps"],
            "reaction": reaction,
        }
        if ML_model in ["uPET"]:
            job_info.update({"model_name": model})
        else:
            job_info.update({"model_path": model_path})

        builder.job_info = Dict(job_info)

        return builder
