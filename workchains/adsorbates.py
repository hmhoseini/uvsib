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
        for structure_uuid, surface_id in self.ctx.structure_surface_rows:
            slab_row = query_by_columns(DBSurface, {"id":surface_id})[0]
            uuid_str = str(structure_uuid)
            builder = self._construct_adsorbate_builder(slab_row.slab, self.ctx.ML_model, self.ctx.reaction, self.ctx.reaction_path)
            future = self.submit(builder)
            self.to_context(**{f"ads_{uuid_str}_{surface_id}": future})

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

    @staticmethod
    def _construct_adsorbate_builder(slab, ML_model, reaction, pathway):
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
            "pathway": pathway
        }
        if ML_model in ["uPET"]:
            job_info.update({"model_name": model})
        else:
            job_info.update({"model_path": model_path})

        builder.job_info = Dict(job_info)

        return builder

#def calculate_noxrr_overpotential(adsorption_energies, pathway_name, method, func):
#   """Calculate overpotential for NOXRR given reaction energy set
#
#   Parameters
#   ----------
#   adsorption_energies : dict
#       Dictionary of adsorbate energies keyed by species name (e.g., "*NO", "*NH3")
#   pathway_name : str
#       Name of the NOXRR pathway. Supported: "no_dissociative", "no_to_nh3_noh",
#       "no_to_nh3_nhoh", "no_to_n2o", "no2_to_no", "no3_to_nh3", "no3_to_n2"
#
#   Returns
#   -------
#   tuple
#       (overpotential, dga_list) where overpotential is float and dga_list is list of
#       Gibbs free energies at 0V RHE for each step
#   """
#   local_energy = adsorption_energies.copy()
#   local_energy['H2'] = -7.02570471              # includes zpe corrections for VASP r2SCAN
#   local_energy['H2O'] = -15.41801614            # includes zpe corrections for VASP r2SCAN
#   local_energy['NO'] = -9.90                    # approximate NO gas phase energy
#   local_energy['NO2'] = -12.5                   # approximate NO2 gas phase energy
#   local_energy['NO3'] = -18.0                   # approximate NO3 gas phase energy
#   local_energy['N2O'] = -9.25                   # approximate N2O gas phase energy
#   local_energy['N2'] = -16.5                    # approximate N2 gas phase energy
#   local_energy['NH3'] = -19.5                   # approximate NH3 gas phase energy
#   local_energy['O2'] = -12.0                    # approximate O2 gas phase energy
#   local_energy['O'] = local_energy['O2'] / 2    #

#   # ZPE corrections specific to NOXRR intermediates (in eV)
#   noxrr_zpe = {
#       '*': 0.0,
#       '*NO': 0.10,
#       '*NO2': 0.20,
#       '*NO3': 0.30,
#       '*NOH': 0.20,
#       '*HNO': 0.20,
#       '*N2O2': 0.30,
#       '*N2O': 0.25,
#       '*N': 0.0,
#       '*NH': 0.10,
#       '*NH2': 0.20,
#       '*NH3': 0.30,
#       '*NHOH': 0.35,
#       '*NH2OH': 0.40,
#       '*O': 0.05,
#       '*OH': 0.35,
#       '*H2O': 0.55,
#       'H2': 0.0,
#       'H2O': 0.0,
#       'NO': 0.0,
#       'NO2': 0.0,
#       'NO3': 0.0,
#       'N2O': 0.0,
#       'N2': 0.0,
#       'NH3': 0.0,
#       'O': 0.0,
#       'O2': 0.0
#   }

#   # Define reaction pathways
#   pathways = {
#       "no_dissociative": {
#           "equilibrium_potential": -0.10,
#           "steps": [
#               {},
#               {'*N': +1, '*': -1, 'NO': -1, 'O': +1},         # NO → N + O
#               {'*N2O2': +1, '*N': -2},                      # 2N → N2O2 coupling
#               {'*': +1, 'N2': -1, '*N2O2': -1}                # N2O2 → N2 (desorbed)
#           ]
#       },
#       "no_to_nh3_noh": {
#           "equilibrium_potential": 0.27,
#           "steps": [
#               {},
#               {'*NOH': +1, '*': -1, 'NO': -1, 'H2': 1/2},     # NO + H+ + e- → NOH
#               {'*N': +1, '*NOH': -1, 'H2O': -1, 'H2': 1/2},   # NOH → N + H2O
#               {'*NH': +1, '*N': -1, 'H2': 1/2},               # N + H+ + e- → NH
#               {'*NH2': +1, '*NH': -1, 'H2': 1/2},             # NH + H+ + e- → NH2
#               {'*NH3': +1, '*NH2': -1, 'H2': 1/2},            # NH2 + H+ + e- → NH3
#               {'*': +1, 'NH3': -1, '*NH3': -1}                # NH3 desorbed
#           ]
#       },
#       "no_to_nh3_nhoh": {
#           "equilibrium_potential": 0.27,
#           "steps": [
#               {},
#               {'*NOH': +1, '*': -1, 'NO': -1, 'H2': 1/2},
#               {'*NHOH': +1, '*NOH': -1, 'H2': 1/2},           # NOH + H+ + e- → NHOH
#               {'*NH2': +1, '*NHOH': -1, 'H2O': -1, 'H2': 1/2},# NHOH → NH2 + H2O
#               {'*NH3': +1, '*NH2': -1, 'H2': 1/2},
#               {'*': +1, 'NH3': -1, '*NH3': -1}
#           ]
#       },
#       "no_to_n2o": {
#           "equilibrium_potential": -0.03,
#           "steps": [
#               {},
#               {'*N2O2': +1, '*NO': -2},                     # 2NO → N2O2 coupling
#               {'*N2O': +1, '*': -1, '*N2O2': -1, 'O': +1},   # N2O2 → N2O + O
#               {'*': +1, 'N2O': -1, '*N2O': -1}                # N2O desorbed
#           ]
#       },
#       "no2_to_no": {
#           "equilibrium_potential": 0.80,
#           "steps": [
#               {},
#               {'*NO': +1, '*': -1, 'NO2': -1, 'O': +1},       # NO2 → NO + O
#               {'*OH': +1, '*O': -1, 'H2': 1/2},               # O + H+ + e- → OH
#               {'*': +1, 'H2O': -1, '*OH': -1, 'H2': 1/2}      # OH + H+ + e- → H2O
#           ]
#       },
#       "no3_to_nh3": {
#           "equilibrium_potential": 0.88,
#           "steps": [
#               {},
#               {'*NO2': +1, '*': -1, 'NO3': -1, 'H2O': -1, 'H2': 1/2},  # NO3 + H+ + e- → NO2
#               {'*NO': +1, '*NO2': -1, 'H2O': -1, 'H2': 1/2},          # NO2 + H+ + e- → NO
#               {'*NOH': +1, '*NO': -1, 'H2': 1/2},
#               {'*N': +1, '*NOH': -1, 'H2O': -1, 'H2': 1/2},
#               {'*NH': +1, '*N': -1, 'H2': 1/2},
#               {'*NH2': +1, '*NH': -1, 'H2': 1/2},
#               {'*NH3': +1, '*NH2': -1, 'H2': 1/2},
#               {'*': +1, 'NH3': -1, '*NH3': -1}
#           ]
#       },
#       "no3_to_n2": {
#           "equilibrium_potential": 1.40,
#           "steps": [
#               {},
#               {'*NO2': +1, '*': -1, 'NO3': -1, 'H2O': -1, 'H2': 1/2},
#               {'*NO': +1, '*NO2': -1, 'H2O': -1, 'H2': 1/2},
#               {'*N': +1, '*NO': -1, 'O': +1},                # NO → N + O
#               {'*N2O2': +1, '*N': -2},
#               {'*': +1, 'N2': -1, '*N2O2': -1}
#           ]
#       }
#   }

#   if pathway_name not in pathways:
#       raise ValueError(f"Unsupported NOXRR pathway: {pathway_name}. "
#                      f"Supported pathways: {list(pathways.keys())}")

#   pathway = pathways[pathway_name]
#   reaction_path = pathway["steps"]
#   equilibrium_potential = pathway["equilibrium_potential"]
#   charges = list(range(len(reaction_path)))

#   dga = np.array([])
#   for r in reaction_path:
#       dgi = 0
#       for q, e in r.items():
#           dgi += local_energy[q] * e + noxrr_zpe[q] * e
#       dga = np.append(dga, dgi)

#   # Final gas-phase reference state
#   dga = np.append(dga, 0.0)

#   dg_rel_0_pot = dga[1:] - dga[:-1]
#   overpotential = max(dg_rel_0_pot) - equilibrium_potential
#   dga -= equilibrium_potential * np.array(charges)  # assume equilibrium
#   return overpotential, [], dga.tolist()

#ddef calculate_co2rr_overpotential(adsorption_energies, pathway_name, method, func):
#   """Calculate overpotential for CO2RR given reaction energy set

#   Parameters
#   ----------
#   adsorption_energies : dict
#       Dictionary of adsorbate energies keyed by species name (e.g., "*CO2_ads", "*COOH")
#   pathway_name : str
#       Name of the CO2RR pathway. Supported: "co2_to_co", "co2_to_hcooh", "co_to_ch4",
#       "co_to_ch3oh", "co2_to_c2h4"

#   Returns
#   -------
#   tuple
#       (overpotential, dga_list) where overpotential is float and dga_list is list of
#       Gibbs free energies at 0V RHE for each step
#   """
#   local_energy = adsorption_energies.copy()
#   local_energy['H2'] = -7.02570471              # includes zpe corrections for VASP r2SCAN
#   local_energy['H2O'] = -15.41801614            # includes zpe corrections for VASP r2SCAN
#   local_energy['CO2'] = -22.5                   # approximate CO2 gas phase energy

#   # ZPE corrections specific to CO2RR intermediates (in eV)
#   co2rr_zpe = {
#       '*': 0.0,
#       '*CO2_ads': 0.30,
#       '*COOH': 0.35,
#       '*OCHO': 0.25,
#       '*CO': 0.15,
#       '*CHO': 0.20,
#       '*CHOH': 0.40,
#       '*CH2O': 0.35,
#       '*CH2OH': 0.45,
#       '*CH': 0.10,
#       '*CH2': 0.20,
#       '*CH3': 0.30,
#       '*OCCO': 0.40,
#       '*CCHO': 0.40,
#       '*C2H4_ads': 0.50,
#       'H2': 0.0,
#       'H2O': 0.0,
#       'CO2': 0.0,
#       'CO': 0.0,
#       'CH4': 0.0,
#       'CH3OH': 0.0,
#       'C2H4': 0.0,
#       'HCOOH': 0.0
#   }

#   # Define reaction pathways
#   pathways = {
#       "co2_to_co": {
#           "equilibrium_potential": 0.11,  # V vs RHE
#           "steps": [
#               {},  # initial state (clean surface)
#               {'*COOH': +1, '*': -1, 'CO2': -1, 'H2': 1/2},  # CO2 + H+ + e- → COOH
#               {'*CO': +1, '*': -1, 'H2O': -1, 'H2': 1}       # COOH + H+ + e- → CO + H2O
#           ]
#       },
#       "co2_to_hcooh": {
#           "equilibrium_potential": -0.05,
#           "steps": [
#               {},
#               {'*OCHO': +1, '*': -1, 'CO2': -1, 'H2': 1/2},  # CO2 + H+ + e- → OCHO
#               {'*': +1, 'HCOOH': -1, '*OCHO': -1, 'H2': 1/2}  # OCHO + H+ + e- → HCOOH (desorbed)
#           ]
#       },
#       "co_to_ch4": {
#           "equilibrium_potential": 0.24,
#           "steps": [
#               {},
#               {'*CHO': +1, '*CO': -1, 'H2': 1/2},            # CO + H+ + e- → CHO
#               {'*CHOH': +1, '*CHO': -1, 'H2': 1/2},          # CHO + H+ + e- → CHOH
#               {'*CH': +1, '*CHOH': -1, 'H2O': -1, 'H2': 1},  # CHOH → CH + H2O
#               {'*CH2': +1, '*CH': -1, 'H2': 1/2},            # CH + H+ + e- → CH2
#               {'*CH3': +1, '*CH2': -1, 'H2': 1/2},           # CH2 + H+ + e- → CH3
#               {'*': +1, 'CH4': -1, '*CH3': -1, 'H2': 1/2}    # CH3 + H+ + e- → CH4 (desorbed)
#           ]
#       },
#       "co_to_ch3oh": {
#           "equilibrium_potential": 0.38,
#           "steps": [
#               {},
#               {'*CHO': +1, '*CO': -1, 'H2': 1/2},
#               {'*CHOH': +1, '*CHO': -1, 'H2': 1/2},
#               {'*CH2OH': +1, '*CHOH': -1, 'H2': 1/2},
#               {'*': +1, 'CH3OH': -1, '*CH2OH': -1, 'H2': 1/2}
#           ]
#       },
#       "co2_to_c2h4": {
#           "equilibrium_potential": 0.34,
#           "steps": [
#               {},
#               {'*COOH': +1, '*': -1, 'CO2': -1, 'H2': 1/2},
#               {'*CO': +1, '*': -1, 'H2O': -1, 'H2': 1},
#               {'*OCCO': +1, '*CO': -2},                    # CO + CO → OCCO coupling
#               {'*CCHO': +1, '*OCCO': -1, 'H2': 1/2},
#               {'*': +1, 'C2H4': -1, '*CCHO': -1, 'H2': 1/2}
#           ]
#       }
#   }

#   if pathway_name not in pathways:
#       raise ValueError(f"Unsupported CO2RR pathway: {pathway_name}. "
#                      f"Supported pathways: {list(pathways.keys())}")

#   pathway = pathways[pathway_name]
#   reaction_path = pathway["steps"]
#   equilibrium_potential = pathway["equilibrium_potential"]
#   charges = list(range(len(reaction_path)))

#   dga = np.array([])
#   for r in reaction_path:
#       dgi = 0
#       for q, e in r.items():
#           dgi += local_energy[q] * e + co2rr_zpe[q] * e
#       dga = np.append(dga, dgi)

#   # Final gas-phase reference state
#   dga = np.append(dga, 0.0)

#   dg_rel_0_pot = dga[1:] - dga[:-1]
#   overpotential = max(dg_rel_0_pot) - equilibrium_potential
#   dga -= equilibrium_potential * np.array(charges)  # assume equilibrium
#   return overpotential, [], dga.tolist()


#ef calculate_oer_overpotential(adsorption_energies, pathway_name, method, func):
#   """
#   4e_mechanism pathway
#   Calculate OER thermodynamics using the CHE model.
#   Returns
#           overpotential,
#           dG_steps: [ΔG1, ΔG2, ΔG3, ΔG4] at U=0 V
#           dG_cumulative, [0, G1, G2, G3, 4.92]
#   """
#   local_energy = adsorption_energies.copy()
#   if method == "dft":
#       if func == "PBE":
#           # DFT references PBE (already include ZPE)
#           raise NotImplementedError("PBE reference energies for OER are not defined")

#       elif func == "r2SCAN":
#           # DFT references r2SCAN (already include ZPE)
#           local_energy['H2'] = -7.02570471
#           local_energy['H2O'] = -15.41801614
#       else:
#           raise NotImplementedError("Reference energies for OER are not defined")
#   else:
#       if func == "r2SCAN":
#           # uPET pet-omatpes-l reference plus ZPE
#           local_energy['H2'] = -6.891794204711914 + 0.27
#           local_energy['H2O'] = -15.237101554870605 + 0.56

#   # ZPE + entropy corrections
#   DELTA_OH = 0.35
#   DELTA_O = 0.05
#   DELTA_OOH = 0.40

#   G_OER_TOTAL = 4.92

#   half_H2 = 0.5 * local_energy['H2']
#   E_H2O = local_energy['H2O']

#   E_star = local_energy['*']
#   E_OH = local_energy['*OH']
#   E_O = local_energy['*O']
#   E_OOH = local_energy['*OOH']

#   # Adsorption energies
#   A = E_OH - E_star
#   B = E_O - E_star
#   C = E_OOH - E_star

#   # Elementary OER steps
#   dG1 = A - E_H2O + half_H2 + DELTA_OH
#   dG2 = (B - A) + half_H2 + (DELTA_O - DELTA_OH) # descriptor
#   dG3 = (C - B) - E_H2O + half_H2 + (DELTA_OOH - DELTA_O)
#   dG4 = G_OER_TOTAL - dG1 - dG2 - dG3
#   dG_steps = np.array([dG1, dG2, dG3, dG4])

#   # OER overpotential
#   overpotential = dG_steps.max() - 1.23

#   # Potential determining step
#   #pds = int(np.argmax(dG_steps) + 1)

#   # Cumulative free energies
#   dG_cumulative = [
#       0.0,
#       dG1,
#       dG1 + dG2,
#       dG1 + dG2 + dG3,
#       G_OER_TOTAL
#   ]
#   return overpotential, dG_steps.tolist(), dG_cumulative
