import random
from aiida.engine import WorkChain
from aiida.orm import Str, List, Dict, Int
from aiida.plugins import DataFactory, WorkflowFactory
from uvsib.workchains.utils import (
        unique_low_energy_comp,
        get_output_as_entry,
        get_code,
        get_model_device)
from uvsib.db.utils import add_structures
from uvsib.workflows import settings

StructureData = DataFactory('core.structure')

class CSPWorkChain(WorkChain):
    """WorkChain for Crystal Structure Prediction (CSP)"""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("chemical_formula", valid_type=Str)
        spec.input("ML_model", valid_type=Str)
        spec.input("n_csp", valid_type=Int)
        spec.input("n_mh", valid_type=Int)

        spec.outline(
            cls.setup,
            cls.run_csp,
            cls.inspect_csp_calcs,
            cls.predict_ml_energies,
            cls.collect_ml_energies,
            cls.minimahopping,
            cls.mh_energies,
            cls.final_step,
            cls.final_report
        )

        spec.exit_code(301, "ERROR_CSP_FAILED", message="CSP calculations failed")
        spec.exit_code(302, "ERROR_ML_RELAX_FAILED", message="ML relaxation failed")
        spec.exit_code(303, "ERROR_MINIMAHOPPING_FAILED", message="MinimaHopping calculations failed")

    def setup(self):
        """Setup and report"""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.ML_model = self.inputs.ML_model.value
        self.ctx.n_csp = self.inputs.n_csp.value
        self.ctx.n_mh = self.inputs.n_mh.value
        self.ctx.csp_structures = []
        self.report(f"Launching CSPWorkChain for {self.ctx.chemical_formula}")

    def run_csp(self):
        """Run MatterGen CSP"""
        for i in range(1, self.ctx.n_csp + 1):
            builder = self._construct_mattergen_csp_builder()
            future = self.submit(builder)
            self.to_context(**{f"csp_{i}": future})

    def inspect_csp_calcs(self):
        """Check MatterGen CSP calculations"""
        failed_jobs = 0
        for i in range(1, self.ctx.n_csp + 1):
            csp_wch = self.ctx[f"csp_{i}"]
            if not csp_wch.is_finished_ok:
                failed_jobs += 1
                continue

            try:
                self.ctx.csp_structures.extend(
                    csp_wch.called[-1].outputs.output_dict["structures"]
                )
            except:
                failed_jobs += 1

        if not self.ctx.csp_structures:
            return self.exit_codes.ERROR_CSP_FAILED

        if failed_jobs / self.ctx.n_csp > 0.5:
            return self.exit_codes.ERROR_CSP_FAILED

    def predict_ml_energies(self):
        """Predict the energies of the structures with the given ML model"""
        builder = self._construct_ML_relax_builder()
        future = self.submit(builder)
        self.to_context(**{"ml_e": future})

    def collect_ml_energies(self):
        """ML energies"""
        wch = self.ctx.ml_e

        if not wch.is_finished_ok:
            return self.exit_codes.ERROR_ML_RELAX_FAILED
        try:
            new_entries = get_output_as_entry(wch)
        except:
            return self.exit_codes.ERROR_ML_RELAX_FAILED

        self.ctx.low_energy_entries_csp = unique_low_energy_comp(
                self.ctx.chemical_formula,
                new_entries,
                "GGA"
        )

    def minimahopping(self):
        """Run MinimaHopping"""
        entries_csp = self.ctx.low_energy_entries_csp
        n_mh = min(len(entries_csp), self.ctx.n_mh)
        selected_entries = random.sample(entries_csp, n_mh)
        for i, entry in enumerate(selected_entries):
            struct = StructureData(pymatgen_structure = entry.structure)
            builder = self._construct_mh_builder(struct, self.ctx.ML_model)
            future = self.submit(builder)
            self.to_context(**{f"mh_{i}": future})

    def mh_energies(self):
        """Predict ML energies"""
        n_mh = min(len(self.ctx.low_energy_entries_csp), self.ctx.n_mh)
        all_entries = []
        failed_jobs = 0
        for i in range(n_mh):
            wch = self.ctx[f"mh_{i}"]

            if not wch.is_finished_ok:
                failed_jobs += 1
                continue
            try:
                new_entries = get_output_as_entry(wch)
                all_entries.extend(new_entries)
            except:
                failed_jobs += 1

        if not all_entries or failed_jobs / n_mh > 0.5:
            return self.exit_codes.ERROR_MINIMAHOPPING_FAILED

        self.ctx.low_energy_entries_mh = unique_low_energy_comp(
                self.ctx.chemical_formula,
                new_entries,
                "GGA"
        )

    def final_step(self):
        """Store structures"""
        all_entries = self.ctx.low_energy_entries_csp + self.ctx.low_energy_entries_mh

        low_energy_entries = unique_low_energy_comp(
                self.ctx.chemical_formula,
                all_entries,
                "GGA"
        )
        structure_energy_pairs = []

        for entry in low_energy_entries:
            structure_energy_pairs.append((entry.structure.as_dict(), entry.energy))

        add_structures(
                "csp",
                self.ctx.ML_model,
                structure_energy_pairs
        )

    def final_report(self):
        """Final report"""
        self.report(f"CSPWorkChain for {self.ctx.chemical_formula} finished successfully")

####################################################################
    def _construct_mattergen_csp_builder(self):
        Workflow = WorkflowFactory("mattergen.csp")
        builder = Workflow.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.code = get_code("MatterGen")
        _, model_path, _ = get_model_device("MatterGenCSP")
        builder.job_info = Dict(
            {
                "job_type": "csp",
                "model_path": model_path,
                "batch_size": settings.inputs["MatterGen_CSP"]["batch_size"],
                "num_batches": settings.inputs["MatterGen_CSP"]["num_batches"],
            }
        )
        builder.max_iterations = Int(2)
        return builder

    def _construct_ML_relax_builder(self):
        """
        General builder for MatterSim or MACE for structure opimization
        """
        ML_model = self.ctx.ML_model
        structures = self.ctx.csp_structures

        Workflow = WorkflowFactory(ML_model.lower()) # "MatterSim" -> "mattersim", "MACE" -> "mace"

        builder = Workflow.get_builder()

        builder.input_structures = List(structures)
        builder.code = get_code(ML_model)

        _, model_path, device = get_model_device(ML_model)

        relax_key = "bulk_relax"
        job_info = {
            "job_type": "relax",
            "model_path": model_path,
            "device": device,
            "fmax": settings.inputs[relax_key]["fmax"],
            "max_steps": settings.inputs[relax_key]["max_steps"],
        }

        builder.job_info = Dict(job_info)
        return builder

    @staticmethod
    def _construct_mh_builder(struct, ML_model):
        Workflow = WorkflowFactory("minimahopping")
        builder = Workflow.get_builder()
        builder.structure = struct
        builder.code = get_code("MinimaHopping")

        _, model_path, device = get_model_device(ML_model)
        builder.job_info = Dict(
            {
             "model": ML_model,
             "model_path": model_path,
             "device": device,
             "mh_steps": settings.inputs["MinimaHopping"]["mh_steps"],
             "fmax": settings.inputs["MinimaHopping"]["fmax"],
            }
        )
        return builder
