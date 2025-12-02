from aiida.orm import Int, Str, List, Dict
from aiida.plugins import WorkflowFactory
from aiida.engine import WorkChain
from uvsib.db.utils import update_row, add_structures, query_by_columns
from uvsib.db.tables import DBChemsys
from uvsib.workchains.utils import (get_output_as_entry,
                              unique_low_energy_chemsys,
                              get_code,
                              get_model_device
                             )
from uvsib.workflows import settings

class GenWorkChain(WorkChain):
    """Work chain for generating structures"""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("chemical_systems", valid_type=List)
        spec.input("ML_model", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.generative_calcs,
            cls.inspect_gen_calcs,
            cls.predict_ml_energies,
            cls.store_ml_energies,
            cls.final_report
        )

        spec.exit_code(301, "ERROR_GENERATIVE_FAILED", message="MatterGen generative calculation failed")
        spec.exit_code(302, "ERROR_ML_RELAX_FAILED", message="ML relaxation failed")

    def setup(self):
        """Setup and report"""
        self.ctx.chemical_systems = self.inputs.chemical_systems.get_list()
        self.ctx.ML_model = self.inputs.ML_model.value
        self.ctx.failed_ml_e = []
        self.report(f"Launching MatterGenWorkChain for {self.ctx.chemical_systems}")

    def generative_calcs(self):
        """Run MatterGen workchains for each unique chemical system"""
        for chemical_system in self.ctx.chemical_systems:
            builder = self._construct_mattergen_gen_builder(chemical_system)
            future = self.submit(builder)
            self.to_context(**{f"{chemical_system}_mattergen": future})

    def inspect_gen_calcs(self):
        """Check for any failures among MatterGen calculations"""
        for chemical_system in self.ctx.chemical_systems:
            calculation = self.ctx[f"{chemical_system}_mattergen"]
            if not calculation.is_finished_ok:
                return self.exit_codes.ERROR_GENERATIVE_FAILED

    def predict_ml_energies(self):
        """Predict the energies of the structures with the given ML model"""
        chemical_systems = self.ctx.chemical_systems
        for chemical_system in chemical_systems:

            wch = self.ctx[f"{chemical_system}_mattergen"]
            output_structures = (
                    wch.called[-1]
                    .outputs.output_dict["structures"]
            )
            builder = self._construct_ML_relax_builder(output_structures, self.ctx.ML_model)
            future = self.submit(builder)
            self.to_context(**{f"{chemical_system}_ml_e": future})

    def store_ml_energies(self):
        """Store predicted ML energies"""
        failed_ml_e = []
        chemical_systems = self.ctx.chemical_systems
        for chemical_system in chemical_systems:

            wch = self.ctx[f"{chemical_system}_ml_e"]

            if not wch.is_finished_ok:
                failed_ml_e.append(chemical_system)
                self.report(f"Warning: {self.ctx.ML_model} for {chemical_system} failed")
                continue
            try:
                new_entries = get_output_as_entry(wch)

            except:
                self.report(
                    f"Warning: Failed to store results for {chemical_system}"
                )
                failed_ml_e.append(chemical_system)
                continue

            low_energy_entries = unique_low_energy_chemsys(
                    chemical_system,
                    new_entries,
                    "GGA"
            )
            structure_energy_pairs = []

            for entry in low_energy_entries:
                structure_energy_pairs.append((entry.structure.as_dict(), entry.energy))

            add_structures(
                    "generated",
                    self.ctx.ML_model,
                    structure_energy_pairs
            )
            # DBChemsys status is updated to Ready
            row = query_by_columns(DBChemsys,
                             {"chemsys": chemical_system}
                            )[0]
            update_row(DBChemsys,
                       row.uuid,
                      {"gen_structures": "Ready"}
            )

        if failed_ml_e:
            return self.exit_codes.ERROR_ML_RELAX_FAILED

    def final_report(self):
        self.report(f"MatterGenWorkChain for {self.ctx.chemical_systems} finished successfully")
################################################################################
    @staticmethod
    def _construct_mattergen_gen_builder(chemical_system):
        """MatterGen gen Builder"""
        Workflow = WorkflowFactory("mattergen.base")
        builder = Workflow.get_builder()
        builder.chemical_system = Str(chemical_system)
        builder.code = get_code("MatterGen")

        builder.job_info = Dict(
                {"job_type": "gen",
                 "model_name":  settings.configs["codes"]["MatterGen"]["model_name"],
                 "energy_above_hull": settings.inputs["MatterGen_generate"]["energy_above_hull"],
                 "batch_size": settings.inputs["MatterGen_generate"]["batch_size"],
                 "num_batches": settings.inputs["MatterGen_generate"]["num_batches"]
                }
        )
        builder.max_iterations = Int(2)
        return builder

    @staticmethod
    def _construct_ML_relax_builder(structures, ML_model):
        """
        General builder for MatterSim or MACE for structure opimization
        """
        Workflow = WorkflowFactory(ML_model.lower()) # "MatterSim" -> "mattersim", "MACE" -> "mace"

        builder = Workflow.get_builder()

        builder.input_structures = List(structures)
        builder.code = get_code(ML_model)

        model_path, device = get_model_device(ML_model)

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
