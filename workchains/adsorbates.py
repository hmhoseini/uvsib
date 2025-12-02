from aiida.engine import WorkChain
from aiida.plugins import WorkflowFactory
from aiida.orm import Str, List, Dict
from uvsib.db.tables import DBSurface
from uvsib.db.utils import get_structure_uuid_surface_id, query_by_columns
from uvsib.workchains.utils import get_code, get_model_device
from uvsib.workflows import settings

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
        self.report("Running Adsorbates WorkChain")
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.ML_model = self.inputs.ML_model.value
        self.ctx.reaction = self.inputs.reaction.value
        self.ctx.structure_surface_rows = get_structure_uuid_surface_id(self.ctx.chemical_formula)
        if not self.ctx.structure_surface_rows:
            return self.exit_codes.ERROR_NO_STRUCTURES_FOUND
        self.ctx.ads_slab = []

    def run_adsorbs(self):
        """Run Adsorbates WorkChain"""
        for row in self.ctx.structure_surface_rows:
            structure_uuid = row[0]
            surface_id = row[1]
            slab_row = query_by_columns(DBSurface, {"id":surface_id})[0]
            uuid_str = str(structure_uuid)
            builder = self._construct_adsorbate_builder(slab_row.structure,
                                                        self.ctx.ML_model,
                                                        self.ctx.reaction)
            future = self.submit(builder)
            self.to_context(**{f"ads_{uuid_str}_{surface_id}": future})

    def inspect_adsorbs(self):
        """Inspect Adsorbates WorkChain"""
        failed_jobs = 0
        for row in self.ctx.structure_surface_rows:
            structure_uuid = row[0]
            surface_id = row[1]
            uuid_str = str(structure_uuid)
            ads_wch = self.ctx[f"ads_{uuid_str}_{surface_id}"]
            if not ads_wch.is_finished_ok:
                failed_jobs += 1
                continue
            output_dict = ads_wch.called[-1].outputs.output_dict
            self.ctx.ads_slab.append([output_dict["structures"], uuid_str])

    def store_results(self):
        """Store results"""
        return

    def final_report(self):
        """Final report"""
        self.report(f"AdsorbatesWorkChain for {self.ctx.chemical_formula} finished successfully")

    @staticmethod
    def _construct_adsorbate_builder(ml_structure, ML_model, reaction):
        """
        Builder for generating surface and surface optimiziation with MatterSim or MACE
        """
        structure = [ml_structure]

        Workflow = WorkflowFactory(ML_model.lower()) # "MatterSim" -> "mattersim", "MACE" -> "mace"

        builder = Workflow.get_builder()

        builder.input_structures = List(structure)
        builder.code = get_code(ML_model)

        model_path, device = get_model_device(ML_model)

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
