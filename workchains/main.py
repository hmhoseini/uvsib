from aiida.orm import Str, List, Dict
from aiida.engine import WorkChain
from aiida.plugins import WorkflowFactory
from db.query import query_by_columns
from db.tables import DBComposition
from db.utils import update_row

class MainWorkChain(WorkChain):
    """ Main WorkChain"""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)
        spec.input("chemical_systems", valid_type=List)
        spec.input("uuid", valid_type=Str)
        spec.outline(
            cls.setup,
            cls.run_phase_diagram_ml,
            cls.inspect_phase_diagram_ml,
            cls.dft_verification,
        )
        spec.exit_code(
            300,
            "ERROR_CALCULATION_FAILED",
            message="A sub-workchain did not finish successfully"
        )

    def setup(self):
        """Setup and report"""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.chemical_systems = self.inputs.chemical_systems.value
        self.report(f"Running MainWorkChain for {self.ctx.chemical_formula}")
        self.ctx.skip_pd_ml = False
        row = query_by_columns(DBComposition,
                               {"composition": self.ctx.chemical_formula}
        )[0]
        if row.step_status["phase_diagram_ML"] == "Done":
            self.ctx.skip_pd_ml = True

    def run_phase_diagram_ml(self):
        """Run PhaseDiagramML WorkChain"""
        if self.ctx.skip_pd_ml:
            return

        builder = self._construct_phase_diagram_ml_builder(
                self.ctx.chemical_formula,
                self.ctx.chemical_systems
        )
        future = self.submit(builder)
        self.to_context(**{"phase_diagram_ml": future})

    def inspect_phase_diagram_ml(self):
        """Inspect phase diagram workchain"""
        if self.ctx.skip_pd_ml:
            return

        chemical_formula = self.ctx.chemical_formula
        ml_workchain = self.ctx.phase_diagram_ml
        row = query_by_columns(DBComposition,
                               {"composition": chemical_formula}
        )[0]
        if not ml_workchain.is_finished_ok:
            self.report("PhaseDiagramML Workchain failed")
            # update row status in DBComposition table
            update_row(
                    DBComposition,
                    row.uuid,
                    {"status": "Failed",
                     "step_status": {"phase_diagram_ML": "Failed"}
                    }
            )
            return self.exit_codes.ERROR_CALCULATION_FAILED

        self.ctx.output_uuids = ml_workchain.outputs.output_uuids

        # update row status in DBComposition table
        update_row(
                DBComposition,
                row.uuid,
                {"status": "Running",
                 "step_status": {'phase_diagram_ML': 'Done'},
                }
        )
        self.report("PhaseDiagramML Workchain finished successfully")

    def dft_verification(self):
        """DFT Verification of  ML resutls"""
        builder = self._construct_dft_verification_builder(self.ctx.output_uuids)

    @staticmethod
    def _construct_phase_diagram_ml_builder(chemical_formula, chemical_systems):
        PhaseDiagramMLWorkChain = WorkflowFactory("phasediagramml")
        builder = PhaseDiagramMLWorkChain.get_builder()
        builder.chemical_formula = Str(chemical_formula)
        builder.chemical_systems = List(list=chemical_systems)
        builder.job_info = Dict(dict={"ML_model": "MatterSim",
                                     }
        )
        return builder

    @staticmethod
    def _construct_dft_verification_builder():
        pass

