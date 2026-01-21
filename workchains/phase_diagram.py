from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from aiida.orm import Str, List, Bool
from aiida.engine import WorkChain, if_
from aiida.plugins import WorkflowFactory
from aiida_pythonjob import PythonJob, prepare_pythonjob_inputs, spec
from uvsib.db.tables import DBStructure, DBStructureVersion, DBChemsys, DBComposition
from uvsib.db.session import get_session
from uvsib.db.utils import (
        update_row,
        delete_row,
        query_by_columns,
        get_chemical_systems,
        query_structure)
from uvsib.workchains.utils import unique_low_energy_comp
from uvsib.workchains.pythonjob_inputs import is_data_available
from uvsib.workflows import settings

_EHULL = 0.05

def cleanup_failed_systems(chemical_systems):
    """Remove database entries for failed calculations"""
    for chemsys in chemical_systems:
        result = query_by_columns(DBChemsys, {"chemsys": chemsys})
        if result:
            delete_row(DBChemsys, result[0])

def get_entries_from_db(chemical_formula, method):
    """Retrieve ComputedStructureEntry objects for
    all relevant chemical systems from the database"""
    entries = []
    chemical_systems = get_chemical_systems(chemical_formula, new=False)
    with get_session() as session:
        try:
            results = (
                session.query(DBStructureVersion)
                .join(DBStructure)
                .filter(DBStructure.chemsys.in_(chemical_systems))
                .filter(DBStructureVersion.method == method)
                .all()
            )
        except:
            return None

    for row in results:
        struct = Structure.from_dict(row.structure)
        entries.append(
                ComputedStructureEntry(
                composition=struct.composition,
                structure=struct,
                energy=row.energy,
                data={"struct_uuid": row.structure_uuid})
        )
    return entries

class PhaseDiagramMLWorkChain(WorkChain):
    """Work chain for ML Phase Diagram calculations"""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("chemical_formula", valid_type=Str)
        spec.input("chemical_systems", valid_type=List)
        spec.input("ML_model", valid_type=Str)

        spec.outline(
            cls.setup,
            if_(cls.should_run_csp)(
                cls.csp_calcs,
                cls.inspect_csp_cals,
            ),
            if_(cls.should_run_gen)(
                cls.gen_calcs,
                cls.inspect_gen_calcs,
            ),
            cls.wait_for_data,
            cls.check_pythonjob,
            cls.store_stable_structs,
            cls.final_report
        )

        spec.exit_code(300,
            "ERROR_CALCULATION_FAILED",
            message="The WorkChain did not finish successfully"
        )

    def setup(self):
        """Setup and report"""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.chemical_systems = self.inputs.chemical_systems.get_list()
        self.ctx.ML_model = self.inputs.ML_model.value
        self.ctx.failed_ml_e = []
        self.report(f"Running PhaseDiagramML WorkChain for {self.ctx.chemical_formula}")

    def should_run_csp(self):
        """Check whether should run CSPWorkChain"""
        results = query_structure({"composition": self.ctx.chemical_formula}, source = "csp")
        if results:
            return False
        return True

    def csp_calcs(self):
        """Run CSP MatterGen"""
        builder = self._construct_csp_builder()
        future = self.submit(builder)
        self.to_context(**{"csp": future})

    def inspect_csp_cals(self):
        """Inspect CSPWorkChain"""
        failed_chemsys = []
        if not self.ctx.csp.is_finished_ok:
           # remove corresponding row from DBChemsys
            for chemsys in self.ctx.chemical_systems:
                results = query_by_columns(DBChemsys, {'chemsys': chemsys})
                if results:
                    row = results[0]
                    if not row.gen_structures:
                        failed_chemsys.append(chemsys)
            cleanup_failed_systems(failed_chemsys)

            self.report(f"CSPWorkChain for {self.ctx.chemical_formula} failed. Corresponding rows will be removed from DBChemsys")
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def should_run_gen(self):
        """Check whether should run MatterGen"""
        if self.ctx.chemical_systems:
            return True
        return False

    def gen_calcs(self):
        """Run MatterGen"""
        builder = self._construct_gen_builder()
        future = self.submit(builder)
        self.to_context(**{"gen": future})

    def inspect_gen_calcs(self):
        """Inspect MatterGenWorkChain"""
        failed_chemsys = []
        if not self.ctx.gen.is_finished_ok:
            # remove corresponding row from DBChemsys
            for chemsys in self.ctx.chemical_systems:
                results = query_by_columns(DBChemsys, {'chemsys': chemsys})
                if results:
                    row = results[0]
                    if not row.gen_structures:
                        failed_chemsys.append(chemsys)
            cleanup_failed_systems(failed_chemsys)

            self.report(f"MatterGen (gen) for {failed_chemsys} failed. Corresponding rows will be removed from DBChemsys")
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def wait_for_data(self):
        """Wait until all chemical systems are available"""
        all_chemical_systems = get_chemical_systems(self.ctx.chemical_formula, new=False)
        inputs = prepare_pythonjob_inputs(is_data_available,
            function_inputs= {"chemical_systems": all_chemical_systems},
            computer="localhost",
            outputs_spec=spec.namespace(moveon=Bool),
        )
        future = self.submit(PythonJob, inputs=inputs)
        self.to_context(**{"pyjob": future})

    def check_pythonjob(self):
        """Inspect PythonJob"""
        calculation = self.ctx["pyjob"]
        if not calculation.is_finished_ok or not calculation.outputs.moveon.value:
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def store_stable_structs(self):
        """Return final structures"""
        chemical_formula = self.ctx.chemical_formula
        self.report("Constructing phase diagram")
        entries = get_entries_from_db(chemical_formula, self.ctx.ML_model)
        if not entries:
            self.report(f"Constructing phase diagram for {chemical_formula} failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        uuid_list = []
        for entry in unique_low_energy_comp(chemical_formula, entries, "GGA"):
            uuid_list.append(str(entry.data["struct_uuid"]))

        if not uuid_list:
            self.report(f"WARNING: no stable structure for {self.ctx.chemical_formula} has been found")

        # add uuids of stable structures to the DBComposition table
        row = query_by_columns(DBComposition,
                               {"composition": self.ctx.chemical_formula}
        )[0]

        update_row(
                DBComposition,
                row.uuid,
                {"stable_struct": {"ml_uuid_list": uuid_list},
                }
        )

    def final_report(self):
        """Final report"""
        self.report("PhaseDiagramML WorkChain finished successfully")

    ################################################################################
    def _construct_csp_builder(self):
        Workflow = WorkflowFactory("csp")
        builder = Workflow.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.ML_model = Str(self.ctx.ML_model)
        builder.n_csp = settings.inputs["MatterGen_CSP"]["num_runs"]
        builder.n_mh = settings.inputs["MinimaHopping"]["num_runs"]
        return builder

    def _construct_gen_builder(self):
        Workflow = WorkflowFactory("gen")
        builder = Workflow.get_builder()
        builder.chemical_systems = List(self.ctx.chemical_systems)
        builder.ML_model = Str(self.ctx.ML_model)
        return builder
