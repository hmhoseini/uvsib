from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from aiida.orm import Bool, Str, List, Dict
from aiida.engine import WorkChain, if_
from aiida.plugins import WorkflowFactory
import aiida_pythonjob
from uvsib.db.tables import DBStructure, DBStructureVersion, DBChemsys, DBComposition
from uvsib.db.session import get_session
from uvsib.db.utils import (
        update_row,
        delete_row,
        query_by_columns,
        get_chemical_systems,
        query_structure)
from uvsib.workchains.utils import (
        unique_low_energy_comp,
        get_ref_entries,
        get_code,
        get_model_device)
from uvsib.workchains.pythonjob_inputs import is_data_available
from uvsib.workflows import settings

_SKIP_CSP = settings._SKIP_CSP
_SKIP_GEN = settings._SKIP_GEN
DFT_FUNC = settings.DFT_FUNC
EHULL_ML = settings.EHULL_ML
MAX_NUM_BULK = settings.MAX_NUM_BULK

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
    chemical_systems, _ = get_chemical_systems(chemical_formula)
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
            if_(cls.should_run_mpdb_ml)(
                cls.mpdb_ml,
                cls.inspect_mpdb_ml
            ),
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

        spec.exit_code(300, "ERROR_CALCULATION_FAILED", message="The WorkChain did not finish successfully")
        spec.exit_code(301, "ERROR_NO_STRUCTURES_FOUND", message="No experimentally observed structures were found")
        spec.exit_code(302, "ERROR_NO_CHEMSYS_FOUND", message="Chemical system does not exist in DBChemsys")

    def setup(self):
        """Setup and report"""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.chemical_systems = self.inputs.chemical_systems.get_list()
        self.ctx.ML_model = self.inputs.ML_model.value
        self.report(f"Running PhaseDiagramML WorkChain for {self.ctx.chemical_formula}")
        if self.ctx.chemical_systems:
            self.report(f"New chemical systems:{self.ctx.chemical_systems}")
        else:
            self.report(f"No new chemical system!")

    def should_run_mpdb_ml(self):
        """Check whether should run MPDBMLWorkChain"""
        results = query_structure({"composition": self.ctx.chemical_formula}, method = "DFT")
        existing_sources = {result.source for result in results}
        if "MPDB_stb" in existing_sources or "MPDB_exp" in existing_sources:
            self.report(f"Skipping MPDBMLWorkChain for {self.ctx.chemical_formula}")
            return False
        return True

    def should_run_csp(self):
        """Check whether should run CSPWorkChain"""
        if _SKIP_CSP:
            self.report(f"Skipping CSP calculations for {self.ctx.chemical_formula}")
            return False
        results = query_structure({"composition": self.ctx.chemical_formula}, source = "csp")
        if results:
            self.report(f"Nothing to do! Skipping CSP calculations for {self.ctx.chemical_formula}")
            return False
        return True

    def should_run_gen(self):
        """Check whether should run MatterGen"""
        if _SKIP_GEN:
            # DBChemsys status is updated to Ready
            for chemical_system in self.ctx.chemical_systems:
                r = query_by_columns(DBChemsys,{"chemsys": chemical_system})
                if r:
                    row = r[0]
                    update_row(DBChemsys, row.uuid,{"gen_structures": "Ready"})
                else:
                    self.report(f"ERROR: {chemical_system} does not exist in the DBChemsys database")
                    return self.exit_codes.ERROR_NO_CHEMSYS_FOUND
            self.report(f"Skipping Gen calculations for {self.ctx.chemical_formula}")
            return False
        if not self.ctx.chemical_systems:
            self.report(f"Nothing to do! Skipping MatterGen calculations for {self.ctx.chemical_formula}")
            return False
        return True

    def mpdb_ml(self):
        """Run MPDBML WorkChain"""
        builder = self._construct_mpdb_ml_builder()
        future = self.submit(builder)
        self.to_context(**{"mpdb_ml": future})

    def inspect_mpdb_ml(self):
        """Inspect MPDBML WorkChain"""
        if not self.ctx.mpdb_ml.is_finished_ok:
            self.report(f"MPDBMLWorkChain for {self.ctx.chemical_formula} failed.")
            return self.exit_codes.ERROR_CALCULATION_FAILED

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

            self.report(f"CSPWorkChain for {self.ctx.chemical_formula} failed. Corresponding rows will be removed from DBChemsys.")
            return self.exit_codes.ERROR_CALCULATION_FAILED

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
        if _SKIP_CSP or _SKIP_GEN:
            return
        all_chemical_systems, _ = get_chemical_systems(self.ctx.chemical_formula)
        inputs = aiida_pythonjob.prepare_pythonjob_inputs(is_data_available,
            function_inputs= {"chemical_systems": all_chemical_systems},
            computer="localhost",
            outputs_spec=aiida_pythonjob.spec.namespace(moveon=Bool),
        )
        future = self.submit(aiida_pythonjob.PythonJob, inputs=inputs)
        self.to_context(**{"pyjob": future})

    def check_pythonjob(self):
        """Inspect PythonJob"""
        if _SKIP_CSP or _SKIP_GEN:
            return
        calculation = self.ctx["pyjob"]
        if not calculation.is_finished_ok or not calculation.outputs.moveon.value:
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def store_stable_structs(self):
        """Return final structures"""
        chemical_formula = self.ctx.chemical_formula
        model = self.ctx.ML_model
        self.report(f"Constructing phase diagram for {chemical_formula}")
        entries = get_entries_from_db(chemical_formula, model)

        if not entries:
            self.report(f"Constructing phase diagram for {chemical_formula} failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        ref_entries, _ = get_ref_entries(self.ctx.chemical_formula, self.ctx.ML_model)

        uuid_list = []
        unique_entries, _ = unique_low_energy_comp(
            chemical_formula, entries, DFT_FUNC, EHULL_ML, min_n_return=1, element_entries=ref_entries)
        for entry in unique_entries:
            uuid_list.append(str(entry.data["struct_uuid"]))

        if not uuid_list:
            self.report(f"WARNING: no stable structure for {self.ctx.chemical_formula} has been found")

        # add uuids of stable structures to the DBComposition table
        row = query_by_columns(DBComposition,{"composition": self.ctx.chemical_formula})[0]

        update_row(DBComposition, row.uuid,{"stable_struct": {"ml_uuid_list": uuid_list}})

    def final_report(self):
        """Final report"""
        self.report("PhaseDiagramML WorkChain finished successfully")

    ################################################################################
    def _construct_mpdb_ml_builder(self):
        """Build MPDBML WorkChain builder"""
        MPDBMLWorkChain = WorkflowFactory("mpdbml")
        builder = MPDBMLWorkChain.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.ML_model = self.ctx.ML_model
        return builder

    def _construct_csp_builder(self):
        Workflow = WorkflowFactory("csp")
        builder = Workflow.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.n_csp = settings.inputs["MatterGen_CSP"]["num_runs"]
        builder.n_mh = settings.inputs["MinimaHopping"]["num_runs"]
        builder.ML_model = self.ctx.ML_model
        return builder

    def _construct_gen_builder(self):
        Workflow = WorkflowFactory("gen")
        builder = Workflow.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.chemical_systems = List(self.ctx.chemical_systems)
        builder.ML_model = self.ctx.ML_model
        return builder

    def _construct_ML_relax_builder(self, structures):
        """
        General builder for structure optimization with an ML model
        """
        ML_model = self.ctx.ML_model
        Workflow = WorkflowFactory(ML_model.lower())
        builder = Workflow.get_builder()
        builder.input_structures = List(structures)
        builder.code = get_code(ML_model)
        builder.local_label = Str(f"relax {self.ctx.chemical_formula}")
        model, model_path, device = get_model_device(ML_model)

        job_info = {
            "job_type": "relax",
            "ML_model": ML_model,
            "model_name": model,
            "model_path": model_path,
            "model_head": settings.inputs["bulk_relax"]["head"],
            "device": device,
            "fmax": settings.inputs["bulk_relax"]["fmax"],
            "max_steps": settings.inputs["bulk_relax"]["max_steps"]
        }

        builder.job_info = Dict(job_info)
        return builder
