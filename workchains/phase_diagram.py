from pymatgen.core.structure import Structure, Composition
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
#        add_version_to_existing_structure,
        query_structure)
from uvsib.workchains.utils import unique_low_energy_comp, element_reference_entries
from uvsib.workchains.exp_include import dedup_forced
from uvsib.workchains.pythonjob_inputs import is_data_available
from uvsib.workflows import settings


DFT_FUNC = settings.DFT_FUNC
EHULL_ML = settings.EHULL_ML
#MAX_NUM_BULK = settings.MAX_NUM_BULK

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
#            cls.reformat_entries,
            cls.final_report
        )

        spec.exit_code(300,"ERROR_CALCULATION_FAILED", message="The WorkChain did not finish successfully")

    def setup(self):
        """Setup and report"""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.chemical_systems = self.inputs.chemical_systems.get_list()
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
        self.report(f"Constructing phase diagram for {chemical_formula}")
        entries = get_entries_from_db(chemical_formula, settings.inputs['bulk_relax']['model'])

        if not entries:
            self.report(f"Constructing phase diagram for {chemical_formula} failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        uuid_list = []
        model = settings.inputs['bulk_relax']['model']
        el_entries, _ = element_reference_entries(
            Composition(chemical_formula).chemical_system.split('-'), model)
        unique_entries, _ = unique_low_energy_comp(
            chemical_formula, entries, DFT_FUNC, EHULL_ML, min_n_return=1, element_entries=el_entries)
        for entry in unique_entries:
            uuid_list.append(str(entry.data["struct_uuid"]))
#            self.local_list.append([entry.data["struct_uuid"],
#                                    "{}".format(self.ctx.ML_model),
#                                    {"structure": entry.structure.as_dict(), "energy": entry.energy}])

        # Force-include the stored MP-experimental structures (anti-lottery):
        # an experimentally-known polymorph the ML hull window would drop must
        # still reach the downstream stages. Structurally deduplicated against
        # the ML selection (and each other) with the SAME matcher tolerances,
        # so the manifest never carries the same host twice; PREPENDED because
        # capped consumers (battery max_hosts) must see the trusted hosts
        # first. Their ML e_above_hull is recorded so an MLIP-vs-experiment
        # disagreement stays visible instead of silently vanishing.
        forced_uuids = []
        if settings.MP_EXP_FORCE:
            exp_rows = query_structure({"composition": chemical_formula},
                                       method=model, source="mp_experimental")
            candidates = [(str(r.structure_uuid), r.structure)
                          for r in sorted(exp_rows, key=lambda r: r.energy or 0.0)
                          if str(r.structure_uuid) not in uuid_list]
            kept = dedup_forced([e.structure for e in unique_entries], candidates)
            if kept:
                from pymatgen.analysis.phase_diagram import PhaseDiagram
                try:
                    pd = PhaseDiagram(entries + el_entries)
                except Exception:
                    pd = None
                energy_by_uuid = {str(r.structure_uuid): r.energy for r in exp_rows}
                for uuid, struct in kept:
                    ehull = None
                    if pd is not None:
                        try:
                            ehull = float(pd.get_e_above_hull(ComputedStructureEntry(
                                composition=struct.composition, structure=struct,
                                energy=energy_by_uuid[uuid])))
                        except Exception:
                            pass
                    forced_uuids.append(uuid)
                    self.report(f"force-including MP-experimental structure "
                                f"{uuid} (ML e_above_hull "
                                f"{'unknown' if ehull is None else f'{ehull:.3f} eV/atom'}"
                                f"{'' if ehull is None or ehull <= EHULL_ML else ' -- OUTSIDE the ML window, MLIP disagrees with experiment'})")
                uuid_list = forced_uuids + uuid_list

        if not uuid_list:
            self.report(f"WARNING: no stable structure for {self.ctx.chemical_formula} has been found")

        # add uuids of stable structures to the DBComposition table
        row = query_by_columns(DBComposition,{"composition": self.ctx.chemical_formula})[0]

        update_row(DBComposition, row.uuid,{"stable_struct": {"ml_uuid_list": uuid_list,
                                                              "forced_experimental": forced_uuids}})

#    def reformat_entries(self):
#        for uuid, model, str_en_pair in self.local_list[:MAX_NUM_BULK]:
#            add_version_to_existing_structure(uuid, model,{"structure": str_en_pair['structure'], "energy": str_en_pair['energy']})

    def final_report(self):
        """Final report"""
        self.report("PhaseDiagramML WorkChain finished successfully")

    ################################################################################
    def _construct_csp_builder(self):
        Workflow = WorkflowFactory("csp")
        builder = Workflow.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.n_csp = settings.inputs["MatterGen_CSP"]["num_runs"]
        builder.n_mh = settings.inputs["MinimaHopping"]["num_runs"]
        return builder

    def _construct_gen_builder(self):
        Workflow = WorkflowFactory("gen")
        builder = Workflow.get_builder()
        builder.chemical_systems = List(self.ctx.chemical_systems)
        return builder
