import os
from pymatgen.core import Composition, Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from aiida.orm import Int, Str, List, Dict, Code
from aiida.engine import WorkChain
from aiida.plugins import WorkflowFactory
from aiida_pythonjob import PythonJob
from db.tables import DBChemsys, DBStructure, DBStructureVersion
from db.query import query_by_columns
from db.session import get_session
from db.utils import update_row, delete_row, get_chemical_systems, add_structures
from codes.utils import select_charge_neutral, get_structures_from_mpdb
from workchains.pythonjob_inputs import get_pythonjob_input
from workflows import settings

def construct_phase_diagram(chemical_systems, method):
    entries = []
    with get_session() as session:
        for chemical_system in chemical_systems:
            try:
                results = (
                        session.query(DBStructureVersion)
                       .join(DBStructure)
                       .filter(DBStructure.chemsys == chemical_system)
                       .filter(DBStructureVersion.method == method)
                       .all()
                )
            except:
                return False

            for indx, row in enumerate(results):
                entries.append(PDEntry(
                    composition=Structure.from_dict(row.structure).composition,
                    energy=row.energy,
                    name = f"{chemical_system}-{indx}",
                    attribute=row.structure_uuid)
                )
    return PhaseDiagram(entries)

class PhaseDiagramMLWorkChain(WorkChain):
    """Work chain for ML Phase Diagram calculations"""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("chemical_formula", valid_type=Str)
        spec.input("chemical_systems", valid_type=List)
        spec.input("job_info", valid_type=Dict)

        spec.output("output_uuids", valid_type=List)

        spec.outline(
            cls.setup,
            cls.run_mattergen_gen_calculations,
            cls.handle_failed_gen_calculations,
            cls.run_mattergen_csp_calculations,
            cls.handle_failed_csp_calculations,
            cls.predict_ml_energies,
            cls.store_ml_energies,
            cls.handle_failed_ml_calculations,
            cls.wait_for_ml_data,
            cls.check_pythonjob,
            cls.construct_phase_diagram,
            cls.return_stable_structures
        )
        spec.exit_code(300,
            "ERROR_CALCULATION_FAILED",
            message="The calculation did not finish successfully"
        )

    def setup(self):
        """Setup and report"""
        job_info = self.inputs.job_info.get_dict()
        self.ctx.ML_model = job_info["ML_model"]
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.failed_gen = []
        self.ctx.failed_ml_e = []
        self.ctx.chemical_systems = self.inputs.chemical_systems.get_list()
        self.ctx.all_chemical_systems = get_chemical_systems(self.ctx.chemical_formula, new=False)
        self.report("Running PhaseDiagramML Workchain")
        self.ctx.n = 0
        if not self.ctx.chemical_systems:
            self.report(f"No new chemical systems found for {self.ctx.chemical_formula}")

    def run_mattergen_gen_calculations(self):
        """Launching MatterGen workchains for each unique chemical system"""
        for chemical_system in self.ctx.chemical_systems:
            self.report(f"Running MatterGen (gen) for {chemical_system}")
            builder = self._construct_mattergen_gen_builder(chemical_system)
            future = self.submit(builder)
            self.to_context(**{f"{chemical_system}_mattergen": future})

    def handle_failed_gen_calculations(self):
        """Checking for any failures among MatterGen calculations"""
        for chemical_system in self.ctx.chemical_systems:
            calculation = self.ctx[f"{chemical_system}_mattergen"]
            if not calculation.is_finished_ok:
                self.ctx.failed_gen.append(chemical_system)
        if self.ctx.failed_gen:
            self.report(f"MatterGen (gen) for {self.ctx.failed_gen} failed")
            # delete rows from DBChemsys if a calculation failed
            for chemsys in self.ctx.chemical_systems:
                self._cleanup_failed_system(chemsys)
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def run_mattergen_csp_calculations(self):
        """Launching MatterGen workchains for CSP"""
        chemical_formula = self.ctx.chemical_formula
        self.report(f"Running MatterGen (CSP) for {chemical_formula}")
        for i in range(1, settings.inputs["MatterGen_CSP"]["num_runs"]+1):
            builder = self._construct_mattergen_csp_builder(chemical_formula)
            future = self.submit(builder)
            self.to_context(**{f"{chemical_formula}_csp-{str(i)}": future})

    def handle_failed_csp_calculations(self):
        """Check MatterGen CSP calculations"""
        chemical_formula = self.ctx.chemical_formula
        for i in range(1, settings.inputs["MatterGen_CSP"]["num_runs"]+1):
            calculation = self.ctx[f"{chemical_formula}_csp-{str(i)}"]
            if not calculation.is_finished_ok:
                self.report(f"MatterGen (CSP) for {chemical_formula} failed")
                return self.exit_codes.ERROR_CALCULATION_FAILED

    def predict_ml_energies(self):
        """Predicting the energies of the structures with the given ML model"""
        self.ctx.ml_jobs = {}
        # generated structures by mattergen gen
        chemical_systems = self.ctx.chemical_systems
        for chemical_system in chemical_systems:
            calculation = self.ctx[f"{chemical_system}_mattergen"]
            output_structures = (
                    calculation.called[-1]
                    .outputs.output_dict["structures"]
            )
            compositions = []
            neutral_structures = []
            for struct in output_structures:
                compositions.append(Structure.from_dict(struct).composition)
            indices = select_charge_neutral(compositions)
            for i in indices:
                neutral_structures.append(output_structures[i])
            # get structures from MPDB
            mpd_structures = get_structures_from_mpdb(chemical_system)

            self.ctx.ml_jobs[f"{chemical_system}_gen"] = neutral_structures + mpd_structures

        # generated structures by mattergen csp
        chemical_formula = self.ctx.chemical_formula
        chemical_system = Composition(chemical_formula).chemical_system
        csp_structures = []
        for i in range(1, settings.inputs["MatterGen_CSP"]["num_runs"]+1):
            calculation = self.ctx[f"{chemical_formula}_csp-{str(i)}"]
            output_structures = (
                    calculation.called[-1]
                    .outputs.output_dict["structures"]
            )
            csp_structures.extend(output_structures)

        self.ctx.ml_jobs[f"{chemical_system}_csp"] = csp_structures


        builders = {
                "MatterSim": self._construct_mattersim_builder,
                "MACE": self._construct_mace_builder
        }

        for job in self.ctx.ml_jobs.keys():
            self.report(f"Running {self.ctx.ML_model} for {job.split('_')[0]} generated by MatterGen ({job.split('_')[-1]})")
            builder_fn = builders[self.ctx.ML_model]
            builder = builder_fn(self.ctx.ml_jobs[job])
            future = self.submit(builder)
            self.to_context(**{job: future})

    def store_ml_energies(self):
        """Storing predicted ML energies"""
        self.report(f"Storing {self.ctx.ML_model} results")
        for job in self.ctx.ml_jobs.keys():
            chemical_system = job.split("_")[0]
            calculation = self.ctx[job]

            if not calculation.is_finished_ok:
                self.ctx.failed_ml_e.append(chemical_system)
                self.report(f"{self.ctx.ML_model} for {chemical_system} failed")
                continue
            try:
                new_structures, new_energies, new_epas = self._extract_outputs(calculation)

                selected_structures, selected_energies, selected_epas = self._select_low_energy_structures(
                        new_structures,
                        new_energies,
                        new_epas,
                        50
                )

                add_structures(chemical_system,
                               self.ctx.ML_model,
                               selected_structures,
                               selected_energies,
                               selected_epas)

                # DBChemsys statu is updated only if gen calculations are done
                if job.split("_")[-1] == "gen":
                    row = query_by_columns(DBChemsys,
                                     {"chemsys": chemical_system}
                                    )[0]
                    update_row(DBChemsys,
                               row.uuid,
                              {"gen_structures": "Ready"}
                    )

            except (AttributeError, IndexError, KeyError):
                self.report(
                    f"Warning: Failed to store results for {chemical_system}"
                )
                self.ctx.failed_ml_e.append(chemical_system)

    def handle_failed_ml_calculations(self):
        """Delete row from db_chemsys if a calculation failed"""
        if self.ctx.failed_ml_e:
            self.report(f"{self.ctx.ML_model} for {self.ctx.failed_ml_e} failed. Corrsponding rows will be removed.")
            # remove corresponding rows from DBChemsys
            for failed_chemsys in self.ctx.failed_ml_e:
                self._cleanup_failed_system(failed_chemsys)
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def wait_for_ml_data(self):
        """ Wait until all chemical systems are available"""
        self.report("Check if required data is available for constructing phase diagram")
        inputs = get_pythonjob_input("is_data_available",
                                     {"chemical_systems": self.ctx.all_chemical_systems},
                                     "localhost"
        )
        future = self.submit(PythonJob, inputs=inputs)
        self.to_context(**{"pyjob": future})

    def check_pythonjob(self):
        calculation = self.ctx["pyjob"]
        if not calculation.is_finished_ok:
            self.report("PythonJob failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def construct_phase_diagram(self):
        """Constructing the compositional phase diagram"""
        chemical_formula = self.ctx.chemical_formula
        self.report("Constructing phase diagram")
        self.ctx.phase_diagram = construct_phase_diagram(
                self.ctx.all_chemical_systems,
                self.ctx.ML_model
        )
        if not self.ctx.phase_diagram:
            self.report(f"Constructing phase diagram for {chemical_formula} failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def return_stable_structures(self):
        chemical_formula = self.ctx.chemical_formula
        comp_red_f = Composition(chemical_formula).reduced_formula
        pd = self.ctx.phase_diagram
        uuid_list = []

        for entry in pd.entries:
            ehull = pd.get_e_above_hull(entry)
            if entry.composition.reduced_formula == comp_red_f and ehull < 0.1:
                uuid_list.append(str(entry.attribute))
        self.out("output_uuids", List(list=uuid_list).store())

    @staticmethod
    def _construct_mattergen_gen_builder(chemical_system):
        Workflow = WorkflowFactory("mattergen.base")
        builder = Workflow.get_builder()
        builder.chemical_system = Str(chemical_system)
        builder.code = Code.get_from_string(
                settings.configs["codes"]["MatterGen"]["code_string"]
        )
#        model_path = settings.configs["codes"]["MatterGen"]["path_to_pretrained_models_gen"]
        model_name = settings.configs["codes"]["MatterGen"]["model_name"]
        builder.job_info = Dict(
                {"job_type": "gen",
                 "model_name": model_name,
                 "energy_above_hull": settings.inputs["MatterGen_generate"]["energy_above_hull"],
                 "batch_size": settings.inputs["MatterGen_generate"]["batch_size"],
                 "num_batches": settings.inputs["MatterGen_generate"]["num_batches"]
                }
        )
        builder.max_iterations = Int(2)
        return builder

    @staticmethod
    def _construct_mattergen_csp_builder(chemical_formula):
        Workflow = WorkflowFactory("mattergen.csp")
        builder = Workflow.get_builder()

        builder.chemical_formula = Str(chemical_formula)
        builder.code = Code.get_from_string(
                settings.configs["codes"]["MatterGen"]["code_string"]
        )
        model_path = settings.configs["codes"]["MatterGen"]["path_to_pretrained_models_csp"]
        builder.job_info = Dict(
                {"job_type": "csp",
                 "model_path": model_path,
                 "batch_size": settings.inputs["MatterGen_CSP"]["batch_size"],
                 "num_batches": settings.inputs["MatterGen_CSP"]["num_batches"]
                }
        )
        builder.max_iterations = Int(2)
        return builder

    @staticmethod
    def _construct_mattersim_builder(structures_list):
        Workflow = WorkflowFactory("mattersim.base")
        builder = Workflow.get_builder()
        builder.input_structures = List(structures_list)
        builder.code = Code.get_from_string(
                settings.configs["codes"]["MatterSim"]["code_string"]
        )
        model_path = os.path.join(
                settings.configs["codes"]["MatterSim"]["path_to_pretrained_models"],
                settings.configs["codes"]["MatterSim"]["pretrained_model"]
        )
        device = settings.configs["codes"]["MatterSim"]["job_script"]["device"]
        builder.job_info = Dict(
                {"job_type": "relax",
                 "num_structures": str(len(structures_list)),
                 "model_path": model_path,
                 "model_name": "chemical_system_energy_above_hull",
                 "device": device,
                 "fmax": settings.inputs["MatterSim_relax"]["fmax"],
                 "max_steps": settings.inputs["MatterSim_relax"]["max_steps"]
                }
        )
        return builder

    @staticmethod
    def _construct_mace_builder(structures_list):
        Workflow = WorkflowFactory("mace.base")
        builder = Workflow.get_builder()
        builder.input_structures = List(structures_list)
        builder.code = Code.get_from_string(
                settings.configs["codes"]["MACE"]["code_string"]
        )
        model_path = os.path.join(
                settings.configs["codes"]["MACE"]["path_to_pretrained_models"],
                settings.configs["codes"]["MACE"]["pretrained_model"]
        )
        device = settings.configs["codes"]["MACE"]["job_script"]["device"]
        builder.job_info = Dict(
                {"job_type": "relax",
                 "num_structures": str(len(structures_list)),
                 "model_path": model_path,
                 "device": device,
                 "fmax": settings.inputs["MACE_relax"]["fmax"],
                 "max_steps": settings.inputs["MACE_relax"]["max_steps"]
                }
        )
        return builder

    def _extract_outputs(self, calculation):
        """Extract structure/energy/epa outputs from a calculation."""
        output_dict = calculation.called[-1].outputs.output_dict
        structures = output_dict["structures"]
        energies = output_dict["energies"]
        epas = output_dict["epas"]
        return structures, energies, epas

    def _select_low_energy_structures(self, structures, energies, epas, n_struct):
        """Select the lowest n structures based on energy per atom."""
        sorted_indices = sorted(range(len(epas)), key=lambda i: epas[i])[:n_struct]

        selected_structures = [structures[i] for i in sorted_indices]
        selected_energies = [energies[i] for i in sorted_indices]
        selected_epas = [epas[i] for i in sorted_indices]
        return selected_structures, selected_energies, selected_epas

    def _cleanup_failed_system(self, chemsys):
        """Remove database entries for failed calculations"""
        result = query_by_columns(DBChemsys, {"chemsys": chemsys})
        if result:
            delete_row(DBChemsys, result[0])
