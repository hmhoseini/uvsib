from pymatgen.core import Composition
from aiida.orm import Str, List, Dict
from aiida.engine import WorkChain
from aiida.plugins import WorkflowFactory
from uvsib.db.utils import (
        add_version_to_existing_structure,
        query_structure)
from uvsib.workchains.utils import (
        get_output_as_entry,
        get_ref_entries,
        get_code,
        add_from_mpdb,
        get_model_device)
from uvsib.workflows import settings

def get_struct_uuid(chemical_formula):
    struct_uuid = []
    results = query_structure({"composition": chemical_formula}, method = "DFT")
    for r in results:
        if r.source in ["MPDB_stb", "MPDB_exp"]:
            struct_uuid.append((r.structure, r.source, r.structure_uuid))
    return struct_uuid

def get_ref_struct_uuid(chemical_formula, ML_model):
    struct_uuid = []
    elements = [element.symbol for element in Composition(chemical_formula).elements]
    for el in elements:
        ML_result = query_structure({"chemsys": el}, source = "MPDB_ref", method = ML_model)
        if not ML_result:
            mpdb_result = query_structure({"chemsys": el}, source = "MPDB_ref", method = "DFT")
            r = mpdb_result[0]
            struct_uuid.append((r.structure, r.source, r.structure_uuid))
    return struct_uuid

class MPDBMLWorkChain(WorkChain):
    """Work chain for ML relaxation of MPDB structures"""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)
        spec.input("ML_model", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.run_relax_mpdb_structures,
            cls.store_ml_energies
        )

        spec.exit_code(300, "ERROR_CALCULATION_FAILED", message="The WorkChain did not finish successfully")
        spec.exit_code(301, "ERROR_NO_STRUCTURES_FOUND", message="No experimentally observed structures were found")

    def setup(self):
        """Setup and report"""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.ML_model = self.inputs.ML_model.value
        self.report(f"Running ML relaxation WorkChain for MPDB structures for {self.ctx.chemical_formula}")
        add_from_mpdb(self.ctx.chemical_formula)
        self.ctx.struct_uuid = get_struct_uuid(self.ctx.chemical_formula)
        if not self.ctx.struct_uuid:
            self.report(f"Warning: no structures from the MPDB was found for {self.ctx.chemical_formula}.")
        else:
            self.report(f"{len(self.ctx.struct_uuid)} structures from the MPDB was found for {self.ctx.chemical_formula}.")
        ref_structs = get_ref_struct_uuid(self.ctx.chemical_formula, self.ctx.ML_model)
        if ref_structs:
            self.ctx.struct_uuid.extend(ref_structs)
            self.report(f"{len(ref_structs)} reference structures")

    def run_relax_mpdb_structures(self):
        """Optimize structures from MPDB"""
        structs = []
        for s, _, _ in self.ctx.struct_uuid:
            structs.append(s)
        builder = self._construct_ML_relax_builder(structs)
        self.to_context(**{"ml_e": self.submit(builder)})

    def store_ml_energies(self):
        """Collect ML-calculated energies"""
        ML_model = self.ctx.ML_model
        wch = self.ctx.ml_e
        if not wch.is_finished_ok:
            return self.exit_codes.ERROR_ML_RELAX_FAILED
        try:
            new_entries = get_output_as_entry(wch)
        except Exception:
            return self.exit_codes.ERROR_ML_RELAX_FAILED

        if len(new_entries) != len(self.ctx.struct_uuid):
            self.report("ERROR: ML relaxation for some structures failed")
            return self.exit_codes.ERROR_ML_RELAX_FAILED

        structure_energy_pairs = []
        for entry in new_entries:
            structure_energy_pairs.append((entry.structure.as_dict(), entry.energy))
        for i, structure_energy in enumerate(structure_energy_pairs):
            add_version_to_existing_structure(
                    self.ctx.struct_uuid[i][-1],
                    structure_energy[0],
                    ML_model,
                    {
                     "source": self.ctx.struct_uuid[i][1],
                     "energy": structure_energy[-1]
                    }
            )

    def final_report(self):
        """Final report"""
        _, missing = get_ref_entries(self.ctx.chemical_formula, self.ctx.ML_model)
        if missing:
            self.report(f"Warning: DFT-fallback elemental refs for {missing} (per-element offset risk)")
        self.report("ML relaxation WorkChain for MPDB structures finished successfully")

    ################################################################################
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
