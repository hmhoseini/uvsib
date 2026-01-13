import os
import yaml
from pymatgen.core.structure import Structure
from aiida.engine import WorkChain
from aiida.plugins import WorkflowFactory
from aiida.orm import Str, List, Dict, StructureData, load_code
from uvsib.codes.vasp.workchains import construct_vasp_builder
from uvsib.db.utils import query_structure, add_slab
from uvsib.workchains.utils import get_code, get_model_device
from uvsib.workflows import settings

_EG_MIN = 0
_EG_MAX = 6

def get_struct_uuid(chemical_formula):
    """Query structures from the database and return list of (structure_dict, uuid)"""
    struct_uuid = []
    results = query_structure({"composition": chemical_formula}, method = "HSE")
    for hse_row in results:
        eg = hse_row.band_info.get("energy")
        if eg is None or not _EG_MIN <= eg <= _EG_MAX:
            continue
        struct_uuid.append([hse_row.structure, str(hse_row.structure_uuid)])
    return struct_uuid

def read_yaml(file_path):
    """Read a yaml file"""
    with open(file_path, "r", encoding="utf8") as fhandle:
        return yaml.safe_load(fhandle)

class SurfaceBuilderWorkChain(WorkChain):
    """SurfaceBuilder WorkChain"""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("ML_model", valid_type=Str)
        spec.input("chemical_formula", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.run_facebuild,
            cls.inspect_facebuild,
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
            message="No structures were found for the given formula and the band gap criterion"
        )

    def setup(self):
        """Setup and report"""
        self.report("Running SurfaceBuilder WorkChain")
        self.ctx.protocol = read_yaml(
                os.path.join(settings.vasp_files_path, "protocol.yaml")
        )
        self.ctx.potential_family = settings.configs["codes"]["VASP"]["potential_family"]
        potential_mapping = read_yaml(os.path.join(settings.vasp_files_path, "potential_mapping.yaml"))
        self.ctx.potential_mapping = potential_mapping ["potential_mapping"]
        self.ctx.vasp_code = load_code(
                settings.configs["codes"]["VASP"]["code_string"]
        )
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.ML_model = self.inputs.ML_model.value
        self.ctx.struct_uuid = get_struct_uuid(self.ctx.chemical_formula)
        if not self.ctx.struct_uuid:
            self.report("No structures were found for the given formula and the band gap criterion")
            return self.exit_codes.ERROR_NO_STRUCTURES_FOUND
        self.ctx.slabs_uuid = []

    def run_facebuild(self):
        """Run SurfaceBuilder Workchain"""
        for struct_dict, uuid_str in self.ctx.struct_uuid:
            structure_row = query_structure({"uuid": uuid_str}, method = "PBE")[0]
            bulk_energy = structure_row.energy
            builder = self._construct_facebuild_builder(
                    struct_dict,
                    bulk_energy,
                    self.ctx.ML_model)
            future = self.submit(builder)
            self.to_context(**{f"sfb_{uuid_str}": future})

    def inspect_facebuild(self):
        """Inspect SurfaceBuilder WorkChain"""
        failed_jobs = 0
        for _, uuid_str in self.ctx.struct_uuid:
            sfb_wch = self.ctx[f"sfb_{uuid_str}"]
            if not sfb_wch.is_finished_ok:
                failed_jobs += 1
                continue
            output_dict = sfb_wch.called[-1].outputs.output_dict
            try:
                self.ctx.slabs_uuid.append([output_dict["structures"], uuid_str])
            except:
                self.report(f"Warning: no (orthogonal) slab was found for the structure with uuid={uuid_str}")


    def store_results(self):
        """Store results"""
        return
        for slabs, uuid_str in self.ctx.slabs_uuid:
            for slab in slabs:
                add_slab(uuid_str, slab)

    def final_report(self):
        """Final report"""
        self.report(f"SurfaceBuilderWorkChain for {self.ctx.chemical_formula} finished successfully")

    @staticmethod
    def _construct_facebuild_builder(ml_structure, ml_energy, ML_model):
        """
        Builder for generating surface and surface optimiziation with MatterSim or MACE
        """
        structure = [ml_structure]

        Workflow = WorkflowFactory(ML_model.lower()) # "MatterSim" -> "mattersim", "MACE" -> "mace"

        builder = Workflow.get_builder()

        builder.input_structures = List(structure)
        builder.code = get_code(ML_model)

        _, model_path, device = get_model_device(ML_model)

        relax_key = "face_build"
        job_info = {
            "job_type": "facebuild",
            "model_path": model_path,
            "device": device,
            "fmax": settings.inputs[relax_key]["fmax"],
            "max_steps": settings.inputs[relax_key]["max_steps"],
            "max_miller_idx": settings.inputs[relax_key]["max_miller_idx"],
            "bulk_energy": ml_energy,
            "percentage_to_select": settings.inputs[relax_key]["percentage_to_select"]
        }

        builder.job_info = Dict(job_info)
        return builder
