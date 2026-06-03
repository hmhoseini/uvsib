import yaml
from aiida.engine import WorkChain
from aiida.plugins import WorkflowFactory
from aiida.orm import Str, List, Dict
from uvsib.db.utils import query_structure, add_slab
from uvsib.workchains.utils import get_code, get_model_device
from uvsib.workflows import settings


_SKIP_PD_VERIFICATION = settings._SKIP_PD_VERIFICATION #TODO for test

def get_struct_uuid(chemical_formula, ml_model): #TODO: will correct after test
    """Query structures from the database by formula and return list of (structure_dict, uuid)"""
    if _SKIP_PD_VERIFICATION:
        results = query_structure({"composition": chemical_formula}, method=ml_model)
        filtered_results = list()
        for s, u in sorted([(row.structure, str(row.structure_uuid)) for row in results], key=lambda x: x[1]):
            filtered_results.append([s, u])
            if len(filtered_results) == 10:
                break
        return filtered_results
    results = query_structure({"composition": chemical_formula}, method = "r2SCAN") or []
    return [(row.structure, str(row.structure_uuid)) for row in results]

def read_yaml(file_path):
    """Read a yaml file"""
    with open(file_path, "r", encoding="utf8") as fhandle:
        return yaml.safe_load(fhandle)

class SurfaceBuilderWorkChain(WorkChain):
    """SurfaceBuilder WorkChain"""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.run_facebuild,
            cls.inspect_facebuild,
            cls.store_results,
            cls.final_report
        )

        spec.exit_code(300,"ERROR_CALCULATION_FAILED", message="The calculation did not finish successfully")
        spec.exit_code(301,"ERROR_NO_STRUCTURES_FOUND", message="No structures were found for the given formula")
        spec.exit_code(302,"ERROR_NO_SURFACE", message="No surface has been generated")

    def setup(self):
        """Setup and report"""
        self.report("Running SurfaceBuilder WorkChain")
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.struct_uuid = get_struct_uuid(self.ctx.chemical_formula, settings.inputs['bulk_relax']['model']) #TODO update
        if not self.ctx.struct_uuid:
            self.report(f"No structures were found for {self.ctx.chemical_formula}")
            return self.exit_codes.ERROR_NO_STRUCTURES_FOUND
        self.ctx.slabs_uuid = []

    def run_facebuild(self):
        """Run SurfaceBuilder Workchain"""
        for struct_dict, uuid_str in self.ctx.struct_uuid:
            builder = self._construct_facebuild_builder(struct_dict)
            future = self.submit(builder)
            self.to_context(**{f"sfb_{uuid_str}": future})

    def inspect_facebuild(self):
        """Inspect SurfaceBuilder WorkChain"""
        for _, uuid_str in self.ctx.struct_uuid:
            wch = self.ctx[f"sfb_{uuid_str}"]
            if not wch.is_finished_ok:
                self.report("Some surface builder jobs crashed.")
                return self.exit_codes.ERROR_CALCULATION_FAILED
            output_dict = wch.outputs.output_dict
            if output_dict:
                self.ctx.slabs_uuid.append([output_dict["slabs"], uuid_str])
            else:
                self.report(f"Warning: no (orthogonal) slab was found for the structure with uuid={uuid_str}")

        if not self.ctx.slabs_uuid:
            self.report("No surface has been generated")
            return self.ERROR_NO_SURFACE

    def store_results(self):
        """Store results"""
        for slabs, uuid_str in self.ctx.slabs_uuid:
            for slab in slabs:
                add_slab(uuid_str, self.ctx.chemical_formula, slab)

    def final_report(self):
        """Final report"""
        self.report(f"SurfaceBuilderWorkChain for {self.ctx.chemical_formula} finished successfully")

    def _construct_facebuild_builder(self, ml_structure):
        """Builder for generating surface and surface optimiziation"""
        ML_model = settings.inputs['face_build']['model']
        structure = [ml_structure]
        Workflow = WorkflowFactory(ML_model.lower())
        builder = Workflow.get_builder()
        builder.input_structures = List(structure)
        builder.code = get_code(ML_model)
        builder.local_label = Str("FaceBuild: {}".format(self.ctx.chemical_formula))
        model, model_path, device = get_model_device(ML_model)

        job_info = {
            "job_type": "face_build",
            "ML_model": ML_model,
            "model_name": model,
            "model_path": model_path,
            "model_head": settings.inputs["face_build"]['head'],
            "device": device,
            "fmax": settings.inputs["face_build"]["fmax"],
            "max_steps": settings.inputs["face_build"]["max_steps"],
            "max_miller_idx": settings.inputs["face_build"]["max_miller_idx"],
            "max_num_surf": settings.MAX_NUM_SURF
        }

        builder.job_info = Dict(job_info)
        return builder
