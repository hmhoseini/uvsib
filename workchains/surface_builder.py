import os
import json
import tempfile
import yaml
from aiida.engine import WorkChain
from aiida.orm import Str, Dict, SinglefileData
from aiida.plugins import CalculationFactory
from uvsib.db.utils import query_structure, add_slab
from uvsib.workchains.utils import get_code, get_model_device
from uvsib.workflows import settings

MAX_NUM_BULK = settings.MAX_NUM_BULK
_PD_VERIFICATION = settings._PD_VERIFICATION

# Slabs are relaxed in balanced batches of at most this many per CalcJob, so a
# structure that generates many faces is spread over several jobs instead of one
# long, unpredictable relaxation.
MAX_SLABS_PER_CHUNK = 250

FaceCalculation = CalculationFactory(str(settings.inputs["face_build"]["model"]).lower())

def get_struct_uuid(chemical_formula): #TODO: will correct after test
    """Query structures from the database by formula and return list of (structure_dict, uuid)"""
    if _PD_VERIFICATION:
        results = query_structure({"composition": chemical_formula}, method = "r2SCAN") or []
        return [(row.structure, str(row.structure_uuid)) for row in results] #TODO it retruns DFT optimized structures
    else:
        model = settings.inputs['bulk_relax']['model']
        results = query_structure({"composition": chemical_formula}, method=model)
        filtered_results = []
        for s, u in sorted([(row.structure, str(row.structure_uuid)) for row in results], key=lambda x: x[1]):
            filtered_results.append([s, u])
            if len(filtered_results) == MAX_NUM_BULK:
                break
        return filtered_results

def read_yaml(file_path):
    """Read a yaml file"""
    with open(file_path, "r", encoding="utf8") as fhandle:
        return yaml.safe_load(fhandle)


def _chunk(seq, size):
    """Split a list into consecutive chunks of at most ``size`` items."""
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _structures_file(payload):
    """Stage ``payload`` (a JSON-serialisable list) as input_structures.json."""
    path = os.path.join(tempfile.mkdtemp(), "input_structures.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return SinglefileData(file=path)


def _model_cmdline():
    """Model/calculator CLI args for the face-build MLIP on the v100 code."""
    ml_model = settings.inputs["face_build"]["model"]
    model_name, model_path, device = get_model_device(ml_model)
    head = settings.inputs["face_build"]["head"]
    return [
        f"--ML_model={ml_model}",
        f"--model={model_name}",
        f"--model_path={model_path}",
        f"--task_name={head}",
        f"--device={device}",
    ]


def _facebuild_options():
    """Scheduler options for a slab job on the minimahopping code (generic parser).
    NOTE: the calculation ``label`` is NOT an option -- it lives at
    ``metadata.label`` (set by the callers). ``metadata.options`` is a fixed
    namespace and rejects unknown keys like 'label'.
    """
    js = settings.configs["codes"][settings.inputs["face_build"]["model"]]["job_script"]
    options = {
        "resources": {
            "num_machines": js["nodes"],
            "num_mpiprocs_per_machine": js["ntasks"],
            "num_cores_per_mpiproc": js["cpus"],
        },
        "max_wallclock_seconds": js["time"],
        "parser_name": "sqs_parser",   # generic output.json -> Dict
    }
    if js.get("exclusive"):
        options["custom_scheduler_commands"] = "#SBATCH --exclusive"
    return options


class SurfaceBuilderWorkChain(WorkChain):
    """SurfaceBuilder WorkChain.

    Slab enumeration (``generate_all_slabs``, now bounded -- see
    ``codes/files/slab_generate.py`` "Robustness": symprec-ladder standardize +
    ``max_normal_search=1`` + size/symmetry skip + per-bulk walltime cap) and slab
    relaxation are two separate CalcJobs on the gnome@v100 code: one
    generation job per bulk structure, then its orthogonal slabs are split into
    balanced chunks (<= MAX_SLABS_PER_CHUNK) and relaxed in parallel. The global
    lowest-surface-energy faces (up to settings.MAX_NUM_SURF) are selected here
    after a structure's chunks all return.
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.run_slabgen,
            cls.inspect_slabgen,
            cls.run_relax,
            cls.inspect_relax,
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
        self.ctx.struct_uuid = get_struct_uuid(self.ctx.chemical_formula) #TODO update
        if not self.ctx.struct_uuid:
            self.report(f"No structures were found for {self.ctx.chemical_formula}")
            return self.exit_codes.ERROR_NO_STRUCTURES_FOUND
        self.ctx.slabs_uuid = []
        # uuid -> {"epa": float, "n_chunks": int} after the generation phase.
        self.ctx.relax_plan = {}

    def run_slabgen(self):
        """Submit a SINGLE slab-generation CalcJob covering all bulk structures.
        The model is loaded once and reused for every bulk inside the runner, so
        all generation happens in one job instead of one per structure."""
        payload = [{"uuid": uuid_str, "structure": struct_dict}
                   for struct_dict, uuid_str in self.ctx.struct_uuid]
        inputs = {
            "code": get_code(settings.inputs["face_build"]["model"]),
            "parameters": Dict(dict={
                "job_type": "face_build",
                "cmdline_params": _model_cmdline() + [
                    f"--max_miller_idx={settings.inputs['face_build']['max_miller_idx']}"],
                "staged_files": [["slab_generate.py", "aiida.py"],
                                 ["_calculators.py", "_calculators.py"]],
                "retrieve_list": ["output.json"],
            }),
            "file": {"input_structures_file": _structures_file(payload)},
            "metadata": {
                "options": _facebuild_options(),
                "label": f"SlabGen: {self.ctx.chemical_formula} ({len(payload)} bulks)",
            },
        }
        self.to_context(slabgen=self.submit(FaceCalculation, **inputs))

    def inspect_slabgen(self):
        """Read epa + orthogonal slabs per bulk from the single gen job."""
        node = self.ctx.slabgen
        if not node.is_finished_ok:
            self.report("Slab generation job failed.")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        for res in node.outputs.output_dict.get_dict().get("results", []):
            uuid_str = res.get("uuid")
            slabs = res.get("slabs", [])
            if not slabs:
                self.report(f"Warning: no orthogonal slab generated for uuid={uuid_str}")
                continue
            n_chunks = len(_chunk(slabs, MAX_SLABS_PER_CHUNK))
            # Store only epa + chunk count; slabs are re-read from the gen node
            # in run_relax to keep the workchain checkpoint small.
            self.ctx.relax_plan[uuid_str] = {"epa": res["epa"], "n_chunks": n_chunks}
            self.report(
                f"uuid={uuid_str}: {len(slabs)} orthogonal slabs "
                f"(of {res.get('n_total', '?')} generated) -> "
                f"{n_chunks} relax chunk(s) of <= {MAX_SLABS_PER_CHUNK}")

        if not self.ctx.relax_plan:
            self.report("No orthogonal slabs were generated for any structure")
            return self.exit_codes.ERROR_NO_SURFACE

    def run_relax(self):
        """Submit one relaxation CalcJob per slab chunk on the model code."""
        # The single gen job holds every bulk's slabs; map uuid -> slabs once.
        results = self.ctx.slabgen.outputs.output_dict.get_dict().get("results", [])
        uuid2slabs = {r["uuid"]: r.get("slabs", []) for r in results}
        for uuid_str, plan in self.ctx.relax_plan.items():
            # Re-chunk deterministically so chunk i here matches inspect_relax.
            slabs = uuid2slabs.get(uuid_str, [])
            for i, chunk in enumerate(_chunk(slabs, MAX_SLABS_PER_CHUNK)):
                inputs = {
                    "code": get_code(settings.inputs['face_build']['model']),
                    "parameters": Dict(dict={
                        "job_type": "face_relax",
                        "cmdline_params": _model_cmdline() + [
                            f"--epa={plan['epa']}",
                            f"--fmax={settings.inputs['face_build']['fmax']}",
                            f"--max_steps={settings.inputs['face_build']['max_steps']}"],
                        "staged_files": [["slab_relax.py", "aiida.py"],
                                         ["_calculators.py", "_calculators.py"]],
                        "retrieve_list": ["output.json"],
                    }),
                    "file": {"input_structures_file": _structures_file(chunk)},
                    "metadata": {
                        "options": _facebuild_options(),
                        "label": f"SlabRelax: {self.ctx.chemical_formula} ({uuid_str}) #{i}",
                    },
                }
                self.to_context(**{f"relax_{uuid_str}_{i}": self.submit(FaceCalculation, **inputs)})

    def inspect_relax(self):
        """Merge a structure's chunks, globally select the lowest-energy faces."""
        max_num_surf = settings.MAX_NUM_SURF
        for uuid_str, plan in self.ctx.relax_plan.items():
            merged = []
            for i in range(plan["n_chunks"]):
                node = self.ctx[f"relax_{uuid_str}_{i}"]
                if not node.is_finished_ok:
                    self.report(f"Relax chunk {i} failed for uuid={uuid_str}; "
                                "its slabs are dropped.")
                    continue
                merged.extend(node.outputs.output_dict.get_dict().get("slabs", []))

            if not merged:
                self.report(f"Warning: no slab converged for uuid={uuid_str}")
                continue

            merged.sort(key=lambda x: x["surface_formation_energy"])
            selected = [entry["slab"] for entry in merged[:max_num_surf]]
            if len(merged) < max_num_surf:
                self.report(f"uuid={uuid_str}: only {len(merged)} slabs converged, "
                            f"requested {max_num_surf}")
            self.ctx.slabs_uuid.append([selected, uuid_str])

        if not self.ctx.slabs_uuid:
            self.report("No surface has been generated")
            return self.exit_codes.ERROR_NO_SURFACE

    def store_results(self):
        """Store results"""
        for slabs, uuid_str in self.ctx.slabs_uuid:
            for slab in slabs:
                add_slab(uuid_str, self.ctx.chemical_formula, slab)

    def final_report(self):
        """Final report"""
        self.report(f"SurfaceBuilderWorkChain for {self.ctx.chemical_formula} finished successfully")
