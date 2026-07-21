import os
import json
import tempfile
import yaml
from aiida.engine import WorkChain
from aiida.orm import Str, Dict, SinglefileData
from aiida.plugins import CalculationFactory
from uvsib.db.utils import query_structure, add_slab, get_structs_by_uuids, query_by_columns
from uvsib.db.tables import DBComposition
from uvsib.workchains.utils import get_code, get_model_device
from uvsib.workflows import settings


_SKIP_PD_VERIFICATION = settings._SKIP_PD_VERIFICATION #TODO for test

MAX_SLABS_PER_CHUNK = 50

FaceCalculation = CalculationFactory(str(settings.inputs["face_build"]["model"]).lower())


def get_struct_uuid(chemical_formula, ml_model):
    """Structures to build surfaces for: ((structure_dict, uuid) pairs, from_manifest).

    The ``DBComposition.stable_struct`` manifest is authoritative when
    present: the producing workchain (phase diagram or SQS) wrote the exact
    uuid list to process. For an SQS sweep the parent formula's row carries
    the union over ALL grid compositions, so one surface-builder run on the
    parent covers the whole sweep. Rows without a manifest fall back to the
    legacy composition+method query (capped at 10).

    ``from_manifest=True`` is forwarded to the slab-generation runner: these
    bulks were deliberately produced (an SQS cell is P1 by construction), so
    the runner's size+symmetry skip -- a guard against ACCIDENTAL large-P1
    cells from relaxation noise -- is bypassed for them; only the per-bulk
    walltime cap still applies.
    """
    rows = query_by_columns(DBComposition, {"composition": chemical_formula})
    ss = rows[0].stable_struct if rows else None
    if ss and ss.get("ml_uuid_list"):
        return get_structs_by_uuids(ss["ml_uuid_list"], ml_model), True

    if _SKIP_PD_VERIFICATION:
        results = query_structure({"composition": chemical_formula}, method=ml_model)
        filtered_results = list()
        for s, u in sorted([(row.structure, str(row.structure_uuid)) for row in results], key=lambda x: x[1]):
            filtered_results.append([s, u])
            if len(filtered_results) == 10:
                break
        return filtered_results, False
    results = query_structure({"composition": chemical_formula}, method="r2SCAN") or []
    return [(row.structure, str(row.structure_uuid)) for row in results], False


def read_yaml(file_path):
    """Read a yaml file"""
    with open(file_path, "r", encoding="utf8") as fhandle:
        return yaml.safe_load(fhandle)


def _chunk(seq, size):
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _structures_file(payload):
    path = os.path.join(tempfile.mkdtemp(), "input_structures.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return SinglefileData(file=path)


def _model_cmdline():
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
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.struct_uuid, self.ctx.from_manifest = \
            get_struct_uuid(self.ctx.chemical_formula, settings.inputs['bulk_relax']['model'])
        if not self.ctx.struct_uuid:
            self.report(f"No structures were found for {self.ctx.chemical_formula}")
            return self.exit_codes.ERROR_NO_STRUCTURES_FOUND
        self.ctx.slabs_uuid = []  # uuid -> {"epa": float, "n_chunks": int} after the generation phase.
        self.ctx.relax_plan = {}
        self.report("Running SurfaceBuilder WorkChain for {}".format(self.ctx.chemical_formula))

    def run_slabgen(self):
        """Submit one slab-generation CalcJob PER bulk structure.

        Mirrors the relax-side chunking: an SQS sweep hands over the whole
        composition grid at once, and a single all-bulks job would grow
        unpredictably long -- per-bulk jobs stay inside walltime and a
        pathological bulk fails alone instead of taking the run down."""
        for struct_dict, uuid_str in self.ctx.struct_uuid:
            payload = [{"uuid": uuid_str, "structure": struct_dict,
                        "from_manifest": self.ctx.from_manifest}]
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
                    "label": f"Faces: {self.ctx.chemical_formula}",
                },
            }
            self.to_context(**{f"slabgen_{uuid_str}": self.submit(FaceCalculation, **inputs)})

    def inspect_slabgen(self):
        """Read epa + orthogonal slabs per bulk from its dedicated gen job."""
        for struct_dict, uuid_str in self.ctx.struct_uuid:
            node = self.ctx[f"slabgen_{uuid_str}"]
            if not node.is_finished_ok:
                self.report(f"Slab generation job failed for uuid={uuid_str}; "
                            "this bulk is dropped.")
                continue
            for res in node.outputs.output_dict.get_dict().get("results", []):
                slabs = res.get("slabs", [])
                if not slabs:
                    self.report(f"Warning: no orthogonal slab generated for uuid={uuid_str}")
                    continue
                n_chunks = len(_chunk(slabs, MAX_SLABS_PER_CHUNK))
                # Store only epa + chunk count; slabs are re-read from the gen
                # node in run_relax to keep the workchain checkpoint small.
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
        for uuid_str, plan in self.ctx.relax_plan.items():
            # Re-read this bulk's slabs from its own gen node; re-chunk
            # deterministically so chunk i here matches inspect_relax.
            results = self.ctx[f"slabgen_{uuid_str}"].outputs.output_dict.get_dict().get("results", [])
            uuid2slabs = {r.get("uuid"): r.get("slabs", []) for r in results}
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
                        "label": f"SlabRelax: {self.ctx.chemical_formula} #{i}",
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
