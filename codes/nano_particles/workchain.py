"""Builder helpers for the nano-particle adsorbates CalcJob.

The outer chain (uvsib.workchains.nano_particles.NanoParticleWorkChain)
submits one NanoParticle per DBNanoParticles row, directly by class
-- no BaseRestartWorkChain wrapper and no CalculationFactory entry point are
needed for that. This module only centralizes the scheduler options and
command line so they live next to the calculation they belong to.
"""
import json
import os
import tempfile

from aiida.orm import Dict, SinglefileData

from uvsib.workflows import settings


def nano_job_options(ml_model):
    """Scheduler options from the MLIP code's job_script config; parsed with
    the generic registered ``sqs_parser`` (output.json -> output_dict)."""
    job_script = settings.configs["codes"][ml_model]["job_script"]
    options = {
        "resources": {
            "num_machines": job_script["nodes"],
            "num_mpiprocs_per_machine": job_script["ntasks"],
            "num_cores_per_mpiproc": job_script["cpus"],
        },
        "max_wallclock_seconds": job_script["time"],
        "parser_name": "sqs_parser",
    }
    if job_script.get("exclusive"):
        options["custom_scheduler_commands"] = "#SBATCH --exclusive"
    return options


def nano_cmdline(ml_model, model_name, model_path, head, device, job_cfg, reaction, pathway):
    """Command line for codes/files/nano_particles.py (argparse contract).

    Unlike the model workflows this does NOT go through codes/utils.get_cmdline
    -- its nano_particles branch belongs to the retired generator flow.
    """
    return [
        f"--ML_model={ml_model}",
        f"--model={model_name}",
        f"--model_path={model_path}",
        f"--task_name={head}",
        f"--device={device}",
        f"--fmax={job_cfg['fmax']}",
        f"--max_steps={job_cfg['max_steps']}",
        f"--reaction={reaction}",
        f"--pathway={pathway}",
        f"--max_sites={job_cfg.get('max_sites', 24)}",
        f"--surface_cn_max={job_cfg.get('surface_cn_max', 9)}",
    ]


def batch_payload_file(particles):
    """input_structures.json for one batch of (uuid, structure_dict) pairs
    (the runner's input contract)."""
    path = os.path.join(tempfile.mkdtemp(), "input_structures.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"uuid": str(u), "structure": s} for u, s in particles], f)
    return SinglefileData(file=path)


def nano_parameters(job_type, cmdline):
    return Dict(dict={"job_type": job_type, "cmdline_params": cmdline})
