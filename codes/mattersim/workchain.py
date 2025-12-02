import os
import json
import tempfile
from aiida.engine import BaseRestartWorkChain, while_
from aiida.orm import List, Dict, SinglefileData, Code
from aiida.plugins import CalculationFactory
from uvsib.workflows import settings


def get_options():
    """Return scheduler options"""
    job_script = settings.configs['codes']['MatterSim']['job_script']
    resources = {
        'num_machines': job_script['nodes'],
        'num_mpiprocs_per_machine': job_script['ntasks'],
        'num_cores_per_mpiproc': job_script['cpus'],
    }
    options = {
        'resources': resources,
        'max_wallclock_seconds': job_script['time'],
        'parser_name': 'mattersim_parser'
    }
    if job_script['exclusive']:
        options.update({'custom_scheduler_commands' : '#SBATCH --exclusive'})
#    if job_script['gpu']:
#        options.update({'custom_scheduler_commands' : '#SBATCH --exclusive'})
    return options

def get_structures_file(structures):
    """ temp structure file """
    filename = "input_structures.json"
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, 'w') as f:
        json.dump(structures, f)
    return SinglefileData(file=file_path)

def get_cmdline(job_info):
    """Construct MatterSim command line"""
    job_type = job_info['job_type']
    if job_type == 'relax':
        cmdline = [
            f"--model_path={job_info['model_path']}",
            f"--device={job_info['device']}",
            f"--fmax={job_info['fmax']}",
            f"--max_steps={job_info['max_steps']}",
        ]
    elif job_type == 'facebuild':
        cmdline = [
            f"--bulk_energy={job_info['bulk_energy']}",
            f"--model_path={job_info['model_path']}",
            f"--device={job_info['device']}",
            f"--fmax={job_info['fmax']}",
            f"--max_steps={job_info['max_steps']}",
            f"--max_miller_idx={job_info['max_miller_idx']}",
            f"--percentage_to_select={job_info['percentage_to_select']}"
        ]
    elif job_type == 'adsorbates':
        cmdline = [
            f"--model_path={job_info['model_path']}",
            f"--device={job_info['device']}",
            f"--fmax={job_info['fmax']}",
            f"--max_steps={job_info['max_steps']}",
            f"--reaction={job_info['reaction']}"
        ]
    else:
        cmdline = [
            f"--model_path={job_info['model_path']}"
        ]
    return cmdline

MatterSimCalculation = CalculationFactory('mattersim')

class MatterSimWorkChain(BaseRestartWorkChain):
    """BaseRestartWorkChain to run MatterSimCalculation with automatic restarts."""

    _process_class = MatterSimCalculation

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # Declare the inputs needed for this workchain:
        spec.input('input_structures', valid_type=List)
        spec.input("code", valid_type=Code)
        spec.input('job_info', valid_type=Dict)
        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )

        spec.exit_code(
            400,
            'ERROR_MAX_RESTARTS_EXCEEDED',
            message='Maximum number of restarts exceeded for MatterSimWorkChain.'
        )

    def setup(self):
        """Initialize context before first calculation."""
        super().setup()

        input_structures = self.inputs.input_structures.get_list()
        job_info = self.inputs.job_info

        input_structures_file = get_structures_file(input_structures)

        self.ctx.inputs = {
            'code': self.inputs.code,
            'file': {'input_structures_file': input_structures_file},
            'parameters': Dict(dict={
                'job_type': job_info['job_type'],
                'cmdline_params': get_cmdline(job_info)
            }),
            'metadata': {
                'options': get_options(),
                'label': 'MatterSim calculation'
            }
        }
