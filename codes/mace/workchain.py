import io
import json
from aiida.engine import BaseRestartWorkChain, while_
from aiida.orm import List, Dict, SinglefileData, Code, Str
from aiida.plugins import CalculationFactory
from uvsib.workflows import settings
from uvsib.codes.utils import get_cmdline


def get_options():
    """Return scheduler options"""
    job_script = settings.configs['codes']['MACE']['job_script']
    resources = {
        'num_machines': job_script['nodes'],
        'num_mpiprocs_per_machine': job_script['ntasks'],
        'num_cores_per_mpiproc': job_script['cpus'],
    }
    options = {
        'resources': resources,
        'max_wallclock_seconds': job_script['time'],
        'parser_name': 'mace_parser'
    }
    if job_script['exclusive']:
        options.update({'custom_scheduler_commands' : '#SBATCH --exclusive'})
    return options

def get_structures_file(structures):
    """Stage the structures as a SinglefileData named input_structures.json.

    Built straight from memory: staging via a fixed path in the system temp dir
    races between daemon workers, which can hand a job another job's structures
    (silently) or a half-written file.
    """
    payload = json.dumps(structures).encode('utf-8')
    return SinglefileData(io.BytesIO(payload), filename='input_structures.json')

MACECalculation = CalculationFactory('mace')

class MACEWorkChain(BaseRestartWorkChain):
    """BaseRestartWorkChain to run MACECalculation with automatic restarts"""
    _process_class = MACECalculation
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('input_structures', valid_type=List)
        spec.input("code", valid_type=Code)
        spec.input('job_info', valid_type=Dict)
        spec.input('local_label', valid_type=Str)
        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results
        )

        spec.expose_outputs(MACECalculation)

        spec.exit_code(400, 'ERROR_MAX_RESTARTS_EXCEEDED', message='Maximum number of restarts exceeded for MACEWorkChain.')

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
                'label': 'MACE: {}'.format(self.inputs.local_label.value)
            }
        }
