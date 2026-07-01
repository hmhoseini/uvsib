import os
import json
import tempfile
from aiida.engine import BaseRestartWorkChain, while_
from aiida.orm import List, Dict, SinglefileData, Code, Str
from aiida.plugins import CalculationFactory
from uvsib.workflows import settings
from uvsib.codes.utils import get_cmdline


def get_options():
    """Return scheduler options"""
    job_script = settings.configs['codes']['SQS']['job_script']
    resources = {
        'num_machines': job_script['nodes'],
        'num_mpiprocs_per_machine': job_script['ntasks'],
        'num_cores_per_mpiproc': job_script['cpus'],
    }
    options = {
        'resources': resources,
        'max_wallclock_seconds': job_script['time'],
        'parser_name': 'sqs_parser'
    }
    if job_script['exclusive']:
        options.update({'custom_scheduler_commands' : '#SBATCH --exclusive'})
    return options

def get_structures_file(structures):
    """ temp structure file """
    # `structures` arrives as an aiida ``List`` node (the workchain input),
    # which json cannot serialise -- unwrap it to the plain Python list first.
    # Its stored contents are already JSON-safe by construction.
    if isinstance(structures, List):
        structures = structures.get_list()
    filename = "input_structures.json"
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, 'w') as f:
        json.dump(structures, f)
    return SinglefileData(file=file_path)


SQSCalculation = CalculationFactory('sqs')


class SQSWorkChain(BaseRestartWorkChain):
    _process_class = SQSCalculation
    @classmethod
    def define(cls, spec):
        super().define(spec)
        # Declare the inputs needed for this workchain:
        spec.input('input_structure', valid_type=List)
        spec.input("code", valid_type=Code)
        spec.input('job_info', valid_type=Dict)
        spec.input('local_label', valid_type=Str)
        spec.expose_outputs(SQSCalculation)

        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results
        )

        spec.exit_code(400,'ERROR_MAX_RESTARTS_EXCEEDED', message='Maximum number of restarts exceeded for SQS WorkChain.')

    def setup(self):
        super().setup()
        input_structure = self.inputs.input_structure
        job_info = self.inputs.job_info
        input_structure_file = get_structures_file(input_structure)

        self.ctx.inputs = {
            'code': self.inputs.code,
            'file': {'input_structures_file': input_structure_file},
            'parameters': Dict(dict={
                'job_type': job_info['job_type'],
                'cmdline_params': get_cmdline(job_info)}),
            'metadata': {
                'options': get_options(),
                'label': 'SQS: {}'.format(self.inputs.local_label.value)}
        }
