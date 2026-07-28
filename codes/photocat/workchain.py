import os
import json
import tempfile
from aiida.engine import BaseRestartWorkChain, while_
from aiida.orm import List, Dict, SinglefileData, Code, Str
from aiida.plugins import CalculationFactory
from uvsib.workflows import settings


def get_options():
    """Scheduler options from config.yaml codes.photocat.job_script."""
    job_script = settings.configs['codes']['photocat']['job_script']
    resources = {
        'num_machines': job_script['nodes'],
        'num_mpiprocs_per_machine': job_script['ntasks'],
        'num_cores_per_mpiproc': job_script['cpus'],
    }
    options = {
        'resources': resources,
        'max_wallclock_seconds': job_script['time'],
        'parser_name': 'photocat_parser'
    }
    if job_script.get('exclusive'):
        options.update({'custom_scheduler_commands': '#SBATCH --exclusive'})
    return options


def get_structures_file(structures):
    """Stage the tagged bulk structures as input_structures.json."""
    file_path = os.path.join(tempfile.gettempdir(), "input_structures.json")
    with open(file_path, 'w') as f:
        json.dump(structures, f)
    return SinglefileData(file=file_path)


def _photocat_cmdline(job_info):
    """Photocat is NOT an MLIP job -- no model/device prefix, only the
    gap-filter parameters the runner understands."""
    gap_max = job_info.get('gap_max')
    return [
        f"--models={','.join(job_info['models'])}",
        f"--gap_min={job_info['gap_min']}",
        f"--gap_max={'none' if gap_max is None else gap_max}",
        f"--sigma_model={job_info['sigma_model']}",
    ]


PhotocatCalculation = CalculationFactory('photocat')


class PhotocatCalcWorkChain(BaseRestartWorkChain):
    """BaseRestartWorkChain around PhotocatCalculation."""
    _process_class = PhotocatCalculation

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
        spec.expose_outputs(PhotocatCalculation)
        spec.exit_code(400, 'ERROR_MAX_RESTARTS_EXCEEDED',
                       message='Maximum number of restarts exceeded for PhotocatCalcWorkChain.')

    def setup(self):
        super().setup()
        job_info = self.inputs.job_info.get_dict()
        self.ctx.inputs = {
            'code': self.inputs.code,
            'file': {'input_structures_file':
                     get_structures_file(self.inputs.input_structures.get_list())},
            'parameters': Dict(dict={
                'job_type': 'photocat',
                'cmdline_params': _photocat_cmdline(job_info),
            }),
            'metadata': {
                'options': get_options(),
                'label': 'PhotoCat: {}'.format(self.inputs.local_label.value),
            }
        }
