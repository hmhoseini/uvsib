import io
import json
from aiida.engine import BaseRestartWorkChain, while_
from aiida.orm import List, Dict, SinglefileData, Code
from aiida.plugins import CalculationFactory
from uvsib.workflows import settings
from uvsib.codes.utils import get_cmdline


def get_options():
    """Return scheduler options"""
    job_script = settings.configs['codes']['uPET']['job_script']
    resources = {
        'num_machines': job_script['nodes'],
        'num_mpiprocs_per_machine': job_script['ntasks'],
        'num_cores_per_mpiproc': job_script['cpus'],
    }
    options = {
        'resources': resources,
        'max_wallclock_seconds': job_script['time'],
        'parser_name': 'similarity_parser'
    }
    if job_script['exclusive']:
        options.update({'custom_scheduler_commands' : '#SBATCH --exclusive'})
    return options

def get_structures_file(filename, structures):
    """Stage the structures as a SinglefileData named ``filename``.

    Built straight from memory: staging via a fixed path in the system temp dir
    races between daemon workers, which can hand a job another job's structures
    (silently) or a half-written file.
    """
    payload = json.dumps(structures).encode('utf-8')
    return SinglefileData(io.BytesIO(payload), filename=filename)


uSimilarityCalculation = CalculationFactory('similarity')


class SimilarityWorkChain(BaseRestartWorkChain):
    """BaseRestartWorkChain to run uPETCalculation with automatic restarts."""
    _process_class = uSimilarityCalculation
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('input_structures', valid_type=List)
        spec.input('comparison_structures', valid_type=List)
        spec.input("code", valid_type=Code)
        spec.input('job_info', valid_type=Dict)

        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process
            ),
            cls.results
        )

        spec.exit_code(400,'ERROR_MAX_RESTARTS_EXCEEDED','Maximum number of restarts exceeded for uPETWorkChain.')

    def setup(self):
        """Initialize context before first calculation."""
        super().setup()

        input_structures = self.inputs.input_structures.get_list()
        comparison_structures = self.inputs.comparison_structures.get_list()
        job_info = self.inputs.job_info

        input_structures_file = get_structures_file(filename='input_structures.json', structures=input_structures)
        comparison_structures_file = get_structures_file(filename='comparison_structures.json', structures=comparison_structures)

        self.ctx.inputs = {
            'code': self.inputs.code,
            'file': {'input_structures_file': input_structures_file,
                     'comparison_structures_file': comparison_structures_file},
            'parameters': Dict(dict={'job_type': job_info['job_type'], 'cmdline_params': get_cmdline(job_info)}),
            'metadata': {'options': get_options(), 'label': 'Similarity analysis with uPET relaxations'}
        }
