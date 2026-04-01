from aiida.orm import Str, Dict, Code
from aiida.engine import BaseRestartWorkChain, while_
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
        'parser_name': 'upet_parser'
    }
    if job_script['exclusive']:
        options.update({'custom_scheduler_commands' : '#SBATCH --exclusive'})
    return options


uPETCalculation = CalculationFactory('nano_particles')


class NanoParticleWorkChain(BaseRestartWorkChain):
    """run NanoParticle generator"""
    _process_class = uPETCalculation
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('elements', valid_type=Str)
        spec.input('particles_range', valid_tpye=Str)
        spec.input('generator', valid_tpye=Str)
        spec.input('code', valid_type=Code)
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
            message='Maximum number of restarts exceeded for NanoParticle generator.'
        )

    def setup(self):
        """Initialize context before first calculation."""
        super().setup()
        elements = self.inputs.elements
        particles_range = self.inputs.particles_range
        job_info = self.inputs.job_info
        self.ctx.inputs = {
            elements: self.inputs.elements,
            'code': self.inputs.code,
            'particles_range': particles_range,
            'generator': self.inputs.generator,
            'parameters': Dict(dict={'job_type': job_info['job_type'], 'cmdline_params': get_cmdline(job_info)}),
            'metadata': {'options': get_options(), 'label': 'NanoParticle Generator'}
        }
