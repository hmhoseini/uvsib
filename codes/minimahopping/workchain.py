from aiida.engine import BaseRestartWorkChain, while_
from aiida.orm import Dict, Code
from aiida.plugins import DataFactory, CalculationFactory
from uvsib.workflows import settings

StructureData = DataFactory('core.structure')

def get_options():
    """Return scheduler options"""
    job_script = settings.configs['codes']['MinimaHopping']['job_script']
    resources = {
        'num_machines': job_script['nodes'],
        'num_mpiprocs_per_machine': job_script['ntasks'],
        'num_cores_per_mpiproc': job_script['cpus'],
    }
    options = {
        'resources': resources,
        'max_wallclock_seconds': job_script['time'],
        'parser_name': 'minimahopping_parser'
    }
    if job_script['exclusive']:
        options.update({'custom_scheduler_commands' : '#SBATCH --exclusive'})
    return options

def get_cmdline(job_info):
    """Construct MinimaHopping command line"""
    cmdline = [
        f"--model={job_info['model']}",
        f"--model_path={job_info['model_path']}",
        f"--mh_steps={job_info['mh_steps']}",
        f"--device={job_info['device']}",
    ]
    return cmdline

MinimaHoppingCalculation = CalculationFactory('mh')

class MinimaHoppingWorkChain(BaseRestartWorkChain):
    """RestartWorkChain to run MinimaHoppingCalculation"""

    _process_class = MinimaHoppingCalculation

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('structure', valid_type=StructureData)
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
            message='Maximum number of restarts exceeded for MinimaHoppingWorkChain.'
        )

    def setup(self):
        """Initialize context before first calculation."""
        super().setup()

        structure = self.inputs.structure
        job_info = self.inputs.job_info

        self.ctx.inputs = {
            'code': self.inputs.code,
            'structure': structure,
            'parameters': Dict(dict={
                'cmdline_params': get_cmdline(job_info)
            }),
            'metadata': {
                'options': get_options(),
                'label': 'MinmaHopping calculation'
            }
        }
