from pymatgen.core import Composition
from aiida.engine import BaseRestartWorkChain, while_
from aiida.orm import Str, Dict, Code
from aiida.plugins import CalculationFactory
from uvsib.workflows import settings

def get_options():
    """Return scheduler options"""
    job_script = settings.configs['codes']['MatterGen']['job_script']
    resources = {
        'num_machines': job_script['nodes'],
        'num_mpiprocs_per_machine': job_script['ntasks'],
        'num_cores_per_mpiproc': job_script['cpus'],
    }
    options = {
        'resources': resources,
        'max_wallclock_seconds': job_script['time'],
        'parser_name': 'mattergen_parser',
        'append_text': 'python aiida.py'
    }
    if job_script['exclusive']:
        options.update({'custom_scheduler_commands' : '#SBATCH --exclusive'})
#    if job_script['gpu']:
#        options.update({'custom_scheduler_commands' : '#SBATCH '})

    return options

def get_cmdline(chemical_system, job_info, diffusion_guidance_factor = 2.0):
    """Construct MatterGen command line"""

    return [
        "$RESULTS_PATH",
        f"--pretrained-name={job_info['model_name']}",
        f"--batch_size={job_info['batch_size']}",
        f"--num_batches={job_info['num_batches']}",
        f"--properties_to_condition_on={{'energy_above_hull': {job_info['energy_above_hull']}, 'chemical_system': '{chemical_system}'}}",
        f"--diffusion_guidance_factor={diffusion_guidance_factor}",
        "--record-trajectories=False"
    ]

def get_cmdline_csp(chemical_formula, job_info):
    """Construct MatterGen CSP command line"""
    comp = Composition(chemical_formula)
    el_amt = comp.get_el_amt_dict()
    natom = comp.num_atoms
    coef = 20//natom
    inner = ", ".join(f"\'{k}\': {int(coef*v)}" for k, v in el_amt.items())
    target_str = f"{{{inner}}}"
    return [
        "$RESULTS_PATH",
        "--sampling-config-name=csp",
        f"--model_path={job_info['model_path']}",
        f"--batch_size={job_info['batch_size']}",
        f"--num_batches={job_info['num_batches']}",
        f"--target_compositions=[{target_str}]",
        "--record-trajectories=False",
    ]

MatterGenCalculation = CalculationFactory('mattergen')

class MatterGenBaseWorkChain(BaseRestartWorkChain):
    """BaseRestartWorkChain to run MatterGenCalculation with automatic restarts"""

    _process_class = MatterGenCalculation

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # Declare the inputs needed for this workchain:
        spec.input('chemical_system', valid_type=Str)
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
            message='Maximum number of restarts exceeded for MatterGenBaseWorkChain'
        )

    def setup(self):
        """Initialize context before first calculation."""
        super().setup()

        chemical_system = self.inputs.chemical_system.value
        job_info = self.inputs.job_info
        cmdline = get_cmdline(chemical_system, job_info)

        self.ctx.inputs = {
            'code': self.inputs.code,
            'parameters': Dict(dict={'cmdline_params': cmdline}),
            'metadata': {
                'options': get_options(),
                'label': 'MatterGen calculation'
            }
        }

class MatterGenCSPWorkChain(BaseRestartWorkChain):
    """BaseRestartWorkChain to run MatterGenCalculation with automatic restarts"""

    _process_class = MatterGenCalculation

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # Declare the inputs needed for this workchain:
        spec.input('chemical_formula', valid_type=Str)
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
            message='Maximum number of restarts exceeded for MatterGenBaseWorkChain'
        )

    def setup(self):
        """Initialize context before first calculation."""
        super().setup()

        chemical_formula = self.inputs.chemical_formula.value
        job_info = self.inputs.job_info
        cmdline = get_cmdline_csp(chemical_formula, job_info)

        self.ctx.inputs = {
            'code': self.inputs.code,
            'parameters': Dict(dict={'cmdline_params': cmdline}),
            'metadata': {
                'options': get_options(),
                'label': 'MatterGen CSP calculation'
            }
        }
