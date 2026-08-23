from pymatgen.core import Composition
from aiida.engine import BaseRestartWorkChain, while_
from aiida.orm import Str, Dict, Code
from aiida.plugins import CalculationFactory
from uvsib.workflows import settings


def get_options():
    """Return scheduler options for the DiffCSP_CSP code."""
    job_script = settings.configs['codes']['DiffCSP_CSP']['job_script']
    resources = {
        'num_machines': job_script['nodes'],
        'num_mpiprocs_per_machine': job_script['ntasks'],
        'num_cores_per_mpiproc': job_script['cpus'],
    }
    options = {
        'resources': resources,
        'max_wallclock_seconds': job_script['time'],
        'parser_name': 'diffcsp_parser',
    }
    if job_script['exclusive']:
        options.update({'custom_scheduler_commands': '#SBATCH --exclusive'})
    return options


def get_cmdline_csp(chemical_formula, job_info):
    """Construct the DiffCSP CSP runner (aiida.py) command line.

    DiffCSP's ``scripts/sample.py`` is genuinely composition-conditioned (it
    fixes the atom types to ``chemical_formula`` and diffuses only the lattice
    and coordinates), unlike ``scripts/generation.py``'s unconditioned
    ab-initio sampling. See ``codes/files/diffcsp_csp.py``.
    """
    comp = Composition(chemical_formula)
    el_amt = comp.get_el_amt_dict()
    natom = comp.num_atoms

    if natom > 20:
        raise ValueError(
            f"DiffCSP CSP formula scaling supports target compositions up to "
            f"20 atoms, but {chemical_formula} has {natom} atoms."
        )

    coef = 20 // natom
    scaled_formula = "".join(f"{el}{int(coef * amt)}" for el, amt in el_amt.items())

    return [
        f"--repo_path={job_info['repo_path']}",
        f"--model_path={job_info['model_path']}",
        f"--chemical_formula={scaled_formula}",
        f"--num_evals={job_info.get('num_evals', 20)}",
        f"--batch_size={job_info.get('batch_size', 20)}",
        f"--step_lr={job_info.get('step_lr', 1e-5)}",
    ]


DiffCSPCalculation = CalculationFactory('diffcsp')


class DiffCSPCSPWorkChain(BaseRestartWorkChain):
    """BaseRestartWorkChain to run DiffCSPCalculation with automatic restarts.

    Composition-targeted structure prediction (parallel to mattergen.csp and
    gnome.csp).
    """

    _process_class = DiffCSPCalculation

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('chemical_formula', valid_type=Str)
        spec.input("code", valid_type=Code)
        spec.input('job_info', valid_type=Dict)

        spec.expose_outputs(DiffCSPCalculation)

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
            message='Maximum number of restarts exceeded for DiffCSPCSPWorkChain'
        )

    def setup(self):
        """Initialize context before first calculation."""
        super().setup()

        chemical_formula = self.inputs.chemical_formula.value
        job_info = self.inputs.job_info.get_dict()
        cmdline = get_cmdline_csp(chemical_formula, job_info)

        self.ctx.inputs = {
            'code': self.inputs.code,
            'parameters': Dict(dict={'cmdline_params': cmdline}),
            'metadata': {
                'options': get_options(),
                'label': 'DiffCSP: CSP for {}'.format(chemical_formula)
            }
        }
