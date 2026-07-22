import os
import json
import tempfile

from aiida.engine import BaseRestartWorkChain, while_
from aiida.orm import Str, Dict, Code, SinglefileData
from aiida.plugins import CalculationFactory

from uvsib.workflows import settings
from uvsib.codes.gnome.templates import build_template_pool


def get_options():
    """Return scheduler options for the GNoME code."""
    job_script = settings.configs['codes']['GNoME']['job_script']
    resources = {
        'num_machines': job_script['nodes'],
        'num_mpiprocs_per_machine': job_script['ntasks'],
        'num_cores_per_mpiproc': job_script['cpus'],
    }
    options = {
        'resources': resources,
        'max_wallclock_seconds': job_script['time'],
        'parser_name': 'gnome_parser',
    }
    if job_script['exclusive']:
        options.update({'custom_scheduler_commands': '#SBATCH --exclusive'})
    return options


def _templates_file(structures):
    """Stage the seed pool as a SinglefileData named templates.json."""
    path = os.path.join(tempfile.gettempdir(), "templates.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(structures, f)
    return SinglefileData(file=path)


def _get_cmdline(mode, target, ji):
    """Build the runner command line from a job_info dict."""
    cmdline = [
        f"--mode={mode}",
        f"--target={target}",
        "--templates=templates.json",
        f"--n_max={ji['n_max']}",
        f"--max_per_template={ji['max_per_template']}",
        f"--threshold={ji['threshold']}",
        f"--partial={ji['partial']}",
        f"--keep={ji['keep']}",
        f"--screen={ji['screen']}",
    ]
    if str(ji['screen']).lower() != "none":
        cmdline.extend([
            f"--model={ji['model_name']}",
            f"--model_path={ji['model_path']}",
            f"--task_name={ji['model_head']}",
            f"--device={ji['device']}",
        ])
    return cmdline


GNoMECalculation = CalculationFactory('gnome')


class _GNoMEWorkChainBase(BaseRestartWorkChain):
    """Shared restart machinery for the GNoME SAPS generator."""

    _process_class = GNoMECalculation
    _mode = None             # 'gen' or 'csp'
    _target_input = None     # 'chemical_system' or 'chemical_formula'

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(cls._target_input, valid_type=Str)
        spec.input("code", valid_type=Code)
        spec.input('job_info', valid_type=Dict)
        spec.expose_outputs(GNoMECalculation)
        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )
        spec.exit_code(
            400, 'ERROR_MAX_RESTARTS_EXCEEDED',
            message='Maximum number of restarts exceeded for GNoME workchain'
        )

    def setup(self):
        super().setup()
        target = self.inputs[self._target_input].value
        ji = self.inputs.job_info.get_dict()

        templates = build_template_pool(
            target, self._mode,
            k_donors=ji.get('k_donors', 3),
            ehull=ji.get('seed_ehull', 0.10),
            cap=ji.get('seed_cap', 60),
            use_icet=ji.get('icet_seeds', True),
            icet_max_size=ji.get('icet_max_size', 4),
            exp_seeds=ji.get('exp_seeds', True),
            exp_cap=ji.get('exp_cap', 10),
        )

        self.ctx.inputs = {
            'code': self.inputs.code,
            'file': {'templates_file': _templates_file(templates)},
            'parameters': Dict(dict={
                'cmdline_params': _get_cmdline(self._mode, target, ji),
                'staged_files': [("gnome_generate.py", "aiida.py"),   # the runner becomes aiida.py
                                 ("_saps.py", "_saps.py"),            # symmetry-aware partial substitution
                                 ("refine.py", "refine.py"),          # charge-neutral + primitive dedup (shared w/ MatterGen)
                                 ("_calculators.py", "_calculators.py"),  # ASE calculator factory for the GNN screen
                                 ],
                "retrieve_list": ["output.json"],
            }),
            'metadata': {
                'options': get_options(),
                'label': f'GNoME: {self._mode} for {target}',
            },
        }


class GNoMEBaseWorkChain(_GNoMEWorkChainBase):
    """De-novo generation toward a chemical system (parallel to mattergen.base)."""
    _mode = 'gen'
    _target_input = 'chemical_system'


class GNoMECSPWorkChain(_GNoMEWorkChainBase):
    """Composition-targeted generation (parallel to mattergen.csp)."""
    _mode = 'csp'
    _target_input = 'chemical_formula'
