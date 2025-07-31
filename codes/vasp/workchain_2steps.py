import os
from random import uniform
import yaml
from aiida.plugins import WorkflowFactory, DataFactory
from aiida.engine import WorkChain
from aiida.orm import Int, Str, List, Dict, Code, Bool, SinglefileData, load_group
import workflows.settings as settings

StructureData = DataFactory('structure')

def get_options():
    job_script = settings.job_script
    resources = {
        'num_machines': job_script['geopt']['nodes'],
        'num_mpiprocs_per_machine': job_script['geopt']['ntasks']}
    if job_script['geopt']['ncpu']:
        resources['num_cores_per_mpiproc'] = job_script['geopt']['ncpu']
    options = {'resources': resources,
               'max_wallclock_seconds': job_script['geopt']['time'],
               }
    if job_script['geopt']['exclusive']:
        options.update({'custom_scheduler_commands' : '#SBATCH --exclusive'})
    return options

def construct_builder(structure, protocol, potential_family, potential_mapping):
    Workflow = WorkflowFactory('vasp.vasp')
    builder = Workflow.get_builder()
    builder.structure = structure
    builder.parameters = Dict(dict={'incar':protocol['incar']})
    builder.potential_family = Str(potential_family)
    builder.potential_mapping = Dict(dict=potential_mapping)
    KpointsData = DataFactory("array.kpoints")
    kpoints = KpointsData()
    kpoints.set_cell_from_structure(structure)
    kpoints.set_kpoints_mesh_from_density(protocol['kpoint_distance'])
    builder.kpoints = kpoints
    builder.code = Code.get_from_string(settings.configs['aiida_settings']['DFT_code_string'])

    parser_settings = {'add_structure': True,
                        'add_trajectory': True,
                        'add_energies': True,
                      }
    builder.settings = Dict(dict={'parser_settings': parser_settings})
#    builder.settings = Dict(dict={'CHECK_IONIC_CONVERGENCE': False})
    builder.options = Dict(dict=get_options())
    builder.clean_workdir = Bool(False)
    builder.max_iterations = Int(2)
    builder.verbose = Bool(True)
    builder.metadata['label'] = protocol['name']
    return builder

class R2SCANRelaxWorkChain(WorkChain):
    """
    A WorkChain that performs a two-step VASP relaxation:
    1. PBEsol optimization
    2. r2SCAN optimization using WAVECAR, CHGCAR, and CONTCAR from step 1
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData, help='Input structure')

        spec.outline(
            cls.setup,
            cls.run_pbesol,
            cls.run_r2scan,
        )
        spec.output('relaxed_structure', valid_type=StructureData)
        spec.output('r2scan_output_parameters', valid_type=Dict)

    def setup(self):
        with open(os.path.join(settings.VASP_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.vasp_protocol = yaml.safe_load(fhandle)
        with open(os.path.join(settings.VASP_input_files_path,'potential_mapping.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.potential = yaml.safe_load(fhandle)

    def run_pbesol(self):
        structure = self.inputs.structure
        protocol = self.ctx.vasp_protocol['pbesol']
        protocol['name'] = 'pbesol'
        potential_family = self.ctx.potential['potential_family']
        potential_mapping = self.ctx.potential['potential_mapping']
        builder = construct_builder(structure, protocol, potential_family, potential_mapping)
        future = self.submit(builder)
        self.to_context(**{'pbesol': future})

    def inspect_calculation(self):
        if not self.ctx['bulk'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        a_node = self.ctx['bulk']
        results_step1_group.add_nodes(a_node)


    def run_scan(self):
        structure = self.ctx['pbesol'].outputs['structure']
        pbesol_remote = self.ctx['pbesol'].outputs.remote_folder

        protocol = self.ctx.vasp_protocol['r2scan']
        protocol['name'] = 'r2scan'
        potential_family = self.ctx.potential['potential_family']
        potential_mapping = self.ctx.potential['potential_mapping']
        builder = construct_builder(structure, protocol, potential_family, potential_mapping)
        symlink_list = [
            (pbesol_remote, 'WAVECAR', 'WAVECAR'),
            (pbesol_remote, 'CHGCAR', 'CHGCAR'),
            (pbesol_remote, 'CONTCAR', 'POSCAR'),
        ]
        builder.remote_symlink_list = symlink_list

