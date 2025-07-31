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

def construct_builder(structure, protocol):
    Workflow = WorkflowFactory('vasp.vasp')
    builder = Workflow.get_builder()
    builder.structure = structure
    builder.parameters = Dict(dict={'incar':protocol['incar']})
    builder.potential_family = Str(protocol['potential_family'])
    builder.potential_mapping = Dict(dict=protocol['potential_mapping'])
    KpointsData = DataFactory("array.kpoints")
    kpoints = KpointsData()
    kpoints.set_cell_from_structure(structure)
    kpoints.set_kpoints_mesh_from_density(protocol['kpoint_distance'])
    builder.kpoints = kpoints
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
    return builder

class VASPPBERelaxWorkChain(WorkChain):
    """
    A WorkChain that performs geometry and volume optimization
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('protocol', valid_type=Dict)

        spe.outlin(
            cls.run_vasp,
            cls.inspect_vasp,
            cls.results
        )
        spec.output('relaxed_structure', valid_type=StructureData)
        spec.output('output_parameters', valid_type=Dict)
        spec.output('remote_folder', valid_type=RemoteData)

    def run_vasp(self):

        builder = construct_builder(self.inputs.structure,
                                    self.inputs.protocol)
        builder.code = self.inputs.vasp_code

        future = self.submit(builder)
        self.to_context(**{'vasppberelax': future})

    def inspect_vasp(self):
        if not self.ctx['vasppberelax'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def results(self):
        calculation = self.ctx['vasppberelax']
        self.out('relaxed_structure', calculation.outputs.structure)
        self.out('output_parameters', calculation.outputs.parameters)
        self.out('remote_folder', calculation.outputs.remote_folder)

class VASPSCANRelaxWorkChain(WorkChain):
    """
    A WorkChain that performs geometry and volume optimization
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('protocol', valid_type=Dict)
        spec.input('vasp_code', valid_type=Code)
        spec.input('parent_folder', valid_type=RemoteData)

        spec.outline(
            cls.run_vasp,
            cls.inspect_vasp,
            cls.results
        )
        spec.output('relaxed_structure', valid_type=StructureData)
        spec.output('output_parameters', valid_type=Dict)
        spec.outpur('remote_folder', valid_type=RemoteData)

    def run_vasp(self):
        builder = construct_builder(self.inputs.structure,
                                    self.inputs.protocol)
        symlink_list = [
            (pbesol_remote, 'WAVECAR', 'WAVECAR'),
            (pbesol_remote, 'CHGCAR', 'CHGCAR'),
            (pbesol_remote, 'CONTCAR', 'POSCAR'),
        ]
        builder.remote_symlink_list = symlink_list

    def results(self):
        result_wc = self.ctx.base_wc
        self.out('relaxed_structure', result_wc.outputs.structure)
        self.out('output_parameters', result_wc.outputs.parameters)
        self.out('remote_folder', result_wc.outputs.remote_folder)

class VASPHSESPWorkChain(WorkChain):
    """
    A WorkChain that performs single point hybrid DFT
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('protocol', valid_type=Dict)
        spec.input('vasp_code', valid_type=Code)
        spec.input('parent_folder', valid_type=RemoteData)

        spec.outline(
            cls.run_vasp,
            cls.inspect_vasp,
            cls.results
        )
        spec.output('relaxed_structure', valid_type=StructureData)
        spec.output('output_parameters', valid_type=Dict)
        spec.outpur('remote_folder', valid_type=RemoteData)

    def run_vasp(self):
        builder = construct_builder(self.inputs.structure,
                                    self.inputs.protocol)
        symlink_list = [
            (pbesol_remote, 'WAVECAR', 'WAVECAR'),
            (pbesol_remote, 'CHGCAR', 'CHGCAR'),
            (pbesol_remote, 'CONTCAR', 'POSCAR'),
        ]
        builder.remote_symlink_list = symlink_list

    def results(self):
        result_wc = self.ctx.base_wc
        self.out('relaxed_structure', result_wc.outputs.structure)
        self.out('output_parameters', result_wc.outputs.parameters)
        self.out('remote_folder', result_wc.outputs.remote_folder)

