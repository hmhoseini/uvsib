from aiida.plugins import DataFactory
from aiida.orm import Int, Str, Dict, Bool
from aiida_vasp.workchains.v2 import VaspBuilderUpdater
from uvsib.workflows import settings

def get_options(protocol_name):
    job_script = settings.configs["codes"]["VASP"]["job_script"][protocol_name]
    resources = {
        "num_machines": job_script["nodes"],
        "num_mpiprocs_per_machine": job_script["ntasks"]}
    if job_script["cpus"]:
        resources["num_cores_per_mpiproc"] = job_script["cpus"]
    options = {"resources": resources,
               "max_wallclock_seconds": job_script["time"],
               }
    if job_script["exclusive"]:
        options.update({"custom_scheduler_commands" : "#SBATCH --exclusive"})
    return options

def construct_vasp_builder(structure, protocol, potentials, vasp_code, parent_folder = None):
    """VASP Builder"""
    upd = VaspBuilderUpdater()
    upd.builder.structure = structure
    upd.builder.parameters = Dict(dict={"incar":protocol["incar"]})
    upd.builder.potential_family = Str(potentials["potential_family"])
    upd.builder.potential_mapping = Dict(dict=potentials["potential_mapping"])
    KpointsData = DataFactory("array.kpoints")
    kpoints = KpointsData()
    kpoints.set_cell_from_structure(structure)
    kpoints.set_kpoints_mesh_from_density(protocol["kpoint_distance"])
    upd.builder.kpoints = kpoints
    upd.builder.options = Dict(dict=get_options(protocol["name"]))
    upd.builder.clean_workdir = Bool(False)
    n_iter = 2
    retrieve_temporary_list = []
    if "HSE" in protocol["name"]:
        n_iter = 1
        retrieve_temporary_list.append("vasprun.xml")
    parser_settings = {"add_structure": True,
                        "add_energies": True
                      }
    upd.builder.settings = Dict(dict={"parser_settings": parser_settings,
                                  "retrieve_temporary_list": ['vasprun.xml']}
    )
    upd.builder.max_iterations = Int(n_iter)
    upd.builder.verbose = Bool(True)
    if parent_folder:
        upd.builder.restart_folder = parent_folder
    upd.builder.code = vasp_code
    return upd.builder
