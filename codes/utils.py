import os
import json
from ase.constraints import FixAtoms
from pymatgen.core import Lattice, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from mp_api.client import MPRester
from ase import Atoms
from uvsib.workflows import settings


def get_cmdline(job_info):
    """Construct command line"""
    cmdline = []
    cmdline.append(f"--ML_model={job_info['ML_model']}")
    cmdline.append(f"--model={job_info['model_name']}")
    cmdline.append(f"--model_path={job_info['model_path']}")
    cmdline.append(f"--task_name={job_info['model_head']}")
    cmdline.append(f"--device={job_info['device']}")

    job_type = job_info['job_type']

    if job_type == 'relax':
        cmdline.extend([
            f"--fmax={job_info['fmax']}",
            f"--max_steps={job_info['max_steps']}"]
        )
    elif job_type == 'hopping':
        cmdline.extend([
            f"--mh_steps={job_info['mh_steps']}"
        ])
    elif job_type == 'face_build':
        cmdline.extend([
            f"--fmax={job_info['fmax']}",
            f"--max_steps={job_info['max_steps']}",
            f"--max_miller_idx={job_info['max_miller_idx']}",
            f"--max_num_surf={job_info['max_num_surf']}"]
        )
    elif job_type == 'adsorbates':
        cmdline.extend([
            f"--slab_energy={job_info['slab_energy']}",
            f"--fmax={job_info['fmax']}",
            f"--max_steps={job_info['max_steps']}",
            f"--reaction={job_info['reaction']}",
            f"--pathway={job_info['pathway']}",
            f"--no-validate"]
        )
    elif job_type == 'akmc':
        cmdline.extend([
            f"--fmax={job_info['fmax']}",
            f"--dimer_fmax={job_info['dimer_fmax']}",
            f"--relax_steps={job_info['relax_steps']}",
            f"--dimer_steps={job_info['dimer_steps']}",
            f"--searches_per_minimum={job_info['searches_per_minimum']}",
            f"--temperature={job_info['temperature']}",
            f"--prefactor={job_info['prefactor']}",
            f"--dimer_displacement={job_info['dimer_displacement']}",
            f"--product_displacement={job_info['product_displacement']}",
            f"--maxstep={job_info['maxstep']}",
            f"--seed={job_info['seed']}"]
        )
    elif job_type == 'nano_particles':
        cmdline.extend([
            '--fmax={}'.format(job_info['fmax']),
            '--max_steps={}'.format(job_info['max_steps']),
            '--elements={}'.format(job_info['elements']),
            '--min_natoms={}'.format(job_info['particles_range'].split('-')[0]),
            '--max_natoms={}'.format(job_info['particles_range'].split('-')[1]),
            '--generator={}'.format(job_info['generator'])]
        )
    # print('DBG codes/utils: ', cmdline)
    return cmdline

def get_element_entries(chemsys_list, functional):
    if functional == "GGA":
        file = os.path.join(settings.uvsib_directory, 'codes', 'files', 'gga_ggau_entries.json')
    else:
        file = os.path.join(settings.uvsib_directory, 'codes', 'files', 'r2scan_entries.json')
    with open(file, "r") as f:
        entries = json.load(f)
    output_entries = []
    for entry in entries:
        cse = ComputedStructureEntry.from_dict(entry["entries"][functional])
        if cse.composition.chemical_system in chemsys_list:
            output_entries.append(cse)
    return output_entries

def pmg_to_ase(structure):
    """
    Convert a pymatgen Structure to an ASE Atoms object
    """
    symbols = [str(site.specie) for site in structure.sites]
    positions = structure.cart_coords
    cell = structure.lattice.matrix
    pbc = [True, True, True]
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
    return atoms

def ase_to_pmg(atoms):
    """
    Convert an ASE Atoms object to a pymatgen Structure 
    """
    lattice = atoms.cell.array.tolist()
    symbols = atoms.get_chemical_symbols()
    frac_coords = atoms.get_scaled_positions().tolist()
    lattice_obj = Lattice(lattice)

    selective_dynamics = [[True, True, True] for _ in atoms]

    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            for idx in constraint.index:
                selective_dynamics[idx] = [False, False, False]
    return Structure(
        lattice=Lattice(lattice),
        species=symbols,
        coords=frac_coords,
        coords_are_cartesian=False,
        site_properties={"selective_dynamics": selective_dynamics},
    )

def get_structures_from_mpdb_by_composition(chemical_formula, e_hull=0.1):
    """Get stable structures from the MPDB.

    Parameters
    ----------
    chemical_formula : str
    e_hull : float
        Maximum energy above hull in eV/atom.
    """
    stable_structures = []
    exp_structures = []
    api_key = settings.api_key

    search_kwargs = {
        "formula": chemical_formula,
        "energy_above_hull": (0, e_hull),
        "fields": ["structure", "energy_above_hull", "theoretical"],
    }

    with MPRester(api_key) as mpr:
        summaries = mpr.materials.summary.search(**search_kwargs)

    if not summaries:
        return [], []

    for summary in summaries:
        if summary.theoretical:
            stable_structures.append(summary.structure.as_dict())
        else:
            exp_structures.append(summary.structure.as_dict())

    return stable_structures, exp_structures

#def get_mp_element_structures(elements):
#    """Stable elemental ground-state structures from the Materials Project.
#
#    Returns ``{element_symbol: structure_dict}`` with the lowest-e_above_hull
#    elemental polymorph per element -- the seed crystals to be relaxed with the
#    project MLIP so the hull's elemental endpoints are on-method.
#    """
#    out = {}
#    with MPRester(settings.api_key) as mpr:
#        for el in elements:
#            try:
#                docs = mpr.materials.summary.search(
#                    chemsys=el, fields=["structure", "energy_above_hull"])
#            except Exception:
#                continue
#            docs = [d for d in docs if getattr(d, "energy_above_hull", None) is not None]
#            if not docs:
#                continue
#            best = min(docs, key=lambda d: d.energy_above_hull)
#            out[el] = best.structure.as_dict()
#    return out

#def get_entries_from_mpdb(chemical_formula, run_type, ehull):
#    """Get structures entry from the MPDB
#       run_type: GGA or r2SCAN
#    """
#    entries = []
#    api_key = settings.api_key
#
#    with MPRester(api_key) as mpr:
#        material_data = mpr.materials.summary.search(
#                formula=chemical_formula,
#                fields=["material_id", "energy_above_hull", "task_ids"]
#        )
#
#        for summary in material_data:
#            if summary.energy_above_hull is None or summary.energy_above_hull > ehull:
#                continue
#
#            tasks = mpr.tasks.search(task_ids=summary.task_ids)
#
#            task = next((t for t in tasks if t.run_type == run_type), None)
#            if task and task.structure_entry:
#                entries.append(task.structure_entry)
#    return entries

#def get_energy_per_atom(functional):
#    elements = ["H","He",
#                "Li","Be","B","C","N","O","F","Ne",
#                "Na","Mg","Al","Si","P","S","Cl","Ar",
#                "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
#                "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
#                "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta",
#                "W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Ac","Th","Pa","U","Np","Pu"]
#    with MPRester(settings.api_key) as mpr:
#        entries = mpr.materials.thermo.search(
#                chemsys=elements,
#                thermo_types=[functional],
#                energy_above_hull=(0,0),
#                fields=["entries"]
#        )
#    to_dump = []
#    for ents in entries:
#        to_dump.append(ents.dict())
#    return to_dump
