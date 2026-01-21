import os
import json
from pymatgen.core import Lattice, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from mp_api.client import MPRester
from ase import Atoms
from uvsib.workflows import settings

def get_cmdline(job_info):
    """Construct command line"""

    cmdline = []

    cmdline.append(f"--ML_model={job_info['ML_model']}")

    model_name = job_info.get("model_name")
    model_path = job_info.get("model_path")

    if model_name:
        cmdline.append(f"--model={model_name}")
    if model_path:
        cmdline.append(f"--model_path={model_path}")

    cmdline.append(f"--device={job_info['device']}")

    job_type = job_info['job_type']

    if job_type == 'relax':
        cmdline.extend([
            f"--fmax={job_info['fmax']}",
            f"--max_steps={job_info['max_steps']}"]
        )
    elif job_type == 'facebuild':
        cmdline.extend([
            f"--bulk_energy={job_info['bulk_energy']}",
            f"--fmax={job_info['fmax']}",
            f"--max_steps={job_info['max_steps']}",
            f"--max_miller_idx={job_info['max_miller_idx']}",
            f"--percentage_to_select={job_info['percentage_to_select']}"]
        )
    elif job_type == 'adsorbates':
        cmdline.extend([
            f"--slab_energy={job_info['slab_energy']}",
            f"--fmax={job_info['fmax']}",
            f"--max_steps={job_info['max_steps']}",
            f"--reaction={job_info['reaction']}"]
        )
    return cmdline

def get_element_entries(chemsys_list, functional):
    if functional == "GGA":
        file = os.path.join(settings.uvsib_directory, 'codes', 'files', 'gga_ggau_entries.json')
    else:
        file = os.path.join(settings.uvsib_directory, 'codes', 'files', 'r2scan_entries.json')
    with open(file, "r") as f:
        entries = json.loads(f.read())
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
    Convert an ASE Atoms object to a pymatgen Structure dictionary
    """
    lattice = atoms.cell.array.tolist()
    symbols = atoms.get_chemical_symbols()
    frac_coords = atoms.get_scaled_positions().tolist()
    lattice_obj = Lattice(lattice)
    structure = Structure(lattice_obj, symbols, frac_coords, coords_are_cartesian=False)
    return structure.as_dict()

def get_structures_from_mpdb_by_composition(chemical_formula, e_hull):
    """Get the most stable structures from the MPDB"""
    stable_structures = []
    api_key = settings.api_key
    with MPRester(api_key) as mpr:
        summaries = mpr.materials.summary.search(
                formula=chemical_formula,
                fields=["structure", "energy_above_hull"]
        )
    if not summaries:
        return False
    for summary in summaries:
        if summary.energy_above_hull <= e_hull:
            stable_structures.append(summary.structure.as_dict())
    return stable_structures

def get_entries_from_mpdb(chemical_formula, run_type, ehull):
    """Get structures entry from the MPDB
       run_type: GGA or r2SCAN
    """
    entries = []
    api_key = settings.api_key

    with MPRester(api_key) as mpr:
        material_data = mpr.materials.summary.search(
                formula=chemical_formula,
                fields=["material_id", "energy_above_hull", "task_ids"]
        )

        for summary in material_data:
            if summary.energy_above_hull is None or summary.energy_above_hull > ehull:
                continue

            tasks = mpr.tasks.search(task_ids=summary.task_ids)

            task = next((t for t in tasks if t.run_type == run_type), None)
            if task and task.structure_entry:
                entries.append(task.structure_entry)
    return entries

def get_energy_per_atom(functional):
    elements = ["H","He",
                "Li","Be","B","C","N","O","F","Ne",
                "Na","Mg","Al","Si","P","S","Cl","Ar",
                "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
                "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
                "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi",
                          "Ac","Th","Pa","U","Np","Pu"]
    with MPRester(settings.api_key) as mpr:
        entries = mpr.materials.thermo.search(
                chemsys=elements,
                thermo_types=[functional],
                energy_above_hull=(0,0),
                fields=["entries"]
        )
    to_dump = []
    for ents in entries:
        to_dump.append(ents.dict())
    return to_dump
