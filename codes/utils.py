import os
import yaml
from pymatgen.core import Composition, Lattice, Structure
from mp_api.client import MPRester
from ase import Atoms
from itertools import product
from workflows import settings

def select_charge_neutral(composition_list):
    """
    Checks whether a composition can be charge neutral
    given possible oxidation states from the YAML dictionary.
    """
    yaml_file = os.path.join(settings.code_folder_path, 'files', 'oxidation_states_all.yaml')
    with open(yaml_file, 'r') as f:
        oxidation_states_dict = yaml.safe_load(f)

    neutral_composition_indices = []
    for i, c in enumerate(composition_list):
        composition = Composition(c)
        if composition.is_element:
            neutral_composition_indices.append(i)
            continue
        elements = list(composition.elements)
        amounts = [composition[el] for el in elements]

        oxidation_state_lists = []
        for el in elements:
            el_symbol = el.symbol
            ox_states = oxidation_states_dict.get(el_symbol)
            oxidation_state_lists.append(ox_states)
        for ox_state_combo in product(*oxidation_state_lists):
            total_charge = sum(ox * amt for ox, amt in zip(ox_state_combo, amounts))
            if total_charge == 0:
                neutral_composition_indices.append(i)
                break
    return neutral_composition_indices

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

def get_structures_from_mpdb(chemical_system):
    """Get structures from MPDB"""
    structures_list = []
    API_KEY = settings.api_key
    with MPRester(API_KEY) as mpr:
        entries = mpr.materials.summary.search(chemsys=chemical_system, energy_above_hull=(0,0.1), fields=["structure"])
    for entry in entries:
        structure = entry.structure
        structures_list.append(structure.as_dict())
    return structures_list
