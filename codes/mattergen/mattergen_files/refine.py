import json
import numpy as np
from pymatgen.core.structure import Lattice, Structure
from ase.io import read
from itertools import product

oxidation_states_dict = {
 'H': [-1, 1],
 'Li': [-1, 1],
 'Be': [1, 2],
 'B': [-5, -1, 1, 2, 3],
 'C': [-4, -3, -2, -1, 1, 2, 3, 4],
 'N': [-3, -2, -1, 1, 2, 3, 4, 5],
 'O': [-2, -1, 1, 2],
 'F': [-1],
 'Na': [-1, 1],
 'Mg': [1, 2],
 'Al': [-2, -1, 1, 2, 3],
 'Si': [-4, -3, -2, -1, 1, 2, 3, 4],
 'P': [-3, -2, -1, 1, 2, 3, 4, 5],
 'S': [-2, -1, 1, 2, 3, 4, 5, 6],
 'Cl': [-1, 1, 2, 3, 4, 5, 6, 7],
 'K': [-1, 1],
 'Ca': [1, 2],
 'Sc': [1, 2, 3],
 'Ti': [-2, -1, 1, 2, 3, 4],
 'V': [-3, -1, 1, 2, 3, 4, 5],
 'Cr': [-4, -2, -1, 1, 2, 3, 4, 5, 6],
 'Mn': [-3, -2, -1, 1, 2, 3, 4, 5, 6, 7],
 'Fe': [-4, -2, -1, 1, 2, 3, 4, 5, 6, 7],
 'Co': [-3, -1, 1, 2, 3, 4, 5],
 'Ni': [-2, -1, 1, 2, 3, 4],
 'Cu': [-2, 1, 2, 3, 4],
 'Zn': [-2, 1, 2],
 'Ga': [-5, -4, -3, -2, -1, 1, 2, 3],
 'Ge': [-4, -3, -2, -1, 1, 2, 3, 4],
 'As': [-3, -2, -1, 1, 2, 3, 4, 5],
 'Se': [-2, -1, 1, 2, 3, 4, 5, 6],
 'Br': [-1, 1, 2, 3, 4, 5, 7],
 'Rb': [-1, 1],
 'Sr': [1, 2],
 'Y': [1, 2, 3],
 'Zr': [-2, 1, 2, 3, 4],
 'Nb': [-3, -1, 1, 2, 3, 4, 5],
 'Mo': [-4, -2, -1, 1, 2, 3, 4, 5, 6],
 'Tc': [-1, 1, 2, 3, 4, 5, 6, 7],
 'Ru': [-2, 1, 2, 3, 4, 5, 6, 7, 8],
 'Rh': [-3, -1, 1, 2, 3, 4, 5, 6, 7],
 'Pd': [1, 2, 3, 4, 5],
 'Ag': [-2, -1, 1, 2, 3],
 'Cd': [-2, 1, 2],
 'In': [-5, -2, -1, 1, 2, 3],
 'Sn': [-4, -3, -3, -2, 1, 2, 3, 4],
 'Sb': [-3, -2, -1, 1, 2, 3, 4, 5],
 'Te': [-2, -1, 1, 2, 3, 4, 5, 6],
 'I': [-1, 1, 2, 3, 4, 5, 6, 7],
 'Cs': [-1, 1],
 'Ba': [1, 2],
 'Hf': [-2, 1, 2, 3, 4],
 'Ta': [-3, -1, 1, 2, 3, 4, 5],
 'W': [-4, -2, -1, 1, 2, 3, 4, 5, 6],
 'Re': [-3, -1, 1, 2, 3, 4, 5, 6, 7],
 'Os': [-4, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8],
 'Ir': [-3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
 'Pt': [-3, -2, -1, 1, 2, 3, 4, 5, 6],
 'Au': [-3, -2, -1, 1, 2, 3, 5],
 'Hg': [-2, 1, 2],
 'Tl': [-5, -2, -1, 1, 2, 3],
 'Pb': [-4, -2, -1, 1, 2, 3, 4],
 'Bi': [-3, -2, -1, 1, 2, 3, 4, 5],
 'Po': [-2, 2, 4, 5, 6],
 'At': [-1, 1, 3, 5, 7]
}

def ase_to_pmg(atoms):
    """
    Convert an ASE Atoms object to a pymatgen Structure dictionary
    """
    lattice = atoms.cell.array.tolist()
    symbols = atoms.get_chemical_symbols()
    frac_coords = atoms.get_scaled_positions().tolist()
    lattice_obj = Lattice(lattice)
    return Structure(lattice_obj,
                          symbols,
                          frac_coords,
                          coords_are_cartesian=False
    )

def select_charge_neutral(structures):
    """
   Filter structures that are charge neutral based on oxidation states from context
    """
    neutral_struct = []

    for struct_dict in structures:
        composition = Structure.from_dict(struct_dict).composition
        if composition.is_element:
            neutral_struct.append(struct_dict)
            continue

        elements = list(composition.elements)
        amounts = [composition[el] for el in elements]

        oxidation_state_lists = []
        for el in elements:
            ox_states = oxidation_states_dict.get(el.symbol)
            oxidation_state_lists.append(ox_states)
        for ox_state_combo in product(*oxidation_state_lists):
            total_charge = sum(ox * amt for ox, amt in zip(ox_state_combo, amounts))
            if total_charge == 0:
                neutral_struct.append(struct_dict)
                break
    return neutral_struct

def refine_and_filter_structures(structures):
    """
    Symmetrize a list of structures and remove duplicates based on structural similarity
    """
    unique_structures = []

    for s_dict in structures:
        structure = Structure.from_dict(s_dict)

        lattice_matrix = structure.lattice.matrix.copy()
        lattice_matrix[np.abs(lattice_matrix) < 0.1] = 0.0

        new_lattice = Lattice(lattice_matrix)
        a, b, c = new_lattice.abc
        alpha, beta, gamma = (round(ang) for ang in new_lattice.angles)

        try:
            final_lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        except ValueError:
            final_lattice = new_lattice

        refined_struct = Structure(
            final_lattice,
            structure.species,
            structure.frac_coords,
            coords_are_cartesian=False)

        if not any(refined_struct.matches(existing) for existing in unique_structures):
            unique_structures.append(refined_struct)

    return [s.as_dict() for s in unique_structures]

if __name__ == "__main__":

    ase_structures = read('generated_crystals.extxyz', index=':')

    pmg_structures = []
    for ase_struct in ase_structures:
        pmg_struct = ase_to_pmg(ase_struct)
        pmg_structures.append(pmg_struct.as_dict())

    neutral_structures = select_charge_neutral(pmg_structures)

    refined_structures = refine_and_filter_structures(neutral_structures)

    with open('output.json', 'w') as f:
        json.dump(refined_structures, f)
