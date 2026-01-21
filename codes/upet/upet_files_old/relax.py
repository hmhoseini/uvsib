import json
import argparse
from pymatgen.core import Lattice, Structure
from ase import Atoms
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.filters import FrechetCellFilter
from upet.calculator import UPETCalculator

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

def pmg_to_ase(pmg_structure):
    """
    Convert a pymatgen Structure to an ASE Atoms object.
    """
    scaled_positions = pmg_structure.frac_coords
    symbols = [str(site.specie) for site in pmg_structure.sites]
    cell = pmg_structure.lattice.matrix

    return Atoms(
        symbols=symbols,
        scaled_positions=scaled_positions,
        cell=cell,
        pbc=True
    )

def relax_structures_with_upet(
    model_name,
    device,
    fmax,
    max_steps):
    """
    Relax a list of ASE Atoms objects using MatterSim as the calculator
    and FrechetCellFilter to relax both atomic positions and cell
    """
    pet_calc = UPETCalculator(model=model_name, device=device)

    with open("input_structures.json", "r") as f:
        structure_list = json.loads(f.read())

    relaxed_structures = []
    energies = []
    epas = []
    num_failed = 0
    for structure in structure_list:
        atoms = pmg_to_ase(Structure.from_dict(structure))

        atoms.calc = pet_calc

        cell_filter = FrechetCellFilter(atoms)

        opt = BFGSLineSearch(cell_filter, logfile="opt.log")

        try:
            converged = opt.run(fmax=fmax, steps=max_steps)
        except:
            converged = False

        if converged:
            energy = float(atoms.get_potential_energy())
            energies.append(energy)
            pmg_structure = ase_to_pmg(atoms)
            relaxed_structures.append(pmg_structure.as_dict())
            epas.append(energy/len(pmg_structure.sites))
        else:
            num_failed += 1
    to_dump = {
        'structures': relaxed_structures,
        'energies': energies,
        'epas': epas
    }

    with open('output.json', 'w') as f:
        json.dump(to_dump, f)

    with open('total.txt', 'w') as f:
        f.write(str(len(structure_list)))

    with open('failed.txt', 'w') as f:
        f.write(str(num_failed))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--fmax", type=float)
    parser.add_argument("--max_steps", type=int)
    args = parser.parse_args()

    relax_structures_with_upet(args.model_name, args.device, args.fmax, args.max_steps)
