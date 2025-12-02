import json
import argparse
from ase.io import read
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from minimahopping.minhop import Minimahopping

def ase_to_pmg(atoms):
    """
    Convert an ASE Atoms object to a pymatgen Structure dictionary
    """
    lattice = atoms.cell.array.tolist()
    symbols = atoms.get_chemical_symbols()
    frac_coords = atoms.get_scaled_positions().tolist()
    lattice_obj = Lattice(lattice)
    structure = Structure(lattice_obj,
                          symbols,
                          frac_coords,
                          coords_are_cartesian=False
    )
    return structure

def run_mh(calc, mh_steps):

    with open("initial_structure.json", "r") as f:
        struct = json.loads(f.read())
    pmg_structure = Structure.from_dict(struct)
    init_conf = AseAtomsAdaptor.get_atoms(pmg_structure)
    init_conf.calc = calc

    with Minimahopping(
            init_conf,
            symprec=5e-3,
            verbose_output=False,
            energy_threshold=0.01,
            fingerprint_threshold=5e-1,
            T0=1000,
            dt0=0.01,
            write_graph_output=False,
            use_MPI=False,
            mdmin=5) as mh:
        mh(totalsteps=int(mh_steps))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--mh_steps", type=str)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    if "MACE" in args.model:
        from mace.calculators import MACECalculator
        c = MACECalculator(
                model_path=args.model_path,
                device=args.device
        )
    else:
        from mattersim.forcefield import MatterSimCalculator
        c = MatterSimCalculator(
                load_path=args.model_path,
                device=args.device
        )

    run_mh(c, args.mh_steps)

    try:
        accepted_minima = read('output/accepted_minima.extxyz', index=':')
    except:
        pass

    structures = []
    energies = []
    for atoms in accepted_minima:
        pmg_struct = ase_to_pmg(atoms)
        energy = float(atoms.get_potential_energy())
        structures.append(pmg_struct.as_dict())
        energies.append(energy)

    to_dump = {
            'structures': structures,
            'energies': energies,
    }

    with open('output.json', 'w') as f:
        json.dump(to_dump, f)
