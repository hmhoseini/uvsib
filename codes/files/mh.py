import json
import argparse
from ase.io import read
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from minimahopping.minhop import Minimahopping

matcher = StructureMatcher(
    ltol=0.3,
    stol=0.5,
    angle_tol=7,
    scale=True,
    attempt_supercell=False,
    allow_subset=False,
    primitive_cell=True,
)

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
            symprec=0.05,
            verbose_output=False,
            energy_threshold=0.05,
            fingerprint_threshold=5e-1,
            T0=1000,
            dt0=0.01,
            write_graph_output=False,
            use_MPI=False,
            mdmin=5) as mh:
        mh(totalsteps=int(mh_steps))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ML_model", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--mh_steps", type=str)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    if "MACE" in args.ML_model:
        from mace.calculators import MACECalculator
        c = MACECalculator(
                model_path=args.model_path,
                device=args.device
        )
    elif "PET" in args.ML_model:
        from upet.calculator import UPETCalculator
        c = UPETCalculator(model=args.model, device=args.device)
    elif "MatterSim" in args.ML_model:
        from mattersim.forcefield import MatterSimCalculator
        c = MatterSimCalculator(
                load_path=args.model_path,
                device=args.device
        )
    else:
        raise ValueError(
            f"Unknown ML_model '{args.ML_model}'. "
            "Expected one of: MACE, PET, MatterSim."
        )

    run_mh(c, args.mh_steps)

    try:
        accepted_minima = read('output/accepted_minima.extxyz', index=':')
    except:
        pass
    existing_structs = []
    structures = []
    energies = []
    for atoms in accepted_minima[1:]:
        pmg_struct = ase_to_pmg(atoms)
        energy = float(atoms.get_potential_energy())

        sga = SpacegroupAnalyzer(
            pmg_struct,
            symprec=0.05,
            angle_tolerance=5,
        )

        try:
            prim_struct = sga.get_primitive_standard_structure()
        except:
            prim_struct = sga.find_primitive() or pmg_struct

        if any(matcher.fit(prim_struct, existing) for existing in existing_structs):
            continue

        existing_structs.append(prim_struct)
        structures.append(prim_struct.as_dict())
        energies.append(energy * (prim_struct.num_sites/pmg_struct.num_sites))

    to_dump = {
            'structures': structures,
            'energies': energies,
    }

    with open('output.json', 'w') as f:
        json.dump(to_dump, f)
