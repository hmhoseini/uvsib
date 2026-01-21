import argparse
from ase.io import read, write
from mace.calculators import MACECalculator

def energy_forces_with_mace(
    model_path,
    device,
    return_forces):
    """
    Relax a list of ASE Atoms objects using MACE as the calculator
    and ExpCellFilter to relax both atomic positions and cell.
    """

    calc = MACECalculator(
        model_path=model_path,
        device=device
    )

    atoms_list = read("input_structures.extxyz", index=":")

    energies = []
    forces = []

    for atoms in atoms_list:
        atoms.calc = calc

        energy = atoms.get_potential_energy()
        energies.append(energy)
        if return_forces:
            forces = atoms.get_forces()
            forces.append(forces)
    final_atoms = []
    for atoms, energy in zip(atoms_list, energies):
        atoms.info["MACE_Energy"] = energy
    write("output_structures.extxyz", final_atoms, format="extxyz")

    if forces:
        with open('forces.dat', 'w') as f:
            for a_force in forces:
                f.write(a_force)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str,
                        required=True, help="Path to MACE model")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"],
                        default="cuda", help="Computation device")
    parser.add_argument("--return_forces", type=bool,
                        default=False , help="Return forces")

    args = parser.parse_args()

    energy_forces_with_mace(args.model_path, args.device, args.return_forces)
