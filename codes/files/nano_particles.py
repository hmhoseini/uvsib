import sys
import json
import argparse
import numpy as np
from ase import Atoms
from ase.units import fs
from ase.build import bulk
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize.bfgslinesearch import BFGSLineSearch
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure


def generate_nano_particles(element_list, calculator, max_force, max_relax_steps, min_natoms, max_natoms, generator):
    """Generate nano-particles"""
    num_failed = 0
    relaxed_particles = list()
    elements = list(str(el) for el in element_list.split('-'))
    rnd_pick = elements[np.random.choice([int(i) for i in range(len(elements))], 1)[0]]
    bulk_atoms = bulk(name=rnd_pick, cubic=True)
    bulk_atoms = bulk_atoms.repeat((40, 40, 40))
    nano_seed = AseAtomsAdaptor().get_structure(bulk_atoms)

    atomic_energy = dict()
    for element in elements:
        atoms = Atoms(symbols=element, positions=[(0, 0, 0)], cell=[[20, 0, 0], [0, 20, 0], [0, 0, 20]])
        atoms.calc = calculator
        atomic_energy[element] = atoms.get_potential_energy()

    generated_clusters = list()
    for num_atoms in range(min_natoms, max_natoms):
        print(num_atoms)
        dr = 1
        coords = None
        species = None
        right_number = False
        while not right_number:
            coords = list()
            species = list()
            for site in nano_seed.get_neighbors_in_shell(origin=bulk_atoms.get_center_of_mass(), r=0, dr=dr):
                coords.append(site.coords)
                species.append(site.specie)
            if len(coords) >= num_atoms:
                right_number = True
                print('Particle diameter: {} nm'.format(np.round(2 * dr / 10, 1)))
            dr += 0.1

        spherical_cut = AseAtomsAdaptor().get_atoms(Structure(lattice=np.array([[100, 0, 0], [0, 100, 0], [0, 0, 100]]),
                                                              species=species, coords=coords, coords_are_cartesian=True))
        spherical_cut.center(vacuum=50)

        overshoot = len(spherical_cut) - num_atoms
        for ipop in sorted(np.random.choice(list(int(i) for i in range(len(spherical_cut))), overshoot, replace=False), reverse=True):
            spherical_cut.pop(ipop)

        spherical_cut.calc = calculator
        relax = BFGSLineSearch(spherical_cut, maxstep=0.1)
        relax.run(fmax=max_force, steps=max_relax_steps)
        if not relax.converged:
            # print('relax failed during generation step 1')
            num_failed += 1
            return

        if generator == 'grand_canonical':
            chemical_potential = 0
            for el in list(set(spherical_cut.get_chemical_symbols())):
                chemical_potential += spherical_cut.get_chemical_symbols().count(el) * atomic_energy[el]
            current_energy = (spherical_cut.get_potential_energy() - chemical_potential) / len(spherical_cut)

            fail_count = 0
            converged = False
            while not converged:
                orig_atoms = spherical_cut.copy()
                replace_element = elements[np.random.choice([int(i) for i in range(len(elements))], 1)[0]]
                spherical_cut[np.random.choice([int(i) for i in range(len(spherical_cut))], 1)[0]].symbol = replace_element

                spherical_cut.calc = calculator
                relax = BFGSLineSearch(spherical_cut, maxstep=0.1)
                relax.run(fmax=max_force, steps=max_relax_steps)
                if not relax.converged:
                    # print('relax failed during generation replacement step')
                    num_failed += 1
                    continue

                chemical_potential = 0
                for el in list(set(spherical_cut.get_chemical_symbols())):
                    chemical_potential += spherical_cut.get_chemical_symbols().count(el) * atomic_energy[el]
                new_energy = (spherical_cut.get_potential_energy() - chemical_potential) / len(spherical_cut)

                tmp = AseAtomsAdaptor().get_structure(spherical_cut)
                tmp.sort()
                spherical_cut = AseAtomsAdaptor().get_atoms(tmp)

                if (new_energy < current_energy) and (np.abs(new_energy - current_energy) > 1E-6):
                    current_energy = new_energy
                    fail_count = 0
                else:
                    fail_count += 1
                    spherical_cut = orig_atoms
                    if fail_count == 5:
                        converged = True

            spherical_cut.center(vacuum=10)
            generated_clusters.append(spherical_cut.copy())

        elif generator == 'systematic':
            if len(list(set(spherical_cut.get_chemical_symbols()))) > 2:
                raise NotImplementedError('systematic generation for more than two species not implemented yet')
            replace_species = [e for e in elements if e not in list(set(spherical_cut.get_chemical_symbols()))][0]
            generated_seed = spherical_cut.copy()
            for num_replace in range(num_atoms + 1):
                for _ in range(5):  # bias hardening
                    structure = AseAtomsAdaptor().get_structure(atoms=generated_seed)
                    for idx in sorted(np.random.choice([int(i) for i in range(len(generated_seed))], num_replace, replace=False), reverse=True):
                        structure.replace(idx, species=replace_species)
                    structure.sort()
                    heat_cell = Structure(lattice=np.array([[100, 0, 0], [0, 100, 0], [0, 0, 100]]),
                                          species=structure.species, coords=structure.cart_coords, coords_are_cartesian=True)

                    atoms = AseAtomsAdaptor().get_atoms(structure=heat_cell)
                    atoms.center(vacuum=10)
                    generated_clusters.append(atoms.copy())
        else:
            raise NotImplementedError('Generator {} not implemented yet'.format(generator))


    print('GENERATED: {}'.format(len(generated_clusters)))
    sys.stdout.flush()

    for atoms in generated_clusters:
        atoms.calc = calculator
        MaxwellBoltzmannDistribution(atoms, temperature_K=100)
        Stationary(atoms)
        ZeroRotation(atoms)

        timestep = 1*fs if np.any(np.array(list(set(atoms.get_chemical_symbols())) == 'H')) else 5*fs

        print('starting bussi on {}'.format(atoms))
        sys.stdout.flush()

        for multi in range(0, 15):
            md = Bussi(atoms, timestep=timestep, temperature_K=100+(100*multi), taut=100*fs)
            md.run(steps=250)

        md = Bussi(atoms, timestep=timestep, temperature_K=1500, taut=100*fs)
        md.run(steps=1000)

        for multi in range(14, -1, -1):
            md = Bussi(atoms, timestep=timestep, temperature_K=100+(100*multi), taut=100*fs)
            md.run(steps=250)

        relax = BFGSLineSearch(atoms, maxstep=0.1)
        relax.run(fmax=max_force, steps=max_relax_steps)
        if not relax.converged:
            print('relax failed during final step')
            num_failed += 1
            continue

        chemical_potential = 0
        for el in list(set(atoms.get_chemical_symbols())):
            chemical_potential += atoms.get_chemical_symbols().count(el) * atomic_energy[el]
        formation_energy = (atoms.get_potential_energy() - chemical_potential) / len(atoms)
        atoms.info['formation_energy'] = formation_energy

        relaxed_particles.append(AseAtomsAdaptor().get_structure(atoms).as_dict())


    output = dict({'structures': relaxed_particles})

    with open('output.json', 'w') as f:
        json.dump(output, f)
    with open('total.txt', 'w') as f:
        f.write(str(len(relaxed_particles)))
    with open('failed.txt', 'w') as f:
        f.write(str(num_failed))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=str)
    parser.add_argument("--ML_model", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--fmax", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--min_natoms", type=int)
    parser.add_argument("--max_natoms", type=int)
    parser.add_argument("--generator", type=str)
    args = parser.parse_args()

    if "MACE" in args.ML_model:
        from mace.calculators import MACECalculator
        calc = MACECalculator(model_paths=args.model_path, device=args.device)
    elif "PET" in args.ML_model:
        from upet.calculator import UPETCalculator
        calc = UPETCalculator(model=args.model, device=args.device)
    elif "MatterSim" in args.ML_model:
        from mattersim.forcefield import MatterSimCalculator
        calc = MatterSimCalculator(load_path=args.model_path, device=args.device)
    else:
        raise ValueError("Unknown ML_model {}, expected one of: MACE, PET, MatterSim.")

    generate_nano_particles(element_list=args.elements, calculator=calc, max_force=args.fmax, max_relax_steps=args.max_steps,
                            min_natoms=args.min_natoms, max_natoms=args.max_natoms, generator=args.generator)
