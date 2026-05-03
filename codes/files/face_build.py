import json
import argparse
import numpy as np
from ase import Atoms
from ase.optimize.bfgslinesearch import BFGSLineSearch
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import Slab, generate_all_slabs


def pmg_to_ase(slab):
    """
    Convert a pymatgen Slab to ASE Atoms object,
    storing slab metadata in Atoms.info.
    """
    ase_atoms = AseAtomsAdaptor.get_atoms(slab)
    ase_atoms.info.update({
        "miller_index": tuple(slab.miller_index),
        "shift": slab.shift,
        "scale_factor": slab.scale_factor,
        "oriented_unit_cell": slab.oriented_unit_cell.as_dict()
    })
    return ase_atoms

def ase_to_pmg(atoms):
    """
    Convert an ASE Atoms object (with slab info in .info)
    back to a pymatgen Slab.
    """
    structure = AseAtomsAdaptor.get_structure(atoms)

    miller_index = tuple(atoms.info["miller_index"])
    shift = atoms.info.get("shift", 0.0)
    scale_factor = atoms.info.get("scale_factor", None)
    oriented_unit_cell = Structure.from_dict(atoms.info["oriented_unit_cell"])
    energy = atoms.info["energy"]

    return Slab(
        lattice=structure.lattice,
        species=structure.species,
        coords=structure.frac_coords,
        miller_index=miller_index,
        oriented_unit_cell=oriented_unit_cell,
        shift=shift,
        scale_factor=scale_factor,
        to_unit_cell=True,
        site_properties=structure.site_properties,
        energy = energy
    )

def process_slab(slab, target_vacuum=10.0, angle_tol=1.0):
    """
    Process a pymatgen Slab:
    - If alpha and beta are not ~90° within tolerance, return None.
    - If they are close, set them to exactly 90°.
    - Adjust vacuum along c to target_vacuum (Å).
    """

    a_len, b_len, c_len = slab.lattice.abc
    alpha, beta, gamma = slab.lattice.angles  # in degrees

    if abs(alpha - 90) > angle_tol or abs(beta - 90) > angle_tol:
        return None

    new_lattice = Lattice.from_parameters(
        a=a_len,
        b=b_len,
        c=c_len,
        alpha=90,
        beta=90,
        gamma=gamma
    )

    cart_coords = np.array(slab.cart_coords)
    z_coords = cart_coords[:, 2]
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    slab_thickness = z_max - z_min

    new_c = slab_thickness + target_vacuum
    scale_factor = new_c / c_len
    new_matrix = new_lattice.matrix.copy()
    new_matrix[2] *= scale_factor
    new_lattice = Lattice(new_matrix)

    return Slab(
        lattice=new_lattice,
        species=[site.specie for site in slab],
        coords=[site.coords for site in slab],
        miller_index=slab.miller_index,
        oriented_unit_cell=slab.oriented_unit_cell,
        shift=slab.shift,
        scale_factor=slab.scale_factor,
        coords_are_cartesian=True,
        to_unit_cell=True,
        site_properties=slab.site_properties
    )

def run_surface_builder(bulk_energy, calc, fmax, max_steps, max_miller_idx, max_num_surf):
    """Build and relax surfaces with given parameters """

    with open('input_structures.json', 'r') as f:
        structure_list = json.load(f)

    structure = Structure.from_dict(structure_list[0])
    nat = structure.num_sites
    epa = bulk_energy/nat

    sga = SpacegroupAnalyzer(structure, symprec=0.01, angle_tolerance=5)
    conv_struct = sga.get_conventional_standard_structure()

#TODO no need, the energy of bulk (bulk_energy) is passed
#    atomic_energy = dict()
#    for element in list(set(AseAtomsAdaptor().get_atoms(conv_struct).get_chemical_symbols())):
#        atoms = Atoms(symbols=element, positions=[(0, 0, 0)], cell=[[20, 0, 0], [0, 20, 0], [0, 0, 20]])
#        atoms.calc = calc
#        atomic_energy[element] = atoms.get_potential_energy()

    slabs = generate_all_slabs(conv_struct,
                               max_index=max_miller_idx,
                               min_slab_size=4,
                               min_vacuum_size=4,
                               max_normal_search=max_miller_idx,
                               in_unit_planes=True)

    orth_slabs = list()
    for s in slabs:
        orth_slab = process_slab(s)
        if not orth_slab:
            continue
        orth_slabs.append(orth_slab)
#TODO why not for all miller indices?
#    special_miller_supercells = list([(0, 0, 1), (1, 1, 1)])
#
#    tmp = list()
#    for slab in orth_slabs:
#        if slab.as_dict()['miller_index'] in special_miller_supercells:
#            one_by_two = slab.copy()
#            one_by_two.make_supercell((1, 2, 1))
#            tmp.append(one_by_two)
#            two_by_one = slab.copy()
#            two_by_one.make_supercell((2, 1, 1))
#            tmp.append(two_by_one)
#    orth_slabs.extend(tmp)

    num_failed = 0
    tmp_atoms = list()
    for slab in orth_slabs:
        atoms = pmg_to_ase(slab)
        atoms.info['miller_index'] = slab.as_dict()['miller_index']
        atoms.calc = calc
        relax = BFGSLineSearch(atoms, maxstep=0.1, logfile='opt.log')
        relax.run(fmax=fmax, steps=max_steps)
        if relax.converged:
            tmp_atoms.append(atoms)
        else:
            num_failed += 1

    slab_data = list()
    for atoms in tmp_atoms:
#        chemical_potential = 0
#        for el in list(set(atoms.get_chemical_symbols())):
#            chemical_potential += atoms.get_chemical_symbols().count(el) * atomic_energy[el]
#        area = atoms.cell.areas()[2] * 2.0
#        atoms.info['energy'] = atoms.get_potential_energy()
#        atoms.info['surface_formation_energy'] = (atoms.get_potential_energy() - chemical_potential) / (float(atoms.get_number_of_atoms()) * area)
        n_slab = len(atoms)
        area = atoms.cell.areas()[2] * 2.0
        surface_energy = (atoms.get_potential_energy() - (n_slab * epa)) / area
        atoms.info['energy'] = atoms.get_potential_energy()
        atoms.info['surface_formation_energy'] = surface_energy
        slab_data.append({"atoms": atoms, "surface_formation_energy": surface_energy})

    slab_data.sort(key=lambda x: x["surface_formation_energy"])
    selected_faces = slab_data[:max_num_surf]

#TODO why loop over special miller indices?
#    selected_faces = list()
#    for miller in special_miller_supercells:
#        tmp1 = list()
#        tmp2 = list()
#        for atoms in slab_data:
#            if atoms.info['miller_index'] == miller:
#                tmp1.append(atoms)
#            else:
#                tmp2.append(atoms)
#        for at, en in sorted(zip(tmp1, [e.info['surface_formation_energy'] for e in tmp1]), key=lambda x: x[1]):
#            selected_faces.append(at)
#            if len(selected_faces) > len(tmp1) * percentage_to_select :
#                break
#        for count, (at, en) in enumerate(sorted(zip(tmp2, [e.info['surface_formation_energy'] for e in tmp2]), key=lambda x: x[1])):
#            selected_faces.append(at)
#            if count > len(tmp2) * percentage_to_select :
#                break

#    built_faces = []
#    for atoms in selected_faces:
#        built_faces.append(ase_to_pmg(atoms).as_dict())

    built_faces = []
    for entry in selected_faces:
        at = entry["atoms"]
        built_faces.append(ase_to_pmg(at).as_dict())


    to_dump = dict({'slabs': built_faces})

    with open('output.json', 'w') as f:
        json.dump(to_dump, f)

    with open('total.txt', 'w') as f:
        f.write(str(len(orth_slabs)))

    with open('failed.txt', 'w') as f:
        f.write(str(num_failed))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bulk_energy", type=float)
    parser.add_argument("--ML_model", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--fmax", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--max_miller_idx", type=int)
    parser.add_argument("--max_num_surf", type=int)
    args = parser.parse_args()

    if "MACE" in args.ML_model:
        from mace.calculators import MACECalculator
        calc = MACECalculator(
                model_paths=args.model_path,
                device=args.device
        )
    elif "PET" in args.ML_model:
        from upet.calculator import UPETCalculator
        calc = UPETCalculator(
                model=args.model,
                device=args.device
        )
    elif "MatterSim" in args.ML_model:
        from mattersim.forcefield import MatterSimCalculator
        calc = MatterSimCalculator(
                load_path=args.model_path,
                device=args.device
        )
    else:
        raise ValueError(
            f"Unknown ML_model '{args.ML_model}'. "
            "Expected one of: MACE, PET, MatterSim."
        )

    run_surface_builder(args.bulk_energy, calc,
                        args.fmax, args.max_steps,
                        args.max_miller_idx,
                        args.max_num_surf)
