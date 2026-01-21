import json
import argparse
import numpy as np
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

def run_surface_builder(bulk_energy,
                        calc,
                        fmax,
                        max_steps,
                        max_miller_idx,
                        percentage_to_select):
    """
    Build and relax surfaces with given parameters
    """

    with open('input_structures.json', 'r') as f:
        structure_list = json.loads(f.read())

    structure = Structure.from_dict(structure_list[0])
    nat = structure.num_sites
    epa = bulk_energy/nat
    sga = SpacegroupAnalyzer(structure, symprec=0.01, angle_tolerance=5)
    conv_struct = sga.get_conventional_standard_structure()

    num_failed = 0
    tmp_atoms = []
    built_faces = []

    slabs = generate_all_slabs(conv_struct,
                               max_index=max_miller_idx,
                               min_slab_size=4,
                               min_vacuum_size=4,
                               max_normal_search=max_miller_idx,
                               in_unit_planes=True)
    orth_slabs = []
    for s in slabs:
        orth_slab = process_slab(s)
        if not orth_slab:
            continue
        orth_slabs.append(orth_slab)

    for slab in orth_slabs:
        atoms = pmg_to_ase(slab)
        atoms.calc = calc
        relax = BFGSLineSearch(atoms, maxstep=0.1, logfile='opt.log')
        try:
            converged = relax.run(fmax=fmax, steps=max_steps)
        except:
            converged = False

        if converged:
            tmp_atoms.append(atoms)
        else:
            num_failed += 1

    slab_data = []
    for atoms in tmp_atoms:
        n_slab = len(atoms)
        n_bulk = n_slab / nat
        area = atoms.cell.areas()[2] * 2.0
        surface_energy = (float(atoms.get_potential_energy()) - n_bulk * epa) / area
        slab_data.append({"atoms": atoms, "surface_energy": surface_energy})

    slab_data.sort(key=lambda x: x["surface_energy"])
    n_select = max(1, int(len(slab_data) * percentage_to_select/100))
    selected = slab_data[:n_select]

    built_faces = []
    for entry in selected:
        at = entry["atoms"]
        at.info["energy"] = float(at.get_potential_energy())
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
    parser.add_argument("--percentage_to_select", type=float)
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
                        args.percentage_to_select)
