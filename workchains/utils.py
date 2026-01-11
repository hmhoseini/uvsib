import os
import numpy as np
from aiida.orm import load_code
from pymatgen.core.structure import Composition, Lattice, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from uvsib.codes.utils import get_element_entries, get_structures_from_mpdb_by_composition
from uvsib.db.utils import query_structure, add_structures
from uvsib.workflows import settings

_EHULL = 0.05

matcher = StructureMatcher(
    ltol=0.7,
    stol=0.7,
    angle_tol=5,
    scale=True,
    attempt_supercell=False,
    allow_subset=False,
    primitive_cell=False,
)

def refine_primitive_cell(struct_dict):
    """Refine a structure dictionary into its primitive cell"""
    structure = Structure.from_dict(struct_dict)
    lattice_matrix = structure.lattice.matrix.copy()
    lattice_matrix[np.abs(lattice_matrix) < 0.1] = 0.0

    new_lattice = Lattice(lattice_matrix)
    a, b, c = new_lattice.abc
    alpha, beta, gamma = (round(ang) for ang in new_lattice.angles)

    try:
        final_lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    except ValueError:
        final_lattice = new_lattice

    refined = Structure(
        final_lattice,
        structure.species,
        structure.frac_coords,
        coords_are_cartesian=False,
    )

    primitive = SpacegroupAnalyzer(
            refined, symprec=0.1, angle_tolerance=5
            ).find_primitive()

    return primitive

def get_output_as_entry(wch):
    """Extract structures and energies from an ML energy prediction calculation"""
    entries = []
    output_dict = wch.called[-1].outputs.output_dict
    for indx, struct in enumerate(output_dict["structures"]):
        entries.append(ComputedStructureEntry(
            structure = Structure.from_dict(struct),
            energy = output_dict["energies"][indx])
        )
    return entries

def get_unique_low_energy(entries, chemical_system, ehull_threshold = _EHULL):
    """Helper to filter unique low-energy structures"""
    pd = PhaseDiagram(entries)
    stable_entries = []
    existing_structs = []

    for entry in pd.entries:
        if entry.composition.chemical_system != chemical_system:
            continue
        if pd.get_e_above_hull(entry) >= ehull_threshold:
            continue

        sga = SpacegroupAnalyzer(
            entry.structure,
            symprec=0.1,
            angle_tolerance=5,
        )
        prim_struct = sga.find_primitive() or entry.structure

        if any(matcher.fit(prim_struct, s) for s in existing_structs):
            continue

        existing_structs.append(prim_struct)
        stable_entries.append(entry)
    return stable_entries

def unique_low_energy_chemsys(chemical_system, entries, method):
    """Select the unique lowest-energy structures for a given chemical system"""
    if "-" in chemical_system:
        elements = chemical_system.split("-")
        entries.extend(get_element_entries(elements, method))
    return get_unique_low_energy(entries, chemical_system)

def unique_low_energy_comp(chemical_formula, entries, method):
    """Select the lowest-energy unique structures for a given chemical formula"""
    chemical_system = Composition(chemical_formula).chemical_system
    if "-" in chemical_system:
        elements = chemical_system.split("-")
        entries.extend(get_element_entries(elements, method))
    return get_unique_low_energy(entries, chemical_system)

def add_from_mpdb(chemical_formula):
    """Add structures from MPDB"""
    results = query_structure({"composition": chemical_formula}, source = "MPDB")
    if results:
        return
    stable_structures = get_structures_from_mpdb_by_composition(chemical_formula, _EHULL)
    if stable_structures:
        for stable_struct in stable_structures:
            add_structures(
                    "MPDB",
                    "mixed",
                    [(stable_struct, None)]
                )

def get_code(model_key):
    """
    Helper to fetch builder.code, model_path, and device 
    """
    return load_code(
        settings.configs["codes"][model_key]["code_string"])

def get_model_device(ML_model):
    """Return (model_path, device) for the given ML model."""
    path_to_pretrained_models = settings.configs["models"]["path_to_pretrained_models"]
    model = settings.configs["models"][ML_model]
    if ML_model in ["MatterGen"]:
        model_path = None
    else:
        model_path = os.path.join(path_to_pretrained_models, model)
    if ML_model in ["MatterGen", "MatterGenCSP"]:
        device = None
    else:
        device = settings.configs["codes"][ML_model]["job_script"]["device"]
    return model, model_path, device
