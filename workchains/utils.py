import os
from aiida.orm import load_code
from pymatgen.core.structure import Composition, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from uvsib.codes.utils import get_element_entries, get_structures_from_mpdb_by_composition
from uvsib.db.utils import query_structure, add_structures
from uvsib.workflows import settings

EHULL = settings.EHULL

matcher = StructureMatcher(
    ltol=0.3,
    stol=0.5,
    angle_tol=7,
    scale=True,
    attempt_supercell=False,
    allow_subset=False,
    primitive_cell=True,
)

def get_primitive_cell(struct_dict):
    """Refine a structure dictionary into its primitive cell"""
    structure = Structure.from_dict(struct_dict)

    sga = SpacegroupAnalyzer(
        structure,
        symprec=0.05,
        angle_tolerance=5,
    )

    try:
        prim_struct = sga.get_primitive_standard_structure()
    except:
        prim_struct = sga.find_primitive() or structure

    return prim_struct

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

def unique_low_energy_chemsys(chemical_system, entries, method):
    """Select the unique lowest-energy structures for a given chemical system"""
    if "-" in chemical_system:
        elements = chemical_system.split("-")
        entries.extend(get_element_entries(elements, method))
    pd = PhaseDiagram(entries)

    stable_entries = []
    existing_structs = []

    for entry in pd.entries:
        if entry.composition.chemical_system != chemical_system:
            continue
        if pd.get_e_above_hull(entry) > EHULL:
            continue

        prim_struct = get_primitive_cell(entry.structure.as_dict())

        if any(matcher.fit(prim_struct, s) for s in existing_structs):
            continue

        existing_structs.append(prim_struct)
        stable_entries.append(entry)
    return stable_entries

def unique_low_energy_comp(chemical_formula, entries, method, min_n_return=None):
    """Select the lowest-energy unique structures for a given chemical formula"""
    chemical_system = Composition(chemical_formula).chemical_system
    if "-" in chemical_system:
        elements = chemical_system.split("-")
        entries.extend(get_element_entries(elements, method))
    pd = PhaseDiagram(entries)

    candidates = []
    existing_structs = []

    for entry in pd.entries:
        if entry.composition.reduced_formula != chemical_formula:
            continue

        ehull = pd.get_e_above_hull(entry)
        prim_struct = get_primitive_cell(entry.structure.as_dict())

        if any(matcher.fit(prim_struct, s) for s in existing_structs):
            continue

        existing_structs.append(prim_struct)
        candidates.append((entry, ehull))

    candidates.sort(key=lambda x: x[1])

    stable_entries = [entry for entry, ehull in candidates if ehull <= EHULL]

    if min_n_return:
        if len(stable_entries) < min_n_return:
            for entry, _ in candidates[len(stable_entries):min_n_return]:
                stable_entries.append(entry)
    return stable_entries

def add_from_mpdb(chemical_formula):
    """Add structures from MPDB"""
    results = query_structure({"composition": chemical_formula}, source = "MPDB")
    if results:
        return
    stable_structures = get_structures_from_mpdb_by_composition(chemical_formula, EHULL)
    if stable_structures:
        for s in stable_structures:

            prim_struct = get_primitive_cell(s)

            add_structures(
                    "MPDB",
                    "mixed",
                    [(prim_struct.as_dict(), None)]
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
    if ML_model in ["MatterGen", "uPET"]:
        model_path = None
    else:
        model_path = os.path.join(path_to_pretrained_models, model)
    if ML_model in ["MatterGen", "MatterGenCSP"]:
        device = None
    else:
        device = settings.configs["codes"][ML_model]["job_script"]["device"]
    return model, model_path, device
