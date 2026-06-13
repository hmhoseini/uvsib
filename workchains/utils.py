import os
import yaml
from aiida.orm import load_code
from pymatgen.core.structure import Composition, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from uvsib.codes.utils import (
    get_element_entries,
    get_structures_from_mpdb_by_composition,
    get_mp_element_structures,
)
from uvsib.db.utils import query_structure, add_structures
from uvsib.db.session import get_session
from uvsib.db.tables import DBStructure, DBStructureVersion
from uvsib.workflows import settings

EHULL_SCAN = settings.EHULL_SCAN

matcher = StructureMatcher(
    ltol=0.3,
    stol=0.5,
    angle_tol=7,
    scale=True,
    attempt_supercell=False,
    allow_subset=False,
    primitive_cell=True,
)

def _read_yaml(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

_workchains_dir = os.path.join(settings.uvsib_directory,"workchains")
_ZPE_ALL: dict    = _read_yaml(os.path.join(_workchains_dir, "zpe_corrections.yaml"))

def load_zpe(reaction):
    """Return the ZPE correction dict (species name → ZPE correction (eV)) for the given reaction type ('oer', 'co2rr', 'noxrr')"""
    try:
        return dict(_ZPE_ALL[reaction])
    except:
        return None

def get_primitive_cell(struct_dict):
    """Refine a structure dictionary into its primitive cell"""
    structure = Structure.from_dict(struct_dict)

    sga = SpacegroupAnalyzer(structure, symprec=0.05, angle_tolerance=5)

    try:
        prim_struct = sga.get_primitive_standard_structure()
    except:
        prim_struct = sga.find_primitive() or structure

    return prim_struct

def get_output_as_entry(wch):
    """Extract structures and energies from an ML energy prediction calculation"""
    entries = []
    output_dict = wch.outputs.output_dict
    for indx, struct in enumerate(output_dict["structures"]):
        entries.append(ComputedStructureEntry(
            structure = Structure.from_dict(struct),
            energy = output_dict["energies"][indx])
        )
    return entries

def unique_low_energy_chemsys(chemical_system, entries, method, ehull, element_entries=None):
    """Select the unique lowest-energy structures for a given chemical system.

    ``element_entries`` are the elemental hull endpoints to add. Pass the
    on-method MLIP references (``element_reference_entries``); when ``None`` the
    bundled DFT references are used (``method`` selects GGA/r2SCAN) -- legacy
    behaviour, kept for backward compatibility.
    """
    if "-" in chemical_system:
        elements = chemical_system.split("-")
        entries.extend(element_entries if element_entries is not None
                       else get_element_entries(elements, method))
    pd = PhaseDiagram(entries)

    stable_entries = []
    existing_structs = []

    for entry in pd.entries:
        if entry.composition.chemical_system != chemical_system:
            continue
        if pd.get_e_above_hull(entry) > ehull:
            continue

        prim_struct = get_primitive_cell(entry.structure.as_dict())

        if any(matcher.fit(prim_struct, s) for s in existing_structs):
            continue

        existing_structs.append(prim_struct)
        stable_entries.append(entry)
    return stable_entries

def unique_low_energy_comp(chemical_formula, entries, method, ehull, min_n_return=None, element_entries=None):
    """Select the lowest-energy unique structures for a given chemical formula.

    ``element_entries``: elemental hull endpoints to add (the on-method MLIP
    references from ``element_reference_entries``); ``None`` -> bundled DFT
    references (legacy).
    """
    chemical_system = Composition(chemical_formula).chemical_system
    if "-" in chemical_system:
        elements = chemical_system.split("-")
        entries.extend(element_entries if element_entries is not None
                       else get_element_entries(elements, method))
    pd = PhaseDiagram(entries)

    candidates = []
    existing_structs = []

    stable_entries = []
    ehulls = []

    for en in pd.entries:
        if en.composition.reduced_formula != chemical_formula:
            continue

        prim_struct = get_primitive_cell(en.structure.as_dict())

        if any(matcher.fit(prim_struct, s) for s in existing_structs):
            continue

        existing_structs.append(prim_struct)
        candidates.append((en, pd.get_e_above_hull(en)))

    candidates.sort(key=lambda x: x[1])

    for en, eh in candidates:
        if eh <= ehull:
            stable_entries.append(en)
            ehulls.append(eh)

    if min_n_return:
        if len(stable_entries) < min_n_return:
            for en, eh in candidates[len(stable_entries):min_n_return]:
                stable_entries.append(en)
                ehulls.append(eh)
    return stable_entries, ehulls

def add_from_mpdb(chemical_formula):
    """Add structures from MPDB"""
    results = query_structure({"composition": chemical_formula}, source = "MPDB")
    if results:
        return
    stable_structures = get_structures_from_mpdb_by_composition(chemical_formula, EHULL_SCAN)
    if stable_structures:
        for s in stable_structures:
            prim_struct = get_primitive_cell(s)
            add_structures("MPDB","mixed",[(prim_struct.as_dict(), None)])

def get_code(model_key):
    """
    Helper to fetch builder.code, model_path, and device 
    """
    return load_code(settings.configs["codes"][model_key]["code_string"])

def get_model_device(ML_model):
    """Return (model_path, device) for the given ML model."""
    path_to_pretrained_models = settings.configs["models"]["path_to_pretrained_models"]
    model = settings.configs["models"][ML_model]
    model_path = os.path.join(path_to_pretrained_models, model)
    device = settings.configs["codes"][ML_model]["job_script"]["device"]

    return model, model_path, device


# --------------------------------------------------------------------------- #
# on-method elemental hull references
#
# All structures on the convex hull must share one energy method, or e_above_hull
# measures the method mismatch rather than (meta)stability (a pure element then
# sits above its own hull). Instead of the bundled DFT elemental energies, we
# take the elemental ground states from the Materials Project and relax them with
# the SAME MLIP used for the generated structures (mirrors how the molecular
# references are relaxed against the same calculator in the adsorbate/SQS paths).
# The relaxed references are cached in the DB as source="reference"; because
# get_entries_from_db filters only on method, they are then picked up as hull
# vertices automatically wherever the hull is built from the DB.
# --------------------------------------------------------------------------- #
def get_ml_element_entries(elements, model):
    """Lowest-energy stored MLIP elemental reference per element (source='reference')."""
    refs = []
    with get_session() as session:
        for el in elements:
            row = (
                session.query(DBStructureVersion)
                .join(DBStructure)
                .filter(DBStructure.chemsys == el)
                .filter(DBStructureVersion.method == model)
                .filter(DBStructureVersion.source == "reference")
                .order_by(DBStructureVersion.energy.asc())
                .first()
            )
            if row is None:
                continue
            refs.append(ComputedStructureEntry(
                structure=Structure.from_dict(row.structure),
                energy=row.energy,
                data={"reference": True, "element": el},
            ))
    return refs


def missing_element_references(elements, model):
    """Elements that have no MLIP reference stored yet (need computing)."""
    have = {e.data.get("element") for e in get_ml_element_entries(elements, model)}
    return [el for el in elements if el not in have]


def element_reference_entries(elements, model):
    """(entries, missing) elemental hull endpoints on the MLIP method.

    Uses the cached MLIP references; any element not yet computed is filled with
    the bundled DFT reference so the hull still builds -- ``missing`` lists those
    DFT-filled elements (a per-element offset for them only), which the caller
    should log. In steady state (references computed) ``missing`` is empty and
    every endpoint is on-method.
    """
    refs = get_ml_element_entries(elements, model)
    have = {e.data.get("element") for e in refs}
    missing = [el for el in elements if el not in have]
    if missing:
        refs = list(refs) + get_element_entries(missing, settings.DFT_FUNC)
    return refs, missing


def construct_reference_relax_builder(structures, model):
    """Builder that ML-relaxes elemental reference structures with the SAME model
    and head as ``bulk_relax`` (so the references are on-method)."""
    from aiida.plugins import WorkflowFactory
    from aiida.orm import List, Str, Dict

    Workflow = WorkflowFactory(model.lower())
    builder = Workflow.get_builder()
    builder.input_structures = List(list(structures))
    builder.code = get_code(model)
    builder.local_label = Str("element references")
    model_name, model_path, device = get_model_device(model)
    builder.job_info = Dict({
        "job_type": "relax",
        "ML_model": model,
        "model_name": model_name,
        "model_path": model_path,
        "model_head": settings.inputs["bulk_relax"]["head"],
        "device": device,
        "fmax": settings.inputs["bulk_relax"]["fmax"],
        "max_steps": settings.inputs["bulk_relax"]["max_steps"],
    })
    return builder
