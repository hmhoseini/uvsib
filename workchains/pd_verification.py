import os
import yaml
from pymatgen.entries.computed_entries import ComputedStructureEntry
from aiida.orm import Str, load_code, StructureData
from aiida.engine import WorkChain
from uvsib.codes.vasp.workchains import construct_vasp_builder
from uvsib.db.tables import DBComposition
from uvsib.db.utils import query_structure, add_version_to_existing_structure, query_by_columns
from uvsib.workchains.utils import get_primitive_cell, unique_low_energy_comp, add_from_mpdb
from uvsib.workflows import settings

def read_yaml(file_path):
    """Read yaml file"""
    with open(file_path, "r", encoding="utf8") as fhandle:
        data = yaml.safe_load(fhandle)
    return data

def get_struct_uuid(chemical_formula, method):
    """Query structures from the database by formula and methed.
    Return list of (structure_dict, uuid)
    """
    struct_uuid = []
    mpdb_results = query_structure(
            {"composition":chemical_formula},
            source="MPDB"
    )
    if mpdb_results:
        for result in mpdb_results:
            struct_uuid.append((result.structure, result.structure_uuid))

    row = query_by_columns(DBComposition, {"composition": chemical_formula})[0]
    uuid_list = row.stable_struct.get("ml_uuid_list")

    for uuid_str in uuid_list:
        result = query_structure({"uuid": uuid_str}, method=method)
        if result:
            struct_uuid.append((result[0].structure, uuid_str))
    return struct_uuid

def get_vasp_output_as_entry(wch, uuid_str):
    """Extract structure and energy outputs from a VASP calculation"""
    outputs = wch.outputs
    return ComputedStructureEntry(
            structure = outputs.structure.get_pymatgen(),
            energy = outputs.misc["total_energies"]["energy_extrapolated"],
            data = {"uuid": uuid_str})

class PDVerificationWorkChain(WorkChain):
    """Work chain for verification"""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("chemical_formula", valid_type=Str)
        spec.input("ML_model", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.run_scan,
            cls.store_scan_result,
            cls.final_report
        )

        spec.exit_code(300,
            "ERROR_CALCULATION_FAILED",
            message="The calculation did not finish successfully"
        )

    def setup(self):
        """Setup and report"""
        self.report("Running PDVerification WorkChain")
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.ML_model = self.inputs.ML_model.value
        add_from_mpdb(self.ctx.chemical_formula)
        self.ctx.struct_uuid = get_struct_uuid(self.ctx.chemical_formula, self.ctx.ML_model)

        self.ctx.protocol = read_yaml(
                os.path.join(settings.vasp_files_path, "protocol.yaml")
        )
        self.ctx.potential_family = settings.configs["codes"]["VASP"]["potential_family"]
        potential_mapping = read_yaml(os.path.join(settings.vasp_files_path, "potential_mapping.yaml"))
        self.ctx.potential_mapping = potential_mapping["potential_mapping"]
        self.ctx.vasp_code = load_code(
                settings.configs["codes"]["VASP"]["code_string"]
        )

    def run_scan(self):
        """Run r2SCAN geometry optimization"""
        for struct_dict, uuid_str in self.ctx.struct_uuid:
            pmg_structure = get_primitive_cell(struct_dict)
            builder = construct_vasp_builder(
                StructureData(pymatgen=pmg_structure),
                self.ctx.protocol["r2SCAN"],
                self.ctx.potential_family,
                self.ctx.potential_mapping,
                self.ctx.vasp_code
            )
            future = self.submit(builder)
            self.to_context(**{f"scan_{uuid_str}": future})

    def store_scan_result(self):
        """Inspect SCAN calculations"""
        scan_entries = []
        failed_scan = []
        for _, uuid_str in self.ctx.struct_uuid:
            scan_wch = self.ctx[f"scan_{uuid_str}"]
            if scan_wch.is_finished_ok:
                scan_entries.append(get_vasp_output_as_entry(scan_wch, uuid_str))
            else:
                failed_scan.append(uuid_str)
        if failed_scan:
            self.report(f"Warning: r2SCAN geometry optimization failed (structure uuids: {failed_scan})")

        low_energy_entries = unique_low_energy_comp(
                self.ctx.chemical_formula,
                scan_entries,
                "r2SCAN"
        )
        for entry in low_energy_entries:
            add_version_to_existing_structure(
                entry.data["uuid"],
                "r2SCAN",
                {"structure": entry.structure.as_dict(),
                 "energy": entry.energy}
            )

    def final_report(self):
        """Final report"""
        self.report("PDVerification WorkChain finished successfully")
