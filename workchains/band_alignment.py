import os
import tempfile
import yaml
from pymatgen.io.vasp import Vasprun
from aiida.orm import Str, load_code, StructureData
from aiida.engine import WorkChain
from uvsib.codes.vasp.workchains import construct_vasp_builder
from uvsib.codes.vasp.band_info import get_band_info, get_core_state
from uvsib.db.utils import query_structure, add_version_to_existing_structure
from uvsib.workchains.utils import get_primitive_cell
from uvsib.workflows import settings

def read_yaml(file_path):
    """Read a yaml file"""
    with open(file_path, "r", encoding="utf8") as fhandle:
        return yaml.safe_load(fhandle)

def get_struct_uuid(chemical_formula):
    """Query structures from the database by formula and return list of (structure_dict, uuid)"""
    results = query_structure({"composition": chemical_formula}, method = "r2SCAN") or []
    return [(row.structure, str(row.structure_uuid)) for row in results]

def load_vasprun_from_content(wch):
    """Load a Vasprun object from the xml content of a workchain output"""
    vasprun_str = wch.called[-1].outputs.retrieved.get_object_content("vasprun.xml")
    with tempfile.NamedTemporaryFile(mode="w", delete=True) as tmp:
        tmp.write(vasprun_str)
        tmp.flush()
        vasprun = Vasprun(tmp.name)
    return vasprun_str, vasprun

class BandAlignmentWorkChain(WorkChain):
    """Work chain for band alignment"""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("chemical_formula", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.run_scan,
            cls.scan_result,
            cls.final_report
        )

        spec.exit_code(300,
            "ERROR_CALCULATION_FAILED",
            message="The calculation did not finish successfully"
        )
        spec.exit_code(
            301,
            "ERROR_NO_STRUCTURES_FOUND",
            message="No structures were found for the given formula"
        )

    def setup(self):
        """Setup and report"""
        self.report("Running BandAlignment WorkChain")
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.struct_uuid = get_struct_uuid(self.ctx.chemical_formula)
        if not self.ctx.struct_uuid:
            self.report(f"No structures were found for {self.ctx.chemical_formula}")
            return self.exit_codes.ERROR_NO_STRUCTURES_FOUND
        self.ctx.protocol = read_yaml(os.path.join(settings.vasp_files_path, "protocol.yaml"))
        self.ctx.potential_family = settings.configs["codes"]["VASP"]["potential_family"]
        potential_mapping = read_yaml(os.path.join(settings.vasp_files_path, "potential_mapping.yaml"))
        self.ctx.potential_mapping = potential_mapping["potential_mapping"]
        self.ctx.vasp_code = load_code(settings.configs["codes"]["VASP"]["code_string"])
        self.ctx.dft_results = list()

    def run_scan(self):
        """Run PBE SinglePoint calculations """
        for struct_dict, uuid_str in self.ctx.struct_uuid:
            pmg_structure = get_primitive_cell(struct_dict)
            builder = construct_vasp_builder(StructureData(pymatgen=pmg_structure), self.ctx.protocol["HSE"],
                                             self.ctx.potential_family, self.ctx.potential_mapping, self.ctx.vasp_code)
            future = self.submit(builder)
            self.to_context(**{f"r2scan_{uuid_str}": future})

    def scan_result(self):
        """Inspect PBE calculations"""
        failed = []
        for _, uuid_str in self.ctx.struct_uuid:
            wch = self.ctx[f"r2scan_{uuid_str}"]
            if wch.is_finished_ok:
                core_state_dict = get_core_state(wch)
                add_version_to_existing_structure(uuid_str,"HSE",
                                                  {"structure": wch.inputs.structure.get_pymatgen().as_dict(),
                                                   "energy": wch.outputs.misc["total_energies"]["energy_extrapolated"],
                                                   "band_info": core_state_dict},"override")
                self.ctx.dft_results.append(uuid_str)
            else:
                failed.append(uuid_str)
        if failed:
            self.report(f"Warning: HSE band alignment failed for structure uuids: {failed}")

    def final_report(self):
        """Final report"""
        self.report("BandAlignment WorkChain finished successfully")
