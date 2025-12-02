import os
import tempfile
import yaml
from pymatgen.io.vasp import Vasprun
from aiida.orm import Str, load_code, StructureData
from aiida.engine import WorkChain
from uvsib.codes.vasp.workchains import construct_vasp_builder
from uvsib.codes.vasp.bp import get_band_info
from uvsib.db.utils import query_structure, add_version_to_existing_structure
from uvsib.workchains.utils import refine_primitive_cell
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
            cls.run_pbe,
            cls.pbe_result,
            cls.run_hse,
            cls.hse_result,
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

        self.ctx.protocol = read_yaml(
                os.path.join(settings.vasp_files_path, "protocol.yaml")
        )
        self.ctx.potentials = read_yaml(
                os.path.join(settings.vasp_files_path, "potentials.yaml")
        )
        self.ctx.vasp_code = load_code(
                settings.configs["codes"]["VASP"]["code_string"]
        )

        self.ctx.struct_uuid = get_struct_uuid(self.ctx.chemical_formula)
        if not self.ctx.struct_uuid:
            return self.exit_codes.ERROR_NO_STRUCTURES_FOUND

    def run_pbe(self):
        """Run PBE SP calculations """
        for struct_dict, uuid_str in self.ctx.struct_uuid:
            pmg_structure = refine_primitive_cell(struct_dict)
            builder = construct_vasp_builder(
                StructureData(pymatgen=pmg_structure),
                self.ctx.protocol["PBE"],
                self.ctx.potentials,
                self.ctx.vasp_code
            )
            future = self.submit(builder)
            self.to_context(**{f"pbe_{uuid_str}": future})

    def pbe_result(self):
        """Inspect PBE calculations"""
        self.ctx.pbe_results = []
        failed_pbe = []
        for _, uuid_str in self.ctx.struct_uuid:
            pbe_wch = self.ctx[f"pbe_{uuid_str}"]
            if pbe_wch.is_finished_ok:
                add_version_to_existing_structure(
                        uuid_str,
                        "PBE",
                        {"structure": pbe_wch.inputs.structure.get_pymatgen().as_dict(),
                         "energy": pbe_wch.outputs.misc["total_energies"]["energy_extrapolated"],
                        },
                        "override"
                )

                self.ctx.pbe_results.append(uuid_str)
            else:
                failed_pbe.append(uuid_str)
        if failed_pbe:
            self.report(f"Warning: PBE geometry optimization failed (structure uuids: {failed_pbe})")

    def run_hse(self):
        """Run HSE calculations"""
        for uuid_str in self.ctx.pbe_results:
            pbe_wch = self.ctx[f"pbe_{uuid_str}"]
            builder = construct_vasp_builder(
                    pbe_wch.outputs.structure,
                    self.ctx.protocol["HSE"],
                    self.ctx.potentials,
                    self.ctx.vasp_code,
                    pbe_wch.outputs.remote_folder
            )
            future = self.submit(builder)
            self.to_context(**{f"hse_{uuid_str}": future})

    def hse_result(self):
        """Inspect HSE calculations"""
        failed_hse = []
        for uuid_str in self.ctx.pbe_results:
            hse_wch = self.ctx[f"hse_{uuid_str}"]
            if hse_wch.is_finished_ok:
                vr_str, band_info_dict = get_band_info(hse_wch)
                add_version_to_existing_structure(
                        uuid_str,
                        "HSE",
                        {"structure": hse_wch.inputs.structure.get_pymatgen().as_dict(),
                         "energy": hse_wch.outputs.misc["total_energies"]["energy_extrapolated"],
                         "vasprun_str": vr_str,
                         "attributes": {"band_info": band_info_dict}
                        },
                        "override"
                )
            else:
                failed_hse.append(uuid_str)
        if failed_hse:
            self.report(f"Warning: HSE calculations failed (structure uuids: {failed_hse})")

    def final_report(self):
        """Final report"""
        self.report("BandAlignment Workchain finished successfully")
