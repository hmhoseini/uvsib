import os
from aiida.engine import CalcJob
from aiida.orm import Dict, SinglefileData
from aiida.common.datastructures import CalcInfo, CodeInfo
from uvsib.workflows import settings


class NanoParticles(CalcJob):
    """One BATCH of nano-particle adsorbates: per particle a bare relax +
    Route-1 site finding/placement + adsorbate relaxations
    (codes/files/nano_particles.py); model and gas references are loaded /
    relaxed once per job.

    Stages, next to the runner (as ``aiida.py``):
      * ``slab_adsorbates.py`` = codes/files/adsorbates.py -- the shared
        X-dummy adsorbate registry, reaction pathways, gas references and
        validators, imported by the runner instead of duplicated;
      * ``_calculators.py`` and the molecular_references/*.vasp gas cells
        (adsorbates.py loads them at import).

    Submitted DIRECTLY by class (not via CalculationFactory -- there is no
    ``aiida.calculations`` entry point registered for it; add one to
    setup.json only if factory access is ever needed). Parsed with the
    generic registered ``sqs_parser`` (output.json -> output_dict).
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("parameters", valid_type=Dict)
        spec.input_namespace("file", valid_type=SinglefileData, dynamic=True)
        spec.output("output_dict", valid_type=Dict, required=True)

        spec.exit_code(100, "ERROR_MISSING_OUTPUT", message="Required output file not found.")
        spec.exit_code(200, "ERROR_NO_RETRIEVED_FOLDER", message="The retrieved folder data node can not be accessed.")
        spec.exit_code(303, "ERROR_OUTPUT_INCOMPLETE", message="The output file is incomplete.")

    def prepare_for_submission(self, folder):
        """Stage the runner, the shared adsorbates module and its data files."""
        parameters = self.inputs.parameters.get_dict()
        cmdline = parameters["cmdline_params"]

        staged = [
            ("nano_particles.py", "aiida.py"),
            ("adsorbates.py", "slab_adsorbates.py"),
            ("_calculators.py", "_calculators.py"),
        ]
        for src, dst in staged:
            with open(os.path.join(settings.files_path, src), "r",
                      encoding="utf-8") as f:
                content = f.read()
            with folder.open(dst, "w", encoding="utf-8") as f:
                f.write(content)

        # gas-phase reference cells; slab_adsorbates loads them at import
        for mol_file in os.listdir(settings.molecular_reference_files):
            with open(os.path.join(settings.molecular_reference_files, mol_file),
                      "r", encoding="utf-8") as f:
                content = f.read()
            with folder.open(mol_file, "w", encoding="utf-8") as f:
                f.write(content)

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = cmdline

        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.retrieve_list = ["output.json", "total.txt", "failed.txt",
                                  "rejected.json"]
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = [(file.uuid, file.filename, file.filename)
                                    for file in self.inputs.file.values()]

        return calcinfo
