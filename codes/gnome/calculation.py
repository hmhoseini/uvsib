import os
from aiida.engine import CalcJob
from aiida.orm import Dict, SinglefileData
from aiida.common.datastructures import CalcInfo, CodeInfo
from uvsib.workflows import settings

# runner + sidecars staged next to aiida.py on the compute node
_STAGED = [
    ("gnome_generate.py", "aiida.py"),   # the runner becomes aiida.py
    ("_saps.py", "_saps.py"),            # symmetry-aware partial substitution
    ("refine.py", "refine.py"),          # charge-neutral + primitive dedup (shared w/ MatterGen)
    ("_calculators.py", "_calculators.py"),  # ASE calculator factory for the GNN screen
]


class GNoMECalculation(CalcJob):
    """AiiDA plugin for the GNoME-style SAPS generator.

    Stages the runner and its sidecars plus a ``templates.json`` seed pool
    (passed in via the ``file`` namespace), runs it, and retrieves
    ``output.json`` -- the same output contract as MatterGen.
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
        parameters = self.inputs.parameters.get_dict()
        cmdline = parameters['cmdline_params']

        for src, dst in _STAGED:
            with open(os.path.join(settings.files_path, src), 'r', encoding='utf-8') as f:
                content = f.read()
            with folder.open(dst, 'w', encoding='utf-8') as f:
                f.write(content)

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = cmdline

        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.retrieve_list = ['output.json']
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = [
            (f.uuid, f.filename, f.filename) for f in self.inputs.file.values()
        ]
        return calcinfo
