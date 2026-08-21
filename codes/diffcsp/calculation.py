import os
from aiida.engine import CalcJob
from aiida.orm import Dict
from aiida.common.datastructures import CalcInfo, CodeInfo
from uvsib.workflows import settings

# runner + sidecar staged next to aiida.py on the compute node
_STAGED = [
    ("diffcsp_csp.py", "aiida.py"),  # the runner becomes aiida.py
    ("refine.py", "refine.py"),      # charge-neutral + primitive dedup (shared w/ MatterGen/GNoME)
]

class DiffCSPCalculation(CalcJob):
    """AiiDA plugin for DiffCSP (https://github.com/jiaor17/DiffCSP).

    Stages the runner (``diffcsp_csp.py`` -> ``aiida.py``) and its sidecars,
    runs it, and retrieves ``output.json`` -- the same output contract as
    MatterGen/GNoME, so the same downstream relax/filter/store path consumes
    it.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("parameters", valid_type=Dict)
        spec.output("output_dict", valid_type=Dict, required=True)
        spec.exit_code(100, "ERROR_MISSING_OUTPUT", message="Required output file not found.")
        spec.exit_code(200, "ERROR_NO_RETRIEVED_FOLDER", message="The retrieved folder data node can not be accessed.")
        spec.exit_code(303, "ERROR_OUTPUT_INCOMPLETE", message="The output file is incomplete.")

    def prepare_for_submission(self, folder):
        """Stage the DiffCSP runner and sidecars. cmdline_params drive aiida.py."""
        parameters = self.inputs.parameters.get_dict()
        cmdline_params = parameters['cmdline_params']

        for src, dst in _STAGED:
            with open(os.path.join(settings.files_path, src), 'r', encoding='utf-8') as f:
                content = f.read()
            with folder.open(dst, 'w', encoding='utf-8') as f:
                f.write(content)

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = cmdline_params

        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.retrieve_list = ['output.json']
        calcinfo.codes_info = [codeinfo]

        return calcinfo
