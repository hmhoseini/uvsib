import os
from aiida.engine import CalcJob
from aiida.orm import Dict, SinglefileData
from aiida.common.datastructures import CalcInfo, CodeInfo
from uvsib.workflows import settings

class MatterSimCalculation(CalcJob):
    """AiiDA plugin for MatterSim"""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("parameters", valid_type=Dict)
        spec.input_namespace("file",
                             valid_type=(SinglefileData),
                             dynamic=True
        )
        spec.output("output_dict",
                    valid_type=Dict,
                    required=True)

        spec.exit_code(
                100,
                "ERROR_MISSING_OUTPUT",
                message="Required output file not found."
        )
        spec.exit_code(
                200,
                "ERROR_NO_RETRIEVED_FOLDER",
                message="The retrieved folder data node can not be accessed."
        )
        spec.exit_code(
                303, "ERROR_OUTPUT_INCOMPLETE",
                message="The output file is incomplete."
        )

    def prepare_for_submission(self, folder):
        """Create input files for MatterGen. Here, adding to the command line"""
        parameters = self.inputs.parameters.get_dict()
        job_type = parameters['job_type']
        cmdline = parameters['cmdline_params']

        if job_type == 'relax':
            input_file = os.path.join(settings.mattersim_files_path, 'relax.py')
        elif job_type == 'facebuild':
            input_file = os.path.join(settings.mattersim_files_path, 'face_build.py')
        elif job_type == 'adsorbates':
            input_file = os.path.join(settings.mattersim_files_path, 'adsorbates.py')
        else:
            input_file = os.path.join(settings.mattersim_files_path, 'energy_forces.py')

        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        with folder.open('aiida.py', 'w', encoding='utf-8') as f:
            f.write(content)

        # Code info
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = cmdline
        # Calc info.
        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.retrieve_list = ['output.json', 'total.txt', 'failed.txt']
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = [
            (file.uuid, file.filename, file.filename)
            for file in self.inputs.file.values()
        ]
        calcinfo.provenance_exclude_list = ['input_structures.extxyz']

        return calcinfo
