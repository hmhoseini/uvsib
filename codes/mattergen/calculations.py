from aiida.engine import CalcJob
from aiida.orm import Dict
from aiida.common.datastructures import CalcInfo, CodeInfo

class MatterGenCalculation(CalcJob):
    """AiiDA plugin for MatterGen."""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("parameters", valid_type=Dict)

        spec.output("output_dict",
                    valid_type=Dict,
                    required=True
        )
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
        cmdline_params = parameters['cmdline_params']

        # Code info
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = cmdline_params
        # Calc info.
        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.retrieve_list = ['generated_crystals.extxyz']
        calcinfo.codes_info = [codeinfo]

        return calcinfo
