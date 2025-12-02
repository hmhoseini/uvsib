import json
from aiida.common import exceptions
from aiida.parsers import Parser
from aiida.orm import Dict

class MatterGenParser(Parser):
    """
    Parser for MatterGenCalculation.

    Reads generated_crystals.extxyz directly from the retrieved folder
    into memory and converts all frames to StructureData.
    """

    def parse(self, **kwargs):
        try:
            retrieved_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        filename = 'output.json'
        if filename not in retrieved_folder.list_object_names():
            return self.exit_codes.ERROR_MISSING_OUTPUT

        pmg_structures = []
        with retrieved_folder.open(filename) as f:
            pmg_structures = json.loads(f.read())

        if not pmg_structures:
            return self.exit_codes.ERROR_OUTPUT_INCOMPLETE

        output_structures = {"structures": pmg_structures}
        self.out("output_dict", Dict(dict=output_structures))
