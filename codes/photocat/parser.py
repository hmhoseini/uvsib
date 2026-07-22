import json
from aiida.common import exceptions
from aiida.parsers import Parser
from aiida.orm import Dict


class PhotocatParser(Parser):
    """Parser for PhotocatCalculation (output.json -> output_dict)."""

    def parse(self, **kwargs):
        try:
            retrieved_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        for name in ('output.json', 'total.txt', 'failed.txt'):
            if name not in retrieved_folder.list_object_names():
                return self.exit_codes.ERROR_MISSING_OUTPUT

        with retrieved_folder.open('output.json', 'r') as f:
            data = json.load(f)

        self.out("output_dict", Dict(dict=data))
