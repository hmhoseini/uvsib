import json
from aiida.common import exceptions
from aiida.parsers import Parser
from aiida.orm import Dict


class GNoMEParser(Parser):
    """Parser for GNoMECalculation.

    Reads ``output.json`` (a list of pymatgen structure dicts) from the
    retrieved folder and wraps it as ``Dict(structures=[...])`` -- identical to
    the MatterGen output contract, so the same downstream relax/filter/store
    path consumes it.
    """

    def parse(self, **kwargs):
        try:
            retrieved_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        filename = 'output.json'
        if filename not in retrieved_folder.list_object_names():
            return self.exit_codes.ERROR_MISSING_OUTPUT

        with retrieved_folder.open(filename) as f:
            pmg_structures = json.load(f)

        if not pmg_structures:
            return self.exit_codes.ERROR_OUTPUT_INCOMPLETE

        self.out("output_dict", Dict(dict={"structures": pmg_structures}))
