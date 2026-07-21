import json
from aiida.common import exceptions
from aiida.parsers import Parser
from aiida.orm import Dict


class NanoParticleParser(Parser):
    """Parser for NanoParticle: output.json -> output_dict.

    NOT currently registered in setup.json (the workchain submits the
    calculation with the generic registered ``sqs_parser``, which has the
    same output.json -> Dict behaviour). Kept in the canonical shape so a
    ``nano_particles_parser`` entry point can be added later without code
    changes here.
    """

    def parse(self, **kwargs):
        try:
            retrieved_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        output_filename = "output.json"
        if output_filename not in retrieved_folder.list_object_names():
            return self.exit_codes.ERROR_MISSING_OUTPUT
        with retrieved_folder.open(output_filename, "r") as f:
            data = json.load(f)

        if "results" not in data:
            return self.exit_codes.ERROR_OUTPUT_INCOMPLETE

        self.out("output_dict", Dict(dict=data))
