import json
from aiida.common import exceptions
from aiida.parsers import Parser
from aiida.orm import Dict


class PrecursorSearchParser(Parser):
    """Parser for :class:`PrecursorSearchCalculation`.

    Reads the agent's ``output.json`` (the DOIs + synthesis routes) from the
    retrieved folder and wraps it as the ``output_dict`` Dict node -- the same
    output contract as the GNoME/MatterGen parsers, so downstream code consumes
    it the same way (``node.outputs.output_dict.get_dict()``).
    """

    def parse(self, **kwargs):
        try:
            retrieved = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        # honor a custom output filename if one was set on the CalcJob inputs
        filename = "output.json"
        try:
            params = self.node.inputs.parameters.get_dict()
            filename = params.get("output_filename", filename)
        except (AttributeError, exceptions.NotExistent):
            pass

        if filename not in retrieved.list_object_names():
            return self.exit_codes.ERROR_MISSING_OUTPUT

        try:
            with retrieved.open(filename) as handle:
                payload = json.load(handle)
        except (ValueError, json.JSONDecodeError):
            return self.exit_codes.ERROR_INVALID_OUTPUT

        # minimal shape check: a dict carrying a results list
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            return self.exit_codes.ERROR_OUTPUT_INCOMPLETE

        payload.setdefault("n_results", len(payload["results"]))
        self.out("output_dict", Dict(dict=payload))
