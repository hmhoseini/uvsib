"""AiiDA CalcJob for the containerized precursor-search agent.

This plugin launches a *self-contained* container (provided as the registered
``code``) whose job is to web-search recent publications for *proven* ways to
synthesize a target material and return the DOIs and synthesis paths it found.

The plugin itself is deliberately thin and container-engine agnostic: register
the agent as a normal ``Code`` or a ``ContainerizedCode`` (apptainer / singularity
/ docker / podman) -- AiiDA composes the container invocation; this CalcJob only
stages the JSON *request*, appends the cmdline, and retrieves the JSON *response*.
The whole search/agent logic lives inside the image, so no runner script is
staged from ``settings.files_path`` (unlike the GNoME/MatterGen CalcJobs).

I/O contract (kept minimal so the container is the only moving part)
-------------------------------------------------------------------

``request.json``  -- staged into the run dir from the ``request`` input::

    {
      "chemical_formula":  "Y2Ru2O7",
      "reduced_formula":   "Y2Ru2O7",
      "elements":          ["Y", "Ru", "O"],
      "candidates":        [{"uuid": "...", "formula": "Y2Ru2O7",
                             "label": "synthesizable", "score": 0.81,
                             "ehull_eV_per_atom": 0.0}, ...],
      "max_results":       20,
      "since_year":        2015,
      "include_preprints": true,
      "methods":           ["solid-state", "sol-gel", "hydrothermal", "flux", "cvd"]
    }

``output.json``  -- written by the agent in the run dir; retrieved + parsed::

    {
      "formula": "Y2Ru2O7",
      "results": [
        {"doi": "10.1021/jacs.xxxxx", "title": "...", "year": 2021,
         "url": "https://doi.org/10.1021/jacs.xxxxx",
         "synthesis_routes": [
            {"method": "solid-state",
             "precursors": ["Y2O3", "RuO2"],
             "steps": ["mix stoichiometric powders", "calcine 1000 C / 24 h / air"],
             "conditions": {"temperature_C": 1000, "time_h": 24, "atmosphere": "air"},
             "product": "Y2Ru2O7",
             "confidence": 0.82,
             "evidence": "verbatim passage supporting the route"}
         ]}
      ],
      "n_results": 1,
      "agent": {"model": "...", "search_provider": "...", "version": "..."}
    }

The matching parser (``precursor_search_parser``) wraps ``output.json`` into the
``output_dict`` Dict node -- the same output-node name as the GNoME/MatterGen
CalcJobs, so the downstream wiring is uniform.

Secrets
-------
The LLM API key must reach the *container on the remote node* and MUST NOT enter
the provenance graph. It is therefore injected at job runtime by the registered
``ContainerizedCode`` -- preferably via the engine command's ``--env-file``
pointing at a 0600 file on the node (udocker), or a Code/Computer ``prepend_text``
that reads such a file (apptainer). It is NOT a CalcJob input and NOT a
``metadata.options.environment_variables`` entry (both would be serialized onto
the node), and NOT baked into the shipped image. Note that ``input.yaml`` /
``config.yaml`` are read on the *submitter* only, so they cannot deliver the key
to the remote container. See ``container/README.md`` -> *Secrets & network*.
"""

from aiida.engine import CalcJob
from aiida.orm import Dict, SinglefileData
from aiida.common.datastructures import CalcInfo, CodeInfo


class PrecursorSearchCalculation(CalcJob):
    """Run the containerized literature/web-search agent for synthesis routes."""

    _DEFAULT_REQUEST_FILENAME = "request.json"
    _DEFAULT_OUTPUT_FILENAME = "output.json"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "request", valid_type=SinglefileData,
            help="JSON search request; staged into the run dir as request.json.",
        )
        spec.input(
            "parameters", valid_type=Dict, required=False,
            help="Optional overrides: cmdline_params (list[str]), request_filename, "
                 "output_filename, retrieve_list (list[str]).",
        )
        spec.output(
            "output_dict", valid_type=Dict, required=True,
            help="Parsed agent response: DOIs + synthesis routes.",
        )
        # default parser; still overridable via metadata.options.parser_name
        spec.inputs["metadata"]["options"]["parser_name"].default = "precursor_search_parser"

        spec.exit_code(100, "ERROR_MISSING_OUTPUT",
                       message="The agent output file was not found in the retrieved folder.")
        spec.exit_code(200, "ERROR_NO_RETRIEVED_FOLDER",
                       message="The retrieved folder data node can not be accessed.")
        spec.exit_code(301, "ERROR_INVALID_OUTPUT",
                       message="The agent output file is not valid JSON.")
        spec.exit_code(303, "ERROR_OUTPUT_INCOMPLETE",
                       message="The agent output file is missing required fields.")

    def prepare_for_submission(self, folder):
        params = self.inputs.parameters.get_dict() if "parameters" in self.inputs else {}
        request_name = params.get("request_filename", self._DEFAULT_REQUEST_FILENAME)
        output_name = params.get("output_filename", self._DEFAULT_OUTPUT_FILENAME)

        # The agent gets one input file (the request) and writes one output file.
        # Default cmdline points the container at both by name; override via
        # parameters['cmdline_params'] if the image expects a different CLI.
        cmdline = params.get("cmdline_params")
        if cmdline is None:
            cmdline = [f"--request={request_name}", f"--output={output_name}"]

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = cmdline

        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.codes_info = [codeinfo]
        # copy the request node into the run dir under the agreed name (no
        # bytes pass through the daemon process -- AiiDA's transport handles it)
        calcinfo.local_copy_list = [
            (self.inputs.request.uuid, self.inputs.request.filename, request_name)
        ]
        calcinfo.retrieve_list = params.get("retrieve_list") or [output_name]
        return calcinfo
