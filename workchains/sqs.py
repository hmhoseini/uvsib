"""User-facing SQS WorkChain.

Submits one SQS request (generate + MLIP-relax, see ``codes/files/sqs.py``)
to the wrapped CalcJob, then analyses the returned structures via
``workchains/sqs_analysis.py`` -- bulk convex hull, slab surface energies and
surface O-vacancy formation energies in one pass.

Inputs
------
- ``request``    (Dict)  : SQS request payload (parent_label, structure,
                            sublattices, composition_grid, supercell,
                            n_per_comp, sqs, surfaces, defects).
- ``mu_O2``      (Float, optional) : eV per O2 molecule, MLIP-relaxed
                            molecular reference. Required for absolute
                            O-vacancy formation energies; without it the
                            ``analysis["ovacancy"]`` entries fall back to
                            ``(E_vac - E_pristine) / n_vac`` and are tagged
                            ``reference="no O2 ref"``.
- ``functional`` (Str, optional)   : elemental reference set for the bulk
                            hull; defaults to ``settings.DFT_FUNC``.
- ``local_label`` (Str, optional)  : label propagated to the CalcJob node.

Outputs
-------
- ``analysis`` (Dict) : ``{"bulk": [...], "surface": [...], "ovacancy": [...]}``
"""
from aiida.engine import WorkChain
from aiida.plugins import WorkflowFactory, CalculationFactory
from aiida.orm import Str, Dict, List, Float

from uvsib.workchains.utils import get_code, get_model_device
from uvsib.workchains.sqs_analysis import (
    split_by_kind,
    analyse_bulk,
    analyse_surface,
    analyse_ovac,
)
from uvsib.workflows import settings


class SQSWorkChain(WorkChain):
    """Run one SQS request and build the bulk/slab/vacancy analysis."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("request", valid_type=Dict)
        spec.input("mu_O2", valid_type=Float, required=False)
        spec.input("local_label", valid_type=Str, required=False)

        spec.outline(
            cls.setup,
            cls.run_sqs,
            cls.inspect_sqs,
            cls.create_hull,
            cls.final_report,
        )

        spec.exit_code(300, "ERROR_SQS_FAILED", message="The SQS generation/relax calculation failed")
        spec.exit_code(400, "ERROR_ANALYSIS_FAILED", message="No usable structures returned from SQS")

    def setup(self):
        self.report("Running SQS WorkChain")
        self.ctx.success = False
        self.ctx.mu_O2 = (float(self.inputs.mu_O2.value) if "mu_O2" in self.inputs else None)
        self.ctx.local_label = self.inputs.local_label.value

    def run_sqs(self):
        builder = self._sqs_builder(self.inputs.request.get_dict())
        future = self.submit(builder)
        self.to_context(sqs=future)

    def inspect_sqs(self):
        wch = self.ctx.sqs
        if not wch.is_finished_ok:
            return self.exit_codes.ERROR_SQS_FAILED

        output_dict = wch.outputs.output_dict.get_dict()

        print('INSPECT SQS')
        print(output_dict)

        self.ctx.structures = output_dict.get("structures", [])
        self.ctx.metadata = output_dict.get("metadata", [])
        if not self.ctx.structures:
            return self.exit_codes.ERROR_ANALYSIS_FAILED
        self.report(f"SQS returned {len(self.ctx.structures)} relaxed structures")


    def create_hull(self):
        self.report("Building convex hull + surface analysis")
        bulk, slab, slab_vac = split_by_kind(self.ctx.structures, self.ctx.metadata)

        bulk_results = analyse_bulk(bulk, self.ctx.functional)
        surf_results = analyse_surface(slab, bulk_results)
        vac_results = analyse_ovac(slab_vac, surf_results, self.ctx.mu_O2)

        analysis = {
            "bulk":     bulk_results,
            "surface":  surf_results,
            "ovacancy": vac_results,
            "summary":  {
                "n_bulk":     len(bulk_results),
                "n_surface":  len(surf_results),
                "n_ovacancy": len(vac_results),
                "n_ground_states": sum(1 for r in bulk_results if r["is_ground_state"]),
                "functional": self.ctx.functional,
                "mu_O2_eV":   self.ctx.mu_O2,
            },
        }

        self.ctx.success = True

    def final_report(self):
        label = self.ctx.local_label
        if self.ctx.success:
            self.report(f"SQSWorkChain for {label} completed")
        else:
            self.report(f"SQSWorkChain for {label} failed")

    # ------------------------------------------------------------------
    # builder
    # ------------------------------------------------------------------

    def _sqs_builder(self, request: dict):
        ML_model = settings.inputs["SQS"]["model"]
        Workflow = WorkflowFactory("sqs_calc")
        builder = Workflow.get_builder()

        # sqs_calc is the BaseRestartWorkChain wrapping the SQS CalcJob; its
        # CalcJob writes the supplied list to input_structures.json verbatim,
        # so wrap the single request in a one-element list.
        builder.input_structure = List(list=[request])
        builder.code = get_code(ML_model)
        builder.local_label = Str(self.ctx.local_label)

        model, model_path, device = get_model_device(ML_model)
        job_info = {
            "job_type":   "sqs",
            "ML_model":   ML_model,
            "model_name": model,
            "model_path": model_path,
            "model_head": settings.inputs["SQS"]["head"],
            "device":     device,
            "fmax":       settings.inputs["SQS"]["fmax"],
            "max_steps":  settings.inputs["SQS"]["max_steps"],
        }
        builder.job_info = Dict(dict=job_info)
        return builder
