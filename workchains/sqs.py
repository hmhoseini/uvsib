import re
from aiida.engine import WorkChain
from aiida.plugins import WorkflowFactory
from aiida.orm import Str, List, Dict, Int
from pymatgen.core import Structure, Composition
# from uvsib.db.utils import add_similarities
from uvsib.workchains.utils import get_code, get_model_device
from uvsib.workflows import settings


class SQSWorkChain(WorkChain):
    """Similarity WorkChain"""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=List)

        spec.outline(
            cls.setup,
            cls.run_sqs,
            cls.inspect_sqs,
            cls.create_hull,
            cls.final_report
        )

        spec.exit_code(300,"ERROR_SQS_FAILED","The crystal structure prediction failed")
        spec.exit_code(400,"ERROR_SQS_FAILED", "The similarity comparison computations failed")

    def setup(self):
        """Setup and report"""
        self.report("Running SQS WorkChain")
        self.ctx.input_structure = self.ctx.inputs.structure
        self.ctx.generated_structures = list()
        self.ctx.success = False

    def run_sqs(self):
        request = {
              "parent_label": "Y2Ru2O7-pyrochlore",
              "structure":   {},  #  <pymatgen Structure.as_dict()>,   # parent unit cell
              "sublattices": {
                # icet labels sublattices 'A','B',... in order of distinct
                # allowed-species sets it encounters. Use the SAME labels here
                # and in composition_grid below.
                "A": {"sites": "Y",  "species": ["Y", "Gd", "Bi"]},
                "B": {"sites": "Ru", "species": ["Ru", "Ir", "Os"]}
              },
              "composition_grid": {
                "A": [{"Y": 1.0},
                      {"Y": 0.5, "Gd": 0.5},
                      {"Bi": 0.5, "Y": 0.5}],
                "B": [{"Ru": 1.0},
                      {"Ru": 0.5, "Ir": 0.5},
                      {"Ir": 1.0}]
              },
              "supercell":   [2, 2, 2],   # max search size (volume = product)
              "n_per_comp":  3,           # SQS samples per composition (different seeds)
              "sqs": {
                "cutoffs":  [7.0, 4.0],   # pair, triplet cutoffs in A (icet)
                "n_steps":  5000,         # MC steps per SQS
                "T_start":  5.0,
                "T_stop":   0.001,
                "include_smaller_cells": False
              },
              "surfaces": [               # OPTIONAL: slab variants to build per SQS
                {"miller": [1,1,1], "min_thickness": 10.0, "vacuum": 15.0}
              ],
              "defects": {                # OPTIONAL: surface defects (experimentally
                                          # relevant for OER, HER, CO2RR active sites)
                "oxygen_vacancy": {
                  "concentrations": [0.0, 0.125, 0.25],  # fraction of surface O
                  "n_per_conc":     2
                }
              }
            }

        builder = self._sqs_builder(request)
        future = self.submit(builder)
        self.to_context(**{f"sqs_": future})

    def inspect_sqs(self):
        failed_jobs = 0
        wch = self.ctx[f"sqs_"]
        if not wch.is_finished_ok:
            failed_jobs += 1
        self.ctx.generated_structures.extend(wch.called[-1].outputs.output_dict["structures"])
        if not self.ctx.generated_structures:
            return self.exit_codes.ERROR_SQS_FAILED

    def create_hull(self):
        print('created {}'.format(len(self.ctx.generated_structures)))
        self.report('Creating convex hulls...')

        assert 1 == 2


    def final_report(self):
        if self.ctx.success:
            self.report(f"SQSWorkChain for {self.ctx.chemical_formula} completed")
        else:
            self.report(f"SQSWorkChain for {self.ctx.chemical_formula} failed")

    def _sqs_builder(self, request: dict):
        ML_model = settings.inputs['SQS']['model']
        Workflow = WorkflowFactory("sqs_calc")
        builder = Workflow.get_builder()
        builder.request = request
        builder.code = get_code(ML_model)
        model, model_path, device = get_model_device(ML_model)

        job_info = {
            "job_type": "sqs",
            "ML_model": ML_model,
            "model_name": model,
            "model_path": model_path,
            "model_head": settings.inputs["SQS"]["head"],
            "device": device,
            "fmax": settings.inputs["SQS"]["fmax"],
            "max_steps": settings.inputs["SQS"]["max_steps"]
        }

        builder.job_info = Dict(job_info)
        # builder.max_iterations = Int(2)
        return builder
