import random
from aiida.engine import WorkChain
from aiida.orm import Str, List, Dict, Int
from aiida.plugins import DataFactory, WorkflowFactory
from pymatgen.core import Composition
from uvsib.workchains.utils import (
        unique_low_energy_comp,
        get_output_as_entry,
        get_code,
        get_model_device,
        element_reference_entries,
        missing_element_references,
        split_relax_output)
from uvsib.codes.utils import get_mp_element_structures
from uvsib.db.utils import add_structures
from uvsib.workflows import settings


StructureData = DataFactory('core.structure')

DFT_FUNC = settings.DFT_FUNC
EHULL_ML = settings.EHULL_ML

class CSPWorkChain(WorkChain):
    """WorkChain for Crystal Structure Prediction (CSP)"""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)
        spec.input("n_csp", valid_type=Int)
        spec.input("n_mh", valid_type=Int)

        spec.outline(
            cls.setup,
            cls.run_csp,
            cls.inspect_csp_calcs,
            cls.predict_ml_energies,
            cls.collect_ml_energies,
            cls.minimahopping,
            cls.mh_energies,
            cls.final_step,
            cls.final_report
        )

        spec.exit_code(301, "ERROR_CSP_FAILED", message="CSP calculations failed")
        spec.exit_code(302, "ERROR_ML_RELAX_FAILED", message="ML relaxation failed")
        spec.exit_code(303, "ERROR_MINIMAHOPPING_FAILED", message="MinimaHopping calculations failed")

    def setup(self):
        """Setup and report"""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.n_csp = self.inputs.n_csp.value
        self.ctx.n_mh = self.inputs.n_mh.value
        self.ctx.csp_structures = []
#        self.ctx.inputs = {"metadata": {"label": "CSP for {}".format(self.ctx.chemical_formula)}}
        self.report(f"Launching CSPWorkChain for {self.ctx.chemical_formula}")

    def run_csp(self):
        """Run MatterGen CSP and/or GNoME (SAPS) CSP in parallel, per input.yaml
        toggles (`mattergen.enabled`, `gnome.enabled`); at least one required."""
        self.ctx.n_csp_jobs = 0
        if settings.MATTERGEN_ENABLED:
            self.ctx.n_csp_jobs = self.ctx.n_csp
            for i in range(1, self.ctx.n_csp + 1):
                builder = self._construct_mattergen_csp_builder()
                future = self.submit(builder)
                self.to_context(**{f"csp_{i}": future})

        self.ctx.n_gnome = 0
        if settings.GNOME_PARALLEL:
            self.ctx.n_gnome = int(settings.inputs.get("GNoME_CSP", {}).get("num_runs", 1))
            for i in range(1, self.ctx.n_gnome + 1):
                gbuilder = self._construct_gnome_csp_builder()
                self.to_context(**{f"gnome_{i}": self.submit(gbuilder)})

        if self.ctx.n_csp_jobs == 0 and self.ctx.n_gnome == 0:
            self.report("No CSP generator enabled (mattergen + gnome both off)")
            return self.exit_codes.ERROR_CSP_FAILED

    def inspect_csp_calcs(self):
        """Collect structures from MatterGen (primary) and GNoME (best-effort) CSP"""
        failed_jobs = 0
        for i in range(1, self.ctx.n_csp_jobs + 1):
            csp_wch = self.ctx[f"csp_{i}"]
            if not csp_wch.is_finished_ok:
                failed_jobs += 1
                continue
            try:
                self.ctx.csp_structures.extend(csp_wch.outputs.output_dict["structures"])
            except:
                failed_jobs += 1

        for i in range(1, self.ctx.n_gnome + 1):
            gnome_wch = self.ctx[f"gnome_{i}"]
            if not gnome_wch.is_finished_ok:
                self.report(f"Warning: GNoME CSP job {i} failed; continuing with MatterGen")
                continue
            try:
                self.ctx.csp_structures.extend(gnome_wch.outputs.output_dict["structures"])
            except Exception:
                self.report(f"Warning: could not read GNoME CSP structures from job {i}")

        if not self.ctx.csp_structures:
            self.report("No structure was found")
            return self.exit_codes.ERROR_CSP_FAILED

        if self.ctx.n_csp_jobs and failed_jobs / self.ctx.n_csp_jobs > 0.5:
            self.report("Many CSP jobs failed")
            return self.exit_codes.ERROR_CSP_FAILED

    def predict_ml_energies(self):
        """One MLIP relax over the CSP structures, bundled with any missing
        elemental references (from MP) so the model is loaded once instead of in
        a separate reference CalcJob. The references are appended after the CSP
        structures; collect_ml_energies peels them back off by input index."""
        model = settings.inputs['bulk_relax']['model']
        elements = Composition(self.ctx.chemical_formula).chemical_system.split('-')
        missing = missing_element_references(elements, model)
        ref_structs = []
        if missing:
            structs = get_mp_element_structures(missing)
            ref_structs = list(structs.values())
            if not structs:
                self.report(f"Warning: no MP elemental structures for {missing}; "
                            "hull will fall back to DFT references")
        self.ctx.n_main = len(self.ctx.csp_structures)
        builder = self._construct_ML_relax_builder(self.ctx.csp_structures + ref_structs)
        self.to_context(**{"ml_e": self.submit(builder)})

    def collect_ml_energies(self):
        """Split the bundled relax: store the elemental references, then hull the
        CSP structures (references stored first so the hull picks them up)."""
        wch = self.ctx.ml_e
        if not wch.is_finished_ok:
            return self.exit_codes.ERROR_ML_RELAX_FAILED
        try:
            new_entries, ref_entries = split_relax_output(wch, self.ctx.n_main)
        except Exception:
            return self.exit_codes.ERROR_ML_RELAX_FAILED

        model = settings.inputs['bulk_relax']['model']
        if ref_entries:
            pairs = [(e.structure.as_dict(), e.energy) for e in ref_entries]
            add_structures("reference", model, pairs)
            self.report(f"Stored {len(pairs)} MLIP elemental reference(s)")

        self.ctx.el_entries, missing = element_reference_entries(
            Composition(self.ctx.chemical_formula).chemical_system.split('-'), model)
        if missing:
            self.report(f"Warning: DFT-fallback elemental refs for {missing} (per-element offset risk)")

        self.ctx.low_energy_entries_csp, _ = unique_low_energy_comp(self.ctx.chemical_formula, new_entries, DFT_FUNC,
                                                                    EHULL_ML, min_n_return=self.ctx.n_mh,
                                                                    element_entries=self.ctx.el_entries)

    def minimahopping(self):
        """Run MinimaHopping"""
        entries_csp = self.ctx.low_energy_entries_csp
        n_mh = min(len(entries_csp), self.ctx.n_mh)
        selected_entries = random.sample(entries_csp, n_mh)
        for i, entry in enumerate(selected_entries):
            struct = StructureData(pymatgen_structure = entry.structure)
            builder = self._construct_mh_builder(struct)
            future = self.submit(builder)
            self.to_context(**{f"mh_{i}": future})

    def mh_energies(self):
        """Predict ML energies"""
        n_mh = min(len(self.ctx.low_energy_entries_csp), self.ctx.n_mh)
        all_entries = []
        failed_jobs = 0
        for i in range(n_mh):
            wch = self.ctx[f"mh_{i}"]
            if not wch.is_finished_ok:
                failed_jobs += 1
                continue
            try:
                new_entries = get_output_as_entry(wch)
                all_entries.extend(new_entries)
            except:
                failed_jobs += 1

        if not all_entries or failed_jobs / n_mh > 0.5:
            return self.exit_codes.ERROR_MINIMAHOPPING_FAILED

        self.ctx.low_energy_entries_mh, _ = unique_low_energy_comp(self.ctx.chemical_formula, new_entries,
                                                                   DFT_FUNC, EHULL_ML, element_entries=self.ctx.el_entries)

    def final_step(self):
        """Store structures"""
        all_entries = self.ctx.low_energy_entries_csp + self.ctx.low_energy_entries_mh
        low_energy_entries, _ = unique_low_energy_comp(self.ctx.chemical_formula, all_entries, DFT_FUNC, EHULL_ML,
                                                       element_entries=self.ctx.el_entries)
        structure_energy_pairs = []

        for entry in low_energy_entries:
            structure_energy_pairs.append((entry.structure.as_dict(), entry.energy))

        add_structures("csp", settings.inputs['bulk_relax']['model'], structure_energy_pairs)

    def final_report(self):
        """Final report"""
        self.report(f"CSPWorkChain for {self.ctx.chemical_formula} finished successfully")

####################################################################
    def _construct_mattergen_csp_builder(self):
        Workflow = WorkflowFactory("mattergen.csp")
        builder = Workflow.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.code = get_code("MatterGen_CSP")
        _, model_path, _ = get_model_device("MatterGen_CSP")
        builder.job_info = Dict(
            {
                "job_type": "csp",
                "model_path": model_path,
                "batch_size": settings.inputs["MatterGen_CSP"]["batch_size"],
                "num_batches": settings.inputs["MatterGen_CSP"]["num_batches"],
            }
        )
        # builder.max_iterations = Int(2)
        return builder

    def _construct_gnome_csp_builder(self):
        """GNoME (SAPS) CSP builder, parallel to MatterGen's csp branch."""
        Workflow = WorkflowFactory("gnome.csp")
        builder = Workflow.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.code = get_code("GNoME")

        ji = dict(settings.inputs["GNoME_CSP"])
        ji["model_head"] = ji.get("head")
        model, model_path, device = get_model_device(ji['screen'])
        ji.update({"model_name": model, "model_path": model_path, "device": device})

        builder.job_info = Dict(ji)
        # builder.max_iterations = Int(2)
        return builder

    def _construct_ML_relax_builder(self, structures):
        """
        General builder for structure optimization with an ML model
        """
        ML_model = settings.inputs['bulk_relax']['model']
        Workflow = WorkflowFactory(ML_model.lower())
        builder = Workflow.get_builder()
        builder.input_structures = List(structures)
        builder.code = get_code(ML_model)
        builder.local_label = Str("relax {}".format(self.ctx.chemical_formula))
        model, model_path, device = get_model_device(ML_model)

        job_info = {
            "job_type": "relax",
            "ML_model": ML_model,
            "model_name": model,
            "model_path": model_path,
            "model_head": settings.inputs["bulk_relax"]["head"],
            "device": device,
            "fmax": settings.inputs["bulk_relax"]["fmax"],
            "max_steps": settings.inputs["bulk_relax"]["max_steps"]
        }

        builder.job_info = Dict(job_info)
        return builder

    def _construct_mh_builder(self, struct):
        Workflow = WorkflowFactory("minimahopping")
        builder = Workflow.get_builder()
        builder.structure = struct
        builder.code = get_code("MinimaHopping")
        builder.this_label = '{}'.format(self.ctx.chemical_formula)
        model, model_path, device = get_model_device(settings.inputs["MinimaHopping"]["model"])

        job_info = {
            "job_type": "hopping",
            "ML_model": settings.inputs["MinimaHopping"]["model"],
            "model_name": model,
            "model_path": model_path,
            "model_head": settings.inputs["MinimaHopping"]["head"],
            "device": device,
            "mh_steps": settings.inputs["MinimaHopping"]["mh_steps"]
        }

        builder.job_info = Dict(job_info)
        return builder
