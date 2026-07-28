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
from uvsib.codes.utils import get_mp_element_structures, get_mp_experimental_structures
from uvsib.db.utils import add_structures
from uvsib.workchains.exp_include import split_output_slices
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
        """One MLIP relax over the CSP structures, bundled with (a) the
        experimentally-known MP structures for this exact formula (injected
        VERBATIM -- the anti-lottery guarantee that the real polymorphs are in
        the pool and relaxed on-method) and (b) any missing elemental
        references. Bundle order: generated | injected | references;
        collect_ml_energies peels the groups apart by input index."""
        model = settings.inputs['bulk_relax']['model']
        elements = Composition(self.ctx.chemical_formula).chemical_system.split('-')
        missing = missing_element_references(elements, model, head=settings.inputs['bulk_relax'].get('head'))
        ref_structs = []
        if missing:
            structs = get_mp_element_structures(missing)
            ref_structs = list(structs.values())
            if not structs:
                self.report(f"Warning: no MP elemental structures for {missing}; "
                            "hull will fall back to DFT references")

        exp_structs = []
        if settings.MP_EXP_INJECT:
            exp = get_mp_experimental_structures(self.ctx.chemical_formula,
                                                 cap=settings.MP_EXP_CAP,
                                                 mode="formula")
            exp_structs = [e["structure"] for e in exp]
            if exp:
                n_exp = sum(1 for e in exp if not e["theoretical"])
                self.report(f"Injecting {len(exp)} MP structure(s) for "
                            f"{self.ctx.chemical_formula} ({n_exp} experimental"
                            f"{'' if n_exp == len(exp) else ', theoretical fallback'}): "
                            + ", ".join(e["mp_id"] for e in exp))
            else:
                self.report("Warning: MP experimental injection returned "
                            "nothing (no entries or MP unreachable)")

        self.ctx.n_gen = len(self.ctx.csp_structures)
        self.ctx.n_exp = len(exp_structs)
        self.ctx.n_ref = len(ref_structs)
        builder = self._construct_ML_relax_builder(
            self.ctx.csp_structures + exp_structs + ref_structs)
        self.to_context(**{"ml_e": self.submit(builder)})

    def collect_ml_energies(self):
        """Split the bundled relax (generated | injected | references): store
        the references and the injected experimental structures (source
        'mp_experimental' -- the phase diagram force-includes them), then hull
        the CSP structures."""
        wch = self.ctx.ml_e
        if not wch.is_finished_ok:
            return self.exit_codes.ERROR_ML_RELAX_FAILED
        try:
            new_entries, exp_entries, ref_entries = split_output_slices(
                wch.outputs.output_dict.get_dict(),
                [self.ctx.n_gen, self.ctx.n_exp, self.ctx.n_ref])
        except Exception:
            return self.exit_codes.ERROR_ML_RELAX_FAILED

        model = settings.inputs['bulk_relax']['model']
        if ref_entries:
            pairs = [(e.structure.as_dict(), e.energy) for e in ref_entries]
            add_structures("reference", model, pairs,
                               head=settings.inputs['bulk_relax'].get('head'))
            self.report(f"Stored {len(pairs)} MLIP elemental reference(s)")

        if exp_entries:
            pairs = [(e.structure.as_dict(), e.energy) for e in exp_entries]
            add_structures("mp_experimental", model, pairs,
                           head=settings.inputs['bulk_relax'].get('head'))
            self.report(f"Stored {len(pairs)} on-method MP-experimental "
                        "structure(s)")
        elif self.ctx.n_exp:
            self.report(f"Warning: none of the {self.ctx.n_exp} injected MP "
                        "structures survived the relax (fail-loudly: check "
                        "the relax job)")

        self.ctx.el_entries, missing = element_reference_entries(
            Composition(self.ctx.chemical_formula).chemical_system.split('-'), model,
            head=settings.inputs['bulk_relax'].get('head'))
        if missing:
            self.report(f"Warning: DFT-fallback elemental refs for {missing} (per-element offset risk)")

        self.ctx.low_energy_entries_csp, _ = unique_low_energy_comp(self.ctx.chemical_formula, new_entries, DFT_FUNC,
                                                                    EHULL_ML, min_n_return=self.ctx.n_mh,
                                                                    element_entries=self.ctx.el_entries)

    def minimahopping(self):
        """Run MinimaHopping (skipped for excluded-element compositions)"""
        excluded = sorted(settings.MH_EXCLUDE_ELEMENTS.intersection(
            Composition(self.ctx.chemical_formula).chemical_system.split('-')))
        if excluded:
            self.report(f"Skipping MinimaHopping for {self.ctx.chemical_formula}: "
                        f"contains excluded element(s) {', '.join(excluded)} "
                        "(MinimaHopping.exclude_elements in input.yaml)")
            self.ctx.mh_skipped = True
            return
        self.ctx.mh_skipped = False
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
        if self.ctx.get('mh_skipped'):
            self.ctx.low_energy_entries_mh = []
            return
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

        # NOTE: pool ALL jobs' minima -- this used to pass `new_entries` (the
        # leftover of the last loop iteration), silently dropping every MH job
        # but the last (and raising UnboundLocalError if the first job failed).
        self.ctx.low_energy_entries_mh, _ = unique_low_energy_comp(self.ctx.chemical_formula, all_entries,
                                                                   DFT_FUNC, EHULL_ML, element_entries=self.ctx.el_entries)

    def final_step(self):
        """Store structures"""
        all_entries = self.ctx.low_energy_entries_csp + self.ctx.low_energy_entries_mh
        low_energy_entries, _ = unique_low_energy_comp(self.ctx.chemical_formula, all_entries, DFT_FUNC, EHULL_ML,
                                                       element_entries=self.ctx.el_entries)
        structure_energy_pairs = []

        for entry in low_energy_entries:
            structure_energy_pairs.append((entry.structure.as_dict(), entry.energy))

        # pooled relax+MH entries: head is only well-defined when both stages
        # ran the same head; otherwise store head-less (and say so)
        _bulk_head = settings.inputs['bulk_relax'].get('head')
        _mh_head = settings.inputs.get('MinimaHopping', {}).get('head')
        _pool_head = _bulk_head if (self.ctx.get('mh_skipped') or _mh_head == _bulk_head) else None
        if _pool_head is None:
            self.report(f"csp pool mixes heads (bulk_relax '{_bulk_head}' vs "
                        f"MinimaHopping '{_mh_head}'): storing WITHOUT head provenance")
        add_structures("csp", settings.inputs['bulk_relax']['model'], structure_energy_pairs,
                       head=_pool_head)

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
