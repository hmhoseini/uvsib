from aiida.orm import Str, List, Dict
from aiida.plugins import WorkflowFactory
from aiida.engine import WorkChain, if_, while_
from aiida_pythonjob import PythonJob, prepare_pythonjob_inputs
from pymatgen.core.composition import Composition
from uvsib.db.tables import DBComposition, DBSurfaceMLAdsorbate, DBNanoParticles, DBBatteryPath, DBBatteryNEB
from uvsib.db.utils import update_row, query_by_columns, update_step_status_path
from uvsib.workchains.pythonjob_inputs import wait_sleep
from uvsib.workflows import settings

_SKIP_PD_VERIFICATION = settings._SKIP_PD_VERIFICATION  #TODO for test


class MainWorkChain(WorkChain):
    """ Main WorkChain"""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)
        spec.input("chemical_systems", valid_type=List)
        spec.input("reaction", valid_type=Str)
        spec.input("reaction_path", valid_type=Str)
        spec.input("nanoparticles", valid_type=Str)
        spec.input("similarities", valid_type=Dict)
        spec.input('sqs', valid_type=Dict)

        spec.outline(
            cls.setup,
            if_(cls.should_run_pd_ml)(
                while_(cls.should_wait_pd_ml)(
                    cls.wait_sleep,
                    cls.check_pythonjob_sleep
                ),
                cls.pd_ml,
                cls.inspect_pd_ml
            ),
            if_(cls.should_run_pd_verification)(
                while_(cls.should_wait_pd_ver)(
                    cls.wait_sleep,
                    cls.check_pythonjob_sleep
                ),
                cls.pd_verification,
                cls.inspect_pd_verification
            ),
            if_(cls.should_run_synthesizability)(
                while_(cls.should_wait_synthesizability)(
                    cls.wait_sleep,
                    cls.check_pythonjob_sleep
                ),
                cls.synthesizability,
                cls.inspect_synthesizability
            ),
            if_(cls.should_run_sqs)(
                while_(cls.should_wait_sqs)(
                    cls.wait_sleep,
                    cls.check_pythonjob_sleep
                ),
                cls.sqs,
                cls.inspect_sqs
            ),
            if_(cls.should_run_surface_builder)(
                while_(cls.should_wait_surface_builder)(
                    cls.wait_sleep,
                    cls.check_pythonjob_sleep
                ),
                cls.surface_builder,
                cls.inspect_surface_builder
            ),
            if_(cls.should_run_adsorbates)(
                while_(cls.should_wait_adsorbates)(
                    cls.wait_sleep,
                    cls.check_pythonjob_sleep
                ),
                cls.adsorbates,
                cls.inspect_adsorbates
            ),
            if_(cls.should_run_photocat)(
                while_(cls.should_wait_photocat)(
                    cls.wait_sleep,
                    cls.check_pythonjob_sleep
                ),
                cls.photocat,
                cls.inspect_photocat
            ),
            if_(cls.should_run_battery)(
                while_(cls.should_wait_battery)(
                    cls.wait_sleep,
                    cls.check_pythonjob_sleep
                ),
                cls.battery,
                cls.inspect_battery
            ),
            if_(cls.should_run_battery_neb)(
                while_(cls.should_wait_battery_neb)(
                    cls.wait_sleep,
                    cls.check_pythonjob_sleep
                ),
                cls.battery_neb,
                cls.inspect_battery_neb
            ),
            if_(cls.should_run_nano_particles)(
                while_(cls.should_wait_nano_particles)(
                    cls.wait_sleep,
                    cls.check_pythonjob_sleep
                ),
                cls.nano_particles,
                cls.inspect_nano_particles
            )
        )

        spec.exit_code(300,"ERROR_CALCULATION_FAILED", message="A sub-WorkChain did not finish successfully")

    def setup(self):
        """Setup and report"""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.chemical_systems = self.inputs.chemical_systems
        self.ctx.reaction = self.inputs.reaction
        self.ctx.reaction_path = self.inputs.reaction_path
        self.ctx.dbcomposition_row = query_by_columns(DBComposition,{"composition": self.ctx.chemical_formula})[0]
        self.ctx.nano_generator = True if len(self.inputs.nanoparticles.value.split('-')) == 2 else False
        self.ctx.similarities = self.inputs.similarities.value
        self.ctx.sqs_request = self.inputs.sqs.get_dict()
        self.ctx.sqs = bool(self.ctx.sqs_request)   # non-empty request -> SQS run
        # battery submissions carry the working ion in reaction_path and run
        # the bulk battery pathway INSTEAD of surface builder + adsorbates
        self.ctx.battery = self.ctx.reaction.value.strip().upper() == "BATTERY"
        if self.ctx.nano_generator:
            self.ctx.nano_particles_range = self.inputs.nanoparticles.value
            elements = '-'.join(sorted(list([str(el) for el in Composition(self.ctx.chemical_formula).elements])))
            self.ctx.nano_row = query_by_columns(DBNanoParticles,{'elements': elements})[0]
            self.report('Running NanoParticles for stored data')
        else:
            self.report(f"Running MainWorkChain for {self.ctx.chemical_formula}: "
                        f"reaction {self.ctx.reaction.value} path {self.ctx.reaction_path.value}")

    def _fresh_step_status(self):
        """Re-query the composition row so the shared-step gates observe a
        sibling chain's transitions.

        The row cached in setup() is a frozen SELECT * snapshot; reading it in a
        wait loop would never see a 'Running' -> 'Done' transition (the chain
        would wait forever) and would let a late chain miss a finished step
        instead of skipping it and reusing the result (e.g. the surfaces).
        Refreshing the cached row also gives the run/inspect writes a fresher
        base. Scoped to the composition-level steps; adsorbates re-queries its
        own per-(reaction, pathway) status.
        """
        row = query_by_columns(DBComposition,{"composition": self.ctx.chemical_formula})[0]
        self.ctx.dbcomposition_row = row
        return row.step_status

    @staticmethod
    def _ads_status(step_status, reaction, reaction_path):
        """Per-(reaction, pathway) adsorbates status from the nested step_status.
        ``step_status["adsorbates"]`` is keyed reaction -> reaction_path -> state so
        that catalyst chains running in parallel for one composition (e.g. CO2RR and
        NOXRR) do not share a single flat 'adsorbates' flag.
        """
        # return ((step_status or {}).get("adsorbates") or {}).get(reaction, {}).get(reaction_path)
        return ((step_status.get("adsorbates") or {}).get(reaction) or {}).get(reaction_path) or {}

    @staticmethod
    def _battery_status(step_status, working_ion):
        """Per-ion battery status from the nested step_status.
        ``step_status["battery"]`` is keyed by working ion so battery chains
        for different ions (Li, Na, ...) on one composition stay independent,
        mirroring the per-(reaction, pathway) adsorbates convention."""
        return ((step_status or {}).get("battery") or {}).get(working_ion) or {}

    def should_run_pd_ml(self):
        """Check whether should run PhaseDiagramML"""
        if self.ctx.sqs:
            return False
        if self.ctx.nano_generator:
            return False
        step_status = self._fresh_step_status().get("pd_ml")
        if step_status in ["Done"]:
            return False
        return True

    def should_run_pd_verification(self):
        """Check whether should run PDVerification"""
        if self.ctx.sqs:
            return False
        if self.ctx.nano_generator:
            return False
        if _SKIP_PD_VERIFICATION: #TODO remove after test
            return False
        step_status = self._fresh_step_status().get("pd_verification")
        if step_status in ["Done"]:
            return False
        return True

    def should_run_sqs(self):
        """Check whether should run PhaseDiagramML"""
        if not self.ctx.sqs:
            return False
        if self.ctx.nano_generator:
            return False
        step_status = self._fresh_step_status().get("sqs")
        if step_status in ["Done"]:
            return False
        return True

    def should_run_synthesizability(self):
        """Check whether should run SynthesizabilityWorkChain"""
        if not settings.SYNTH_ENABLED:
            return False
        if self.ctx.sqs:
            return False
        if self.ctx.nano_generator:
            return False
        step_status = self._fresh_step_status().get("synthesizability")
        if step_status in ["Done"]:
            return False
        return True

    def should_run_surface_builder(self):
        """Check whether should run SurfaceBuilder"""
        if settings.SOFT_STOP_BEFORE_SURFACE:
            self.report("Soft stop (soft_stop.before_surface_builder): ending before the "
                        "surface builder starts; generation/synthesizability stages are complete now.")
            return False
        if self.ctx.battery:
            return False    # bulk pathway: no slabs needed
        if self.ctx.nano_generator:
            return False
        surface_builder_step_status = self._fresh_step_status().get("surface_builder")
        if surface_builder_step_status in ["Done"]:
            return False
        return True

    def should_run_adsorbates(self):
        """Check whether should run Adsorbates for THIS (reaction, pathway)"""
        if settings.SOFT_STOP_BEFORE_SURFACE:
            return False
        if self.ctx.battery:
            return False
        if self.ctx.nano_generator:
            return False
        reaction = self.ctx.reaction.value
        reaction_path = self.ctx.reaction_path.value
        # Re-query for a fresh status: the setup-time row snapshot would not see
        # a sibling chain's transition. Status is per-(reaction, pathway).
        comp_row = query_by_columns(DBComposition,{"composition": self.ctx.chemical_formula})[0]
        if self._ads_status(comp_row.step_status, reaction, reaction_path) in ["Done"]:
            # SQS sweep: the parent being Done is not enough -- re-enter the
            # stage if any grid formula of the union manifest still lacks this
            # (reaction, pathway); the fan-out skips the Done ones.
            per_formula = ((comp_row.stable_struct or {}).get("per_formula")
                           if comp_row.stable_struct else None)
            if per_formula:
                for formula in per_formula:
                    frows = query_by_columns(DBComposition, {"composition": formula})
                    if not frows or self._ads_status(
                            frows[0].step_status, reaction, reaction_path) not in ["Done"]:
                        return True
            row = query_by_columns(DBSurfaceMLAdsorbate,{"composition": self.ctx.chemical_formula,
                                                         "reaction": reaction, "reaction_path": reaction_path})
            if row:
                return False
        return True

    def should_run_battery_neb(self):
        """Tier-2 ion migration: only for battery submissions, only when
        enabled in input.yaml (battery: neb: enabled), only after the tier-1
        battery stage is Done for this ion (it consumes db_battery_path)."""
        if not self.ctx.battery:
            return False
        if self.ctx.nano_generator or self.ctx.sqs:
            return False
        if not (settings.inputs.get("battery") or {}).get("neb", {}).get("enabled", False):
            return False
        working_ion = self.ctx.reaction_path.value
        comp_row = query_by_columns(DBComposition, {"composition": self.ctx.chemical_formula})[0]
        if self._battery_status(comp_row.step_status, working_ion) not in ["Done"]:
            self.report("battery stage not Done -- skipping battery_neb")
            return False
        status = ((comp_row.step_status or {}).get("battery_neb") or {}).get(working_ion)
        if status in ["Done"]:
            row = query_by_columns(DBBatteryNEB, {"composition": self.ctx.chemical_formula,
                                                  "working_ion": working_ion})
            if row:
                return False
        return True

    def should_wait_battery_neb(self):
        """Wait only if THIS (composition, working ion) NEB runs elsewhere."""
        working_ion = self.ctx.reaction_path.value
        comp_row = query_by_columns(DBComposition, {"composition": self.ctx.chemical_formula})[0]
        status = ((comp_row.step_status or {}).get("battery_neb") or {}).get(working_ion)
        if status in ["Running"]:
            self.ctx.sts = f"battery_neb:{working_ion}"
            return True
        return False

    def battery_neb(self):
        """Running BatteryNEBWorkChain"""
        row = self.ctx.dbcomposition_row
        working_ion = self.ctx.reaction_path.value
        update_step_status_path(DBComposition, row.uuid, ["battery_neb", working_ion], "Running")
        builder = self._construct_battery_neb_builder()
        future = self.submit(builder)
        self.to_context(**{"battery_neb_wch": future})

    def inspect_battery_neb(self):
        """Inspecting BatteryNEBWorkChain"""
        wch = self.ctx.battery_neb_wch
        row = self.ctx.dbcomposition_row
        working_ion = self.ctx.reaction_path.value
        state = "Done" if wch.is_finished_ok else "Failed"
        update_step_status_path(DBComposition, row.uuid, ["battery_neb", working_ion], state)
        update_row(DBComposition, row.uuid, {"status": state})
        if state == "Failed":
            self.report(f"BatteryNEB WorkChain failed (exit {wch.exit_status})")
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def should_run_nano_particles(self):
        """Check whether should run Nano Particles routines"""
        if self.ctx.nano_generator:
            nano_particles_step_status = self.ctx.nano_row.step_status.get("nano_generator")
            if nano_particles_step_status in ["Done"]:
                return False
            return True

    def should_wait_nano_particles(self):
        """Should wait for another running WorkChain"""
        if self.ctx.nano_generator:
            nano_particles_step_status = self.ctx.nano_row.step_status.get("nano_generator")
            if nano_particles_step_status in ["Running"]:
                self.ctx.sts = "nano_generator"
                return True
            return False

    def should_wait_pd_ml(self):
        """Should wait for another running WorkChain"""
        if self.ctx.nano_generator:
            return False
        pd_ml_step_status = self._fresh_step_status().get("pd_ml")
        if pd_ml_step_status in ["Running"]:
            self.ctx.sts = "phase diagram"
            return True
        return False

    def should_wait_pd_ver(self):
        """Should wait for another running WorkChain"""
        if self.ctx.nano_generator:
            return False
        pd_ver_step_status = self._fresh_step_status().get("pd_verification")
        if pd_ver_step_status in ["Running"]:
            self.ctx.sts = "phase diagram verification"
            return True
        return False

    def should_wait_sqs(self):
        """Should wait for another running WorkChain"""
        if not self.ctx.sqs:
            return False
        if self.ctx.nano_generator:
            return False
        step_status = self._fresh_step_status().get("sqs")
        if step_status in ["Running"]:
            self.ctx.sts = "sqs"
            return True
        return False

    def should_wait_synthesizability(self):
        """Should wait for another running WorkChain"""
        if self.ctx.nano_generator:
            return False
        step_status = self._fresh_step_status().get("synthesizability")
        if step_status in ["Running"]:
            self.ctx.sts = "synthesizability"
            return True
        return False

    def should_wait_surface_builder(self):
        """Should wait for another running WorkChain"""
        if self.ctx.nano_generator:
            return False
        surface_builder_step_status = self._fresh_step_status().get("surface_builder")
        if surface_builder_step_status in ["Running"]:
            self.ctx.sts = "surface builder"
            return True
        return False

    def should_wait_adsorbates(self):
        """Wait only if THIS (reaction, pathway) is running elsewhere.
        Different reactions/pathways (e.g. CO2RR vs NOXRR) are independent work
        and must not block each other; only a duplicate of the same triple does.
        """
        if self.ctx.battery:
            return False
        if self.ctx.nano_generator:
            return False
        reaction = self.ctx.reaction.value
        reaction_path = self.ctx.reaction_path.value
        comp_row = query_by_columns(DBComposition, {"composition": self.ctx.chemical_formula})[0]
        if self._ads_status(comp_row.step_status, reaction, reaction_path) in ["Running"]:
            self.ctx.sts = f"adsorbates:{reaction}:{reaction_path}"
            return True
        return False

    def wait_sleep(self):
        """Wait until the other workchain for this composition ends"""
        self.report(f"Waiting for a similar WorkChain ({self.ctx.sts})")
        inputs = prepare_pythonjob_inputs(wait_sleep, function_inputs={}, computer="localhost")
        future = self.submit(PythonJob, inputs=inputs)
        self.to_context(**{"pyjob_sleep": future})

    def check_pythonjob_sleep(self):
        """Inspect PythonJob"""
        calculation = self.ctx["pyjob_sleep"]
        if not calculation.is_finished_ok:
            self.report("PythonJob failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def pd_ml(self):
        """Running PhaseDiagramML WorkChain"""
        row = self.ctx.dbcomposition_row
        # update row status in DBComposition table
        row.step_status.update({"pd_ml": "Running"})
        update_row(DBComposition, row.uuid,{"status": "Running", "step_status": row.step_status})
        builder = self._construct_pd_ml_builder()
        future = self.submit(builder)
        self.to_context(**{"pd_ml": future})

    def inspect_pd_ml(self):
        """Inspecting PhaseDiagramML WorkChain"""
        pd_ml_wch = self.ctx.pd_ml
        row = self.ctx.dbcomposition_row
        if not pd_ml_wch.is_finished_ok:
            # update row status in DBComposition table
            row.step_status.update({"pd_ml": "Failed"})
            update_row(DBComposition, row.uuid,{"status": "Failed", "step_status": row.step_status})
            self.report("PhaseDiagramML WorkChain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        # update row status in DBComposition table
        row.step_status.update({"pd_ml": "Done"})
        update_row(DBComposition, row.uuid,{"status": "Running", "step_status": row.step_status})

    def pd_verification(self):
        """Running PDVerificationWorkChain"""
        row = self.ctx.dbcomposition_row
        # update row status in DBComposition table
        row.step_status.update({"pd_verification": "Running"})
        update_row(DBComposition, row.uuid,{"status": "Running", "step_status": row.step_status})
        builder = self._construct_pd_verification_builder()
        future = self.submit(builder)
        self.to_context(**{"pdverification": future})

    def inspect_pd_verification(self):
        """Inspecting PDVerificationWorkChain"""
        pd_ver_wch = self.ctx.pdverification
        row = self.ctx.dbcomposition_row

        if not pd_ver_wch.is_finished_ok:
            # update row status in DBComposition table
            row.step_status.update({"pd_verification": "Failed"})
            update_row(DBComposition, row.uuid,{"status": "Failed","step_status": row.step_status})
            self.report("PDVerification WorkChain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        # update row status in DBComposition table
        row.step_status.update({"pd_verification": "Done"})
        update_row(DBComposition, row.uuid,{"status": "Running","step_status": row.step_status})

    def synthesizability(self):
        """Running SynthesizabilityWorkChain (classify all generated structures)"""
        row = self.ctx.dbcomposition_row
        row.step_status.update({"synthesizability": "Running"})
        update_row(DBComposition, row.uuid, {"status": "Running", "step_status": row.step_status})
        builder = self._construct_synthesizability_builder()
        future = self.submit(builder)
        self.to_context(**{"synthesizability": future})

    def inspect_synthesizability(self):
        """Inspecting SynthesizabilityWorkChain"""
        wch = self.ctx.synthesizability
        row = self.ctx.dbcomposition_row
        if not wch.is_finished_ok:
            row.step_status.update({"synthesizability": "Failed"})
            update_row(DBComposition, row.uuid, {"status": "Failed", "step_status": row.step_status})
            self.report("Synthesizability WorkChain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        row.step_status.update({"synthesizability": "Done"})
        update_row(DBComposition, row.uuid, {"status": "Running", "step_status": row.step_status})

    def sqs(self):
        """Running PhaseDiagramML WorkChain"""
        row = self.ctx.dbcomposition_row
        # update row status in DBComposition table
        row.step_status.update({"sqs": "Running"})
        update_row(DBComposition, row.uuid,{"status": "Running", "step_status": row.step_status})
        builder = self._construct_sqs_builder(self.ctx.sqs_request)
        future = self.submit(builder)
        self.to_context(**{"sqs": future})

    def inspect_sqs(self):
        """Inspecting PhaseDiagramML WorkChain"""
        wch = self.ctx.sqs
        row = self.ctx.dbcomposition_row
        if not wch.is_finished_ok:
            # update row status in DBComposition table
            row.step_status.update({"sqs": "Failed"})
            update_row(DBComposition, row.uuid,{"status": "Failed", "step_status": row.step_status})
            self.report("PhaseDiagramML WorkChain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        # update row status in DBComposition table
        row.step_status.update({"sqs": "Done"})
        update_row(DBComposition, row.uuid,{"status": "Running", "step_status": row.step_status})


#    def band_alignment(self):
#        """Running BandAlignmentWorkChain"""
#        row = self.ctx.dbcomposition_row
#        # update row status in DBComposition table
#        row.step_status.update({"band_alignment": "Running"})
#        update_row(
#                DBComposition,
#                row.uuid,
#                {"status": "Running",
#                 "step_status": row.step_status
#                }
#            )
#        builder = self._construct_band_alignment_builder()
#        future = self.submit(builder)
#        self.to_context(**{"bandalignment": future})

#    def inspect_band_alignment(self):
#        """Inspecting BandAlignemntWorkChain"""
#        b_a_wch = self.ctx.bandalignment
#        row = self.ctx.dbcomposition_row
#
#        if not b_a_wch.is_finished_ok:
#            # update row status in DBComposition table
#            row.step_status.update({"band_alignment": "Failed"})
#            update_row(
#                    DBComposition,
#                    row.uuid,
#                    {"status": "Failed",
#                     "step_status": row.step_status
#                    }
#            )
#            self.report("BandAlignment WorkChain failed")
#            return self.exit_codes.ERROR_CALCULATION_FAILED

    def surface_builder(self):
        """Running SurfaceBuilderWorkChain"""
        row = self.ctx.dbcomposition_row
        row.step_status.update({"surface_builder": "Running"})
        update_row(DBComposition, row.uuid,{"status": "Running", "step_status": row.step_status})
        builder = self._construct_surface_builder()
        future = self.submit(builder)
        self.to_context(**{"surface_builder": future})

    def inspect_surface_builder(self):
        """Inspecting SurfaceBuilderWorkChain"""
        wch = self.ctx.surface_builder
        row = self.ctx.dbcomposition_row
        if not wch.is_finished_ok:
            row.step_status.update({"surface_builder": "Failed"})
            update_row(DBComposition, row.uuid,{"status": "Failed", "step_status": row.step_status})
            self.report("SurfaceBuilder WorkChain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        row.step_status.update({"surface_builder": "Done"})
        update_row(DBComposition, row.uuid,{"status": "Running", "step_status": row.step_status})

    def adsorbates(self):
        """
        For a plain composition this submits one chain for the submission
        formula, as before. When the composition row carries an SQS union
        manifest (``stable_struct.per_formula``, written by
        SQSWorkChain.store_structures), one AdsorbatesWorkChain is submitted
        PER grid formula: the surface builder covered the whole sweep via the
        manifest, but the adsorbates surface query joins on composition, so
        each grid formula needs its own chain. Formulas whose own row is
        already Done for this (reaction, pathway) are skipped, which makes
        re-triggering the parent submission run exactly the missing ones.
        """
        row = self.ctx.dbcomposition_row
        reaction = self.ctx.reaction.value
        reaction_path = self.ctx.reaction_path.value
        path = ["adsorbates", reaction, reaction_path]

        per_formula = ((row.stable_struct or {}).get("per_formula")
                       if row.stable_struct else None)
        formulas = (sorted(set(per_formula) | {self.ctx.chemical_formula})
                    if per_formula else [self.ctx.chemical_formula])

        self.ctx.adsorbates_formulas = []
        for formula in formulas:
            frows = query_by_columns(DBComposition, {"composition": formula})
            frow = frows[0] if frows else None
            if frow is not None and self._ads_status(
                    frow.step_status, reaction, reaction_path) in ["Done"]:
                self.report(f"adsorbates {reaction}/{reaction_path} already "
                            f"Done for {formula}; skipping")
                continue
            if frow is not None:
                # Atomic per-(reaction, pathway) write -- does NOT overwrite
                # siblings' adsorbates keys (a read-modify-write of the whole
                # JSONB would).
                update_step_status_path(DBComposition, frow.uuid, path, "Running")
                update_row(DBComposition, frow.uuid, {"status": "Running"})
            self.ctx.adsorbates_formulas.append(formula)
            future = self.submit(self._construct_adsorbates_builder(formula))
            self.to_context(**{f"adsorbates_{formula}": future})

        if per_formula:
            self.report(f"adsorbates fan-out over the SQS sweep: "
                        f"{len(self.ctx.adsorbates_formulas)} of "
                        f"{len(formulas)} formula(s) submitted")

    def inspect_adsorbates(self):
        """Inspect the fanned-out AdsorbatesWorkChains (one per formula)."""
        reaction = self.ctx.reaction.value
        reaction_path = self.ctx.reaction_path.value
        path = ["adsorbates", reaction, reaction_path]
        failed = []
        for formula in self.ctx.adsorbates_formulas:
            wch = self.ctx[f"adsorbates_{formula}"]
            state = "Done" if wch.is_finished_ok else "Failed"
            frows = query_by_columns(DBComposition, {"composition": formula})
            if frows:
                update_step_status_path(DBComposition, frows[0].uuid, path, state)
                update_row(DBComposition, frows[0].uuid, {"status": state})
            if state == "Failed":
                failed.append((formula, wch.exit_status))

        if failed:
            self.report(f"Adsorbates WorkChain failed for {len(failed)}/"
                        f"{len(self.ctx.adsorbates_formulas)} formula(s): "
                        + ", ".join(f"{f} (exit {x})" for f, x in failed))
            parent_failed = any(f == self.ctx.chemical_formula for f, _ in failed)
            if parent_failed or len(failed) == len(self.ctx.adsorbates_formulas):
                return self.exit_codes.ERROR_CALCULATION_FAILED

    @staticmethod
    def _photocat_status(step_status, reaction, reaction_path):
        """Per-(reaction, pathway) photocat status, nested like adsorbates."""
        return ((step_status or {}).get("photocat") or {}).get(reaction, {}) \
            .get(reaction_path) or {}

    def should_run_photocat(self):
        """Stage-1 gap filter: only after the adsorbates stage is Done for
        THIS (reaction, pathway); needs photocat.enabled in input.yaml and a
        registered codes.photocat (env with the gap models)."""
        if not settings.PHOTOCAT_ENABLED:
            return False
        if self.ctx.battery or self.ctx.nano_generator or self.ctx.sqs:
            return False
        if "photocat" not in (settings.configs.get("codes") or {}):
            self.report("photocat enabled but codes.photocat is not registered "
                        "in config.yaml -- skipping the gap filter")
            return False
        reaction = self.ctx.reaction.value
        reaction_path = self.ctx.reaction_path.value
        comp_row = query_by_columns(DBComposition, {"composition": self.ctx.chemical_formula})[0]
        if self._ads_status(comp_row.step_status, reaction, reaction_path) not in ["Done"]:
            self.report("adsorbates not Done for this (reaction, pathway) -- "
                        "skipping photocat")
            return False
        if self._photocat_status(comp_row.step_status, reaction, reaction_path) in ["Done"]:
            return False
        return True

    def should_wait_photocat(self):
        """Wait only if THIS (reaction, pathway) photocat runs elsewhere."""
        reaction = self.ctx.reaction.value
        reaction_path = self.ctx.reaction_path.value
        comp_row = query_by_columns(DBComposition, {"composition": self.ctx.chemical_formula})[0]
        if self._photocat_status(comp_row.step_status, reaction, reaction_path) in ["Running"]:
            self.ctx.sts = f"photocat:{reaction}:{reaction_path}"
            return True
        return False

    def photocat(self):
        """Running PhotocatWorkChain"""
        row = self.ctx.dbcomposition_row
        reaction = self.ctx.reaction.value
        reaction_path = self.ctx.reaction_path.value
        update_step_status_path(DBComposition, row.uuid,
                                ["photocat", reaction, reaction_path], "Running")
        builder = self._construct_photocat_builder()
        future = self.submit(builder)
        self.to_context(**{"photocat_wch": future})

    def inspect_photocat(self):
        """Inspecting PhotocatWorkChain"""
        wch = self.ctx.photocat_wch
        row = self.ctx.dbcomposition_row
        reaction = self.ctx.reaction.value
        reaction_path = self.ctx.reaction_path.value
        state = "Done" if wch.is_finished_ok else "Failed"
        update_step_status_path(DBComposition, row.uuid,
                                ["photocat", reaction, reaction_path], state)
        if state == "Failed":
            self.report(f"Photocat WorkChain failed (exit {wch.exit_status})")
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def should_run_battery(self):
        """Check whether should run Battery for THIS working ion (the
        submission's reaction_path). Battery replaces the catalysis branch."""
        if not self.ctx.battery:
            return False
        if self.ctx.nano_generator or self.ctx.sqs:
            return False
        working_ion = self.ctx.reaction_path.value
        comp_row = query_by_columns(DBComposition, {"composition": self.ctx.chemical_formula})[0]
        if self._battery_status(comp_row.step_status, working_ion) in ["Done"]:
            row = query_by_columns(DBBatteryPath, {"composition": self.ctx.chemical_formula,
                                                   "working_ion": working_ion})
            if row:
                return False
        return True

    def should_wait_battery(self):
        """Wait only if THIS (composition, working ion) runs elsewhere."""
        working_ion = self.ctx.reaction_path.value
        comp_row = query_by_columns(DBComposition, {"composition": self.ctx.chemical_formula})[0]
        if self._battery_status(comp_row.step_status, working_ion) in ["Running"]:
            self.ctx.sts = f"battery:{working_ion}"
            return True
        return False

    def battery(self):
        """Running BatteryWorkChain"""
        row = self.ctx.dbcomposition_row
        working_ion = self.ctx.reaction_path.value
        # atomic per-ion write, same convention as the adsorbates leaf
        update_step_status_path(DBComposition, row.uuid, ["battery", working_ion], "Running")
        update_row(DBComposition, row.uuid, {"status": "Running"})
        builder = self._construct_battery_builder()
        future = self.submit(builder)
        # NOT the "battery" key: that would clobber the ctx.battery flag set
        # in setup (the sqs stage lives with that clobber; no need to copy it)
        self.to_context(**{"battery_wch": future})

    def inspect_battery(self):
        """Inspecting BatteryWorkChain"""
        wch = self.ctx.battery_wch
        row = self.ctx.dbcomposition_row
        working_ion = self.ctx.reaction_path.value
        state = "Done" if wch.is_finished_ok else "Failed"
        update_step_status_path(DBComposition, row.uuid, ["battery", working_ion], state)
        update_row(DBComposition, row.uuid, {"status": state})
        if state == "Failed":
            self.report(f"Battery WorkChain failed (exit {wch.exit_status})")
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def nano_particles(self):
        """Running NanoParticlesWorkChain"""
        row = self.ctx.nano_row
        row.step_status.update({"nano_particles": "Running"})
        update_row(DBNanoParticles, row.uuid,{"status": "Running", "step_status": row.step_status})
        builder = self._construct_particle_builder()
        future = self.submit(builder)
        self.to_context(**{"nano_particles": future})

    def inspect_nano_particles(self):
        """Analyze results of the builder chain"""
        chain = self.ctx.nano_particles
        row = self.ctx.nano_row
        if not chain.is_finished_ok:
            row.step_status.update({"nano_particles": "Failed"})
            update_row(DBNanoParticles, row.uuid,{"status": "Failed", "step_status": row.step_status})
            self.report("NanoParticles WorkChain failed.")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        row.step_status.update({"nano_particles": "Done"})
        update_row(DBNanoParticles, row.uuid,{"status": "Done", "step_status": row.step_status})
        self.report("NanoParticles WorkChain succeeded.")

    ################################################################################
    def _construct_pd_ml_builder(self):
        """Build PhaseDiagramML WorkChain builder"""
        PhaseDiagramMLWorkChain = WorkflowFactory("phasediagram")
        builder = PhaseDiagramMLWorkChain.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.chemical_systems = self.ctx.chemical_systems
        return builder

    def _construct_pd_verification_builder(self):
        """Build PDVerification WorkChain builder"""
        PDVerificationWorkChain = WorkflowFactory("pdverification")
        builder = PDVerificationWorkChain.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        return builder

    def _construct_synthesizability_builder(self):
        """Build Synthesizability WorkChain builder"""
        SynthesizabilityWorkChain = WorkflowFactory("synthesizability")
        builder = SynthesizabilityWorkChain.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        return builder

#    def _construct_band_alignment_builder(self):
#        """Build PDVerification WorkChain builder"""
#        BandAlignmentWorkChain = WorkflowFactory("bandalignment")
#        builder = BandAlignmentWorkChain.get_builder()
#        builder.chemical_formula = Str(self.ctx.chemical_formula)
#        return builder

    def _construct_surface_builder(self):
        """SurfaceBuilder WorkChain builder"""
        SurfaceBuilderWorkChain = WorkflowFactory("surfacebuilder")
        builder = SurfaceBuilderWorkChain.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        return builder

    def _construct_adsorbates_builder(self, formula=None):
        """Adsorbates WorkChain builder. ``formula`` overrides the submission
        composition for the SQS-sweep fan-out (one chain per grid formula)."""
        AdsorbatesWorkChain = WorkflowFactory("adsorbates")
        builder = AdsorbatesWorkChain.get_builder()
        builder.chemical_formula = Str(formula or self.ctx.chemical_formula)
        builder.reaction = self.ctx.reaction
        builder.reaction_path = self.ctx.reaction_path
        return builder

    def _construct_battery_builder(self):
        """Battery WorkChain builder -- the working ion IS the reaction_path."""
        BatteryWorkChain = WorkflowFactory("battery")
        builder = BatteryWorkChain.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.working_ion = Str(self.ctx.reaction_path.value)
        return builder

    def _construct_photocat_builder(self):
        """Photocat WorkChain builder (stage-1 gap filter)."""
        PhotocatWorkChain = WorkflowFactory("photocat")
        builder = PhotocatWorkChain.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.reaction = self.ctx.reaction
        builder.reaction_path = self.ctx.reaction_path
        return builder

    def _construct_battery_neb_builder(self):
        """BatteryNEB WorkChain builder (tier-2 ion migration)."""
        BatteryNEBWorkChain = WorkflowFactory("battery_neb")
        builder = BatteryNEBWorkChain.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.working_ion = Str(self.ctx.reaction_path.value)
        return builder

    def _construct_particle_builder(self):
        """Nano Particles WorkChain builder (adsorbates on DBNanoParticles rows).

        The MLIP is the adsorbates model from settings, so particle and slab
        energetics come from the same potential. ``particles_range`` (the
        ``nanoparticles`` submission entry, e.g. "40-160") filters the
        particle inventory by atom count.
        """
        WorkChain = WorkflowFactory("nano_particles")
        builder = WorkChain.get_builder()
        builder.elements = Str('-'.join(sorted(str(el) for el in Composition(self.ctx.chemical_formula).elements)))
        builder.particles_range = Str(self.ctx.nano_particles_range)
        builder.reaction = self.ctx.reaction
        builder.reaction_path = self.ctx.reaction_path
        builder.ml_model = Str(settings.inputs['adsorbates']['model'])
        return builder

    def _construct_sqs_builder(self, request):
        """SQS WorkChain builder.

        The ``request`` payload (parent structure, sublattices,
        composition_grid, surfaces, defects) is the ``sqs_request`` dict passed
        through the submission entry (see run_dir/run.py), threaded here via
        ``ctx.sqs_request``. Optional ``mu_O2`` (eV per O2 molecule,
        MLIP-relaxed) and ``functional`` (elemental reference set for the bulk
        hull) are keys inside that same request dict.
        """
        WorkChain = WorkflowFactory("sqs")
        builder = WorkChain.get_builder()
        builder.request = Dict(dict=request)
        builder.local_label = Str('{}'.format(request['parent_label']))
        # if "mu_O2" in settings.inputs['SQS']:
        #     from aiida.orm import Float
        #     builder.mu_O2 = Float(float(settings.inputs['SQS']["mu_O2"]))
        return builder
