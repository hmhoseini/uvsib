from aiida.orm import Str, List
from aiida.plugins import WorkflowFactory
from aiida.engine import WorkChain, if_, while_
from aiida_pythonjob import PythonJob, prepare_pythonjob_inputs
from uvsib.db.tables import DBComposition, DBSurfaceAdsorbate, DBNanoParticles
from uvsib.db.utils import update_row, query_by_columns
from uvsib.workchains.pythonjob_inputs import wait_sleep
from pymatgen.core.composition import Composition


class MainWorkChain(WorkChain):
    """ Main WorkChain"""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)
        spec.input("chemical_systems", valid_type=List)
        spec.input("ML_model", valid_type=Str)
        spec.input("reaction", valid_type=Str)
        spec.input("nanoparticles", valid_type=Str)

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
            if_(cls.should_run_band_alignment)(
                while_(cls.should_wait_band_alignment)(
                    cls.wait_sleep,
                    cls.check_pythonjob_sleep
                ),
                cls.band_alignment,
                cls.inspect_band_alignment
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
            if_(cls.should_run_nano_generator)(
                while_(cls.should_wait_nano_generator)(
                    cls.wait_sleep,
                    cls.check_pythonjob_sleep
                ),
                cls.nano_generator,
                cls.inspect_nano_generator
            )
        )

        spec.exit_code(
            300,
            "ERROR_CALCULATION_FAILED",
            message="A sub-workchain did not finish successfully"
        )

    def setup(self):
        """Setup and report"""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.chemical_systems = self.inputs.chemical_systems
        self.ctx.ML_model = self.inputs.ML_model
        self.ctx.reaction = self.inputs.reaction
        self.ctx.dbcomposition_row = query_by_columns(DBComposition,{"composition": self.ctx.chemical_formula})[0]
        self.ctx.nano_generator = True if len(self.inputs.nanoparticles.value.split('-')) == 2 else False
        if self.ctx.nano_generator:
            self.ctx.nano_particles_range = self.inputs.nanoparticles.value
            # self.ctx.nano_min_atoms = self.inputs.nanoparticles.value.split('-')[0]
            # self.ctx.nano_max_atoms = self.inputs.nanoparticles.value.split('-')[1]
            elements = '-'.join(sorted(list([str(el) for el in Composition(self.ctx.chemical_formula).elements])))
            self.ctx.nano_row = query_by_columns(DBNanoParticles,{'elements': elements})[0]
            self.report('Running NanoParticleGenerator for elements {}'.format(elements))
        else:
            self.report(f"Running MainWorkChain for {self.ctx.chemical_formula}")

    def should_run_pd_ml(self):
        """Check whether should run PhaseDiagramML"""
        if self.ctx.nano_generator:
            return False
        pd_ml_step_status = self.ctx.dbcomposition_row.step_status.get("pd_ml")
        if pd_ml_step_status in ["Done"]:
            return False
        return True

    def should_run_pd_verification(self):
        """Check whether should run PDVerification"""
        if self.ctx.nano_generator:
            return False
        pd_ver_step_status = self.ctx.dbcomposition_row.step_status.get("pd_verification")
        if pd_ver_step_status in ["Done"]:
            return False
        return True

    def should_run_band_alignment(self):
        """Check whether should run BandAlignment"""
        if self.ctx.nano_generator:
            return False
        band_alignment_step_status = self.ctx.dbcomposition_row.step_status.get("band_alignment")
        if band_alignment_step_status in ["Done"]:
            return False
        return True

    def should_run_surface_builder(self):
        """Check whether should run SurfaceBuilder"""
        if self.ctx.nano_generator:
            return False
        surface_builder_step_status = self.ctx.dbcomposition_row.step_status.get("surface_builder")
        if surface_builder_step_status in ["Done"]:
            return False
        return True

    def should_run_adsorbates(self):
        """Check whether should run Adsorbates"""
        if self.ctx.nano_generator:
            return False
        adsorbates_step_status = self.ctx.dbcomposition_row.step_status.get("adsorbates")
        if adsorbates_step_status in ["Done"]:
            row = query_by_columns(DBSurfaceAdsorbate, {"composition": self.ctx.chemical_formula, "reaction": self.ctx.reaction.value})
            if row:
                return False
        return True






    def should_run_nano_generator(self):
        """Check whether should run Nano Particles routines"""
        if self.ctx.nano_generator:
            nano_particles_step_status = self.ctx.nano_row.step_status.get("nano_generator")
            if nano_particles_step_status in ["Done"]:
                return False
            return True

    def should_wait_nano_generator(self):
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
        pd_ml_step_status = self.ctx.dbcomposition_row.step_status.get("pd_ml")
        if pd_ml_step_status in ["Running"]:
            self.ctx.sts = "phase diagram"
            return True
        return False

    def should_wait_pd_ver(self):
        """Should wait for another running WorkChain"""
        if self.ctx.nano_generator:
            return False
        pd_ver_step_status = self.ctx.dbcomposition_row.step_status.get("pd_verification")
        if pd_ver_step_status in ["Running"]:
            self.ctx.sts = "phase diagram verification"
            return True
        return False

    def should_wait_band_alignment(self):
        """Should wait for another running WorkChain"""
        if self.ctx.nano_generator:
            return False
        band_alignment_step_status = self.ctx.dbcomposition_row.step_status.get("band_alignment")
        if band_alignment_step_status in ["Running"]:
            self.ctx.sts = "band alignment"
            return True
        return False

    def should_wait_surface_builder(self):
        """Should wait for another running WorkChain"""
        if self.ctx.nano_generator:
            return False
        surface_builder_step_status = self.ctx.dbcomposition_row.step_status.get("surface_builder")
        if surface_builder_step_status in ["Running"]:
            self.ctx.sts = "surface builder"
            return True
        return False

    def should_wait_adsorbates(self):
        """Should wait for another running WorkChain"""
        if self.ctx.nano_generator:
            return False
        adsorbates_step_status = self.ctx.dbcomposition_row.step_status.get("adsorbates")
        if adsorbates_step_status in ["Running"]:
            self.ctx.sts = "adsorbates"
            return True
        return False

    def wait_sleep(self):
        """Wait until the other workchain for this composition ends"""
        self.report(f"Waiting for a similar WorkChain ({self.ctx.sts})")
        inputs = prepare_pythonjob_inputs(
            wait_sleep,
            function_inputs= {},
            computer="localhost",
        )
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
        update_row(
                DBComposition,
                row.uuid,
                {"status": "Running",
                 "step_status": row.step_status
                }
        )
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
            update_row(
                    DBComposition,
                    row.uuid,
                    {"status": "Failed",
                     "step_status": row.step_status
                    }
            )
            self.report("PhaseDiagramML WorkChain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        # update row status in DBComposition table
        row.step_status.update({"pd_ml": "Done"})
        update_row(
                DBComposition,
                row.uuid,
                {"status": "Running",
                 "step_status": row.step_status
                }
        )

    def pd_verification(self):
        """Running PDVerificationWorkChain"""
        row = self.ctx.dbcomposition_row
        # update row status in DBComposition table
        row.step_status.update({"pd_verification": "Running"})
        update_row(
                DBComposition,
                row.uuid,
                {"status": "Running",
                 "step_status": row.step_status
                }
            )
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
            update_row(
                    DBComposition,
                    row.uuid,
                    {"status": "Failed",
                     "step_status": row.step_status
                    }
            )
            self.report("PDVerification WorkChain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        # update row status in DBComposition table
        row.step_status.update({"pd_verification": "Done"})
        update_row(
                DBComposition,
                row.uuid,
                {"status": "Running",
                 "step_status": row.step_status
                }
        )

    def band_alignment(self):
        """Running BandAlignmentWorkChain"""
        row = self.ctx.dbcomposition_row
        # update row status in DBComposition table
        row.step_status.update({"band_alignment": "Running"})
        update_row(
                DBComposition,
                row.uuid,
                {"status": "Running",
                 "step_status": row.step_status
                }
            )
        builder = self._construct_band_alignment_builder()
        future = self.submit(builder)
        self.to_context(**{"bandalignment": future})

    def inspect_band_alignment(self):
        """Inspecting BandAlignemntWorkChain"""
        b_a_wch = self.ctx.bandalignment
        row = self.ctx.dbcomposition_row

        if not b_a_wch.is_finished_ok:
            # update row status in DBComposition table
            row.step_status.update({"band_alignment": "Failed"})
            update_row(
                    DBComposition,
                    row.uuid,
                    {"status": "Failed",
                     "step_status": row.step_status
                    }
            )
            self.report("BandAlignment WorkChain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        # update row status in DBComposition table
        row.step_status.update({"band_alignment": "Done"})
        update_row(
                DBComposition,
                row.uuid,
                {"status": "Running",
                 "step_status": row.step_status
                }
        )

    def surface_builder(self):
        """Running SurfaceBuilderWorkChain"""
        row = self.ctx.dbcomposition_row
        # update row status in DBComposition table
        row.step_status.update({"surface_builder": "Running"})
        update_row(
            DBComposition,
            row.uuid,
            {"status": "Running",
             "step_status": row.step_status
             }
        )
        builder = self._construct_surface_builder()
        future = self.submit(builder)
        self.to_context(**{"surface_builder": future})

    def inspect_surface_builder(self):
        """Inspecting SurfaceBuilderWorkChain"""
        s_b_wch = self.ctx.surface_builder
        row = self.ctx.dbcomposition_row

        if not s_b_wch.is_finished_ok:
            # update row status in DBComposition table
            row.step_status.update({"surface_builder": "Failed"})
            update_row(
                    DBComposition,
                    row.uuid,
                    {"status": "Failed",
                     "step_status": row.step_status
                    }
            )
            self.report("SurfaceBuilder WorkChain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        # update row status in DBComposition table
        row.step_status.update({"surface_builder": "Done"})
        update_row(
                DBComposition,
                row.uuid,
                {"status": "Running",
                 "step_status": row.step_status
                }
        )

    def adsorbates(self):
        """Running AdsorbatesWorkChain"""
        row = self.ctx.dbcomposition_row
        # update row status in DBComposition table
        row.step_status.update({"adsorbates": "Running"})
        update_row(
            DBComposition,
            row.uuid,
            {"status": "Running",
             "step_status": row.step_status
             }
        )
        builder = self._construct_adsorbates_builder()
        future = self.submit(builder)
        self.to_context(**{"adsorbates": future})

    def inspect_adsorbates(self):
        """Inspecting SurfaceBuilderWorkChain"""
        ad_wch = self.ctx.adsorbates
        row = self.ctx.dbcomposition_row

        if not ad_wch.is_finished_ok:
            # update row status in DBComposition table
            row.step_status.update({"adsorbates": "Failed"})
            update_row(
                    DBComposition,
                    row.uuid,
                    {"status": "Failed",
                     "step_status": row.step_status
                    }
            )
            self.report("Adsorbates WorkChain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        # update row status in DBComposition table
        row.step_status.update({"adsorbates": "Done"})
        update_row(
                DBComposition,
                row.uuid,
                {"status": "Done",
                 "step_status": row.step_status
                }
        )
















    def nano_generator(self):
        """Running NanoParticlesWorkChain"""
        row = self.ctx.nano_row
        row.step_status.update({"nano_generator": "Running"})
        update_row(DBComposition, row.uuid,{"status": "Running", "step_status": row.step_status})
        builder = self._construct_particle_builder()
        future = self.submit(builder)
        self.to_context(**{"nano_particles": future})

    def inspect_nano_generator(self):
        """Analyze results of the builder chain"""
        chain = self.ctx.nano_particles
        row = self.ctx.nano_row
        if not chain.is_finished_ok:
            row.step_status.update({"nano_generator": "Failed"})
            update_row(DBComposition, row.uuid,{"status": "Failed", "step_status": row.step_status})
            self.report("NanoGenerator WorkChain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        row.step_status.update({"nano_generator": "Done"})
        update_row(DBNanoParticles, row.uuid,{"status": "Done", "step_status": row.step_status})








    ################################################################################
    def _construct_pd_ml_builder(self):
        """Build PhaseDiagramML WorkChain builder"""
        PhaseDiagramMLWorkChain = WorkflowFactory("phasediagram")
        builder = PhaseDiagramMLWorkChain.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.chemical_systems = self.ctx.chemical_systems
        builder.ML_model = self.ctx.ML_model
        return builder

    def _construct_pd_verification_builder(self):
        """Build PDVerification WorkChain builder"""
        PDVerificationWorkChain = WorkflowFactory("pdverification")
        builder = PDVerificationWorkChain.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.ML_model = self.ctx.ML_model
        return builder

    def _construct_band_alignment_builder(self):
        """Build PDVerification WorkChain builder"""
        BandAlignmentWorkChain = WorkflowFactory("bandalignment")
        builder = BandAlignmentWorkChain.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        return builder

    def _construct_surface_builder(self):
        """SurfaceBuilder WorkChain builder"""
        SurfaceBuilderWorkChain = WorkflowFactory("surfacebuilder")
        builder = SurfaceBuilderWorkChain.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.ML_model = self.ctx.ML_model
        return builder

    def _construct_adsorbates_builder(self):
        """Adsorbates WorkChain builder"""
        AdsorbatesWorkChain = WorkflowFactory("adsorbates")
        builder = AdsorbatesWorkChain.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.ML_model = self.ctx.ML_model
        builder.reaction = self.ctx.reaction
        return builder

    def _construct_particle_builder(self):
        """Nano Particles WorkChain builder"""
        NanoParticleWorkChain = WorkflowFactory("nano_particles")
        builder = NanoParticleWorkChain.get_builder()
        builder.elements = '-'.join(list(str(el) for el in Composition(self.ctx.chemical_formula).elements))
        builder.particles_range = self.ctx.nano_particles_range
        builder.generator = 'systematic'
        builder.ml_model = self.ctx.ML_model
        return builder
