from aiida.engine import WorkChain
from aiida.plugins import WorkflowFactory
from aiida.orm import Str, Dict, List
from uvsib.workchains.utils import get_code, get_model_device
from uvsib.workflows import settings


class NanoParticleWorkChain(WorkChain):
    """Nano-Particles WorkChain"""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("elements", valid_type=Str)
        spec.input('particles_range', valid_type=Str)
        spec.input('generator', valid_type=Str)
        spec.input("ml_model", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.generate,
            # cls.run_dft,
            # cls.validate,
            cls.store_results,
            cls.final_report
        )

        spec.exit_code(300,
            "ERROR_CALCULATION_FAILED",
            message="The calculation did not finish successfully"
        )
        spec.exit_code(
            301,
            "ERROR_NO_STRUCTURES_FOUND",
            message="No structures were found for the given formula."
        )

    def setup(self):
        """Setup and report"""
        self.ctx.ml_model = self.inputs.ml_model.value
        self.ctx.elements = list(str(self.inputs.elements.value).split('-'))
        self.ctx.particles_range = self.inputs.particles_range.value
        self.ctx.generated_particles = list()
        self.ctx.relax_batches = None

        # self.ctx.structure_surface_rows = get_structure_uuid_surface_id(self.ctx.chemical_formula)
        # if not self.ctx.structure_surface_rows:
        #     return self.exit_codes.ERROR_NO_STRUCTURES_FOUND
        # self.ctx.protocol = read_yaml(os.path.join(settings.vasp_files_path, "protocol.yaml"))
        # self.ctx.potential_family = settings.configs["codes"]["VASP"]["potential_family"]
        # potential_mapping = read_yaml(os.path.join(settings.vasp_files_path, "potential_mapping.yaml"))
        # self.ctx.potential_mapping = potential_mapping["potential_mapping"]
        # self.ctx.vasp_code = load_code(settings.configs["codes"]["VASP"]["code_string"])

        self.report("Running Nano-Particles WorkChain with {} for {}".format(self.ctx.ml_model, self.ctx.elements))

    def generate(self):
        """Run generator routines"""
        self.report('Generating nano-particles')
        elements_string = '-'.join(self.ctx.elements)
        builder = self._particle_builder(elements_string=elements_string, particles_range=self.ctx.particles_range,
                                         ml_model=self.ctx.ml_model)
        future = self.submit(builder)
        # print('nanoparticles_{}'.format(self.ctx.particles_range))
        self.to_context(**{'generated_{}'.format(self.ctx.particles_range): future})

    def cook_and_relax(self):
        self.report('Batching produced particles for cooking and relaxation')
        chain = self.ctx['generated_{}'.format(self.ctx.particles_range)]
        print(chain)
        if not chain.is_finished_ok:
            self.report('Generator for nano-particles failed')
            return
        output_dict = chain.called[-1].outputs.output_dict
        generated_particles = output_dict['structures']
        count = 0
        relax_batch = List()
        for structure_dict in enumerate(generated_particles):
            relax_batch.append(structure_dict)
            if len(relax_batch) % 10 == 0:
                builder = self._particle_relaxer(structures=relax_batch, ml_model=self.ctx.ml_model)
                future = self.submit(builder)
                self.to_context(**{'cooked_{}_{}'.format(self.ctx.particles_range, count): future})
                count += 1
                relax_batch = List()
        if len(relax_batch) > 0:
            count += 1
            builder = self._particle_relaxer(structures=relax_batch, ml_model=self.ctx.ml_model)
            future = self.submit(builder)
            self.to_context(**{'cooked_{}_{}'.format(self.ctx.particles_range, count): future})
        self.ctx.relax_batches = count

    def analyze(self):
        self.report('Analyze produced nano-particles')
        for batch_index in range(len(self.ctx.relax_batches)):
            chain = self.ctx['cooked_{}_{}'.format(self.ctx.particles_range, batch_index)]
            print(batch_index, chain)
            if not chain.is_finished_ok:
                self.report('Generator for nano-particles failed')
                return
            output_dict = chain.called[-1].outputs.output_dict
            cooked_structures = output_dict['structures']
            for structure_dict in cooked_structures:
                self.ctx.generated_particles.append(structure_dict)
            print('{}'.format(len(self.ctx.generated_particles)))
        print('total: {}'.format(len(self.ctx.generated_particles)))

    def run_dft(self):
        """Run dft validation"""
        pass
        # for parent_key, adsorption_sets in self.ctx.ml_results.items():
        #     run_clean_surface = True
        #     for adsorb_set in adsorption_sets:
        #         for ads_json in adsorb_set:
        #             adsorbed = jsonio.decode(ads_json)
        #             if adsorbed.info["adsorbate"] == "*":
        #                 if not run_clean_surface:
        #                     continue
        #                 run_clean_surface = False
        #             unique_idx = adsorbed.info["adsorbate_collection"]
        #             site = adsorbed.info["site"]
        #             ad = adsorbed.info["adsorbate"]
        #             pmg_structure = ase_to_pmg(adsorbed)
        #             struct = StructureData(pymatgen=pmg_structure)
        #             struct.base.attributes.set("site_properties",
        #                                        pmg_structure.site_properties
        #                                        )
        #             builder = construct_vasp_builder(
        #                 struct,
        #                 self.ctx.protocol["r2SCAN_adsorbates"],
        #                 self.ctx.potential_family,
        #                 self.ctx.potential_mapping,
        #                 self.ctx.vasp_code
        #             )
        #             future = self.submit(builder)
        #             self.to_context(**{f"scan_{parent_key}_{site}_{unique_idx}_{ad}": future})

    def validate(self):
        """Validate ML vs r2SCAN VASP"""
        pass
        # failed_jobs = 0
        # for parent_key, adsorption_sets in self.ctx.ml_results.items():
        #     for adsorb_set in adsorption_sets:
        #         ad_set = []
        #         for ads_json in adsorb_set:
        #             adsorbed = jsonio.decode(ads_json)
        #             unique_idx = adsorbed.info["adsorbate_collection"]
        #             site = adsorbed.info["site"]
        #             ad = adsorbed.info["adsorbate"]
        #             if adsorbed.info["adsorbate"] == "*":
        #                 ad_set.append([{}, "*", adsorbed.info[
        #                     "clean_slab_energy"]])  # TODO bare surface energy is calculated only by ML
        #                 continue
        #             scan_wch = self.ctx[f"scan_{parent_key}_{site}_{unique_idx}_{ad}"]
        #             if not scan_wch.is_finished_ok:
        #                 failed_jobs += 1
        #                 break
        #             outputs = scan_wch.outputs
        #             structure = outputs.structure.get_pymatgen()
        #             energy = outputs.misc["total_energies"]["energy_extrapolated"]
        #             ad_set.append([structure.as_dict(), ad, energy])
        #         self.ctx.scan_results[f"{parent_key}_{site}_{unique_idx}"].append(ad_set)
        #
        # if failed_jobs:
        #     self.report(f"{failed_jobs} r2SCAN jobs failed")

    def store_results(self):
        """Store results"""
        self.report('Storing results')
        pass
        # for parent_key, adsorption_sets in self.ctx.ml_results.items():
        #     uuid_str, surface_id = parent_key.split("_", 2)
                # add_surface_adsorbate(
                #     uuid_str,
                #     surface_id,
                #     self.ctx.chemical_formula,
                #     self.ctx.reaction,
                #     site,
                #     idx,
                #     eta,
                #     dG,
                #     adsorb_set,
                # )

    def final_report(self):
        """Final report"""
        self.report('NanoParticlesWorkChain for {} finished successfully'.format(self.ctx.elements))

    @staticmethod
    def _particle_builder(elements_string, particles_range, ml_model):
        """Builder for generating nano-particles"""
        Workflow = WorkflowFactory(ml_model.lower())
        builder = Workflow.get_builder()
        builder.code = get_code(ml_model)
        builder.input_structures = list()
        model, model_path, device = get_model_device(ml_model)

        relax_key = 'nano_particles'

        job_info = {
            'job_type': relax_key,
            'ML_model': ml_model,
            'device': device,
            'elements': elements_string,
            'particles_range': particles_range,
            'generator': 'systematic',
            'fmax': settings.inputs[relax_key]['fmax'],
            'max_steps': settings.inputs[relax_key]['max_steps']
        }
        if ml_model in ['uPET']:
            job_info.update({'model_name': model})
        else:
            job_info.update({'model_path': model_path})

        builder.job_info = Dict(job_info)
        return builder

    @staticmethod
    def _particle_relaxer(structures, ml_model):
        """Builder for generating nano-particles"""
        Workflow = WorkflowFactory(ml_model.lower())
        builder = Workflow.get_builder()
        builder.code = get_code(ml_model)
        builder.input_structures = List(structures)
        model, model_path, device = get_model_device(ml_model)

        relax_key = 'relax'

        job_info = {
            'job_type': relax_key,
            'ML_model': ml_model,
            'device': device,
            'fmax': settings.inputs[relax_key]['fmax'],
            'max_steps': settings.inputs[relax_key]['max_steps']
        }
        if ml_model in ['uPET']:
            job_info.update({'model_name': model})
        else:
            job_info.update({'model_path': model_path})

        builder.job_info = Dict(job_info)
        return builder
