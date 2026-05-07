import re
import yaml
from aiida.engine import WorkChain
from aiida.plugins import WorkflowFactory
from aiida.orm import Str, List, Dict, Int
from pymatgen.core import Structure, Composition
from matseran.db.utils import add_similarities
from matseran.workchains.utils import get_code, get_model_device
from matseran.workflows import settings


def read_yaml(file_path):
    """Read a yaml file"""
    with open(file_path, "r", encoding="utf8") as fhandle:
        return yaml.safe_load(fhandle)


class SimilarityWorkChain(WorkChain):
    """Similarity WorkChain"""
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("ml_model", valid_type=Str)
        spec.input("comparisons", valid_type=List)
        spec.input("chemical_formula", valid_type=Str)
        spec.input("chemical_systems", valid_type=List)

        spec.outline(
            cls.setup,
            cls.run_csp,
            cls.inspect_csp,
            cls.run_gen,
            cls.inspect_gen,
            cls.run_similarity,
            cls.inspect_similarity,
            cls.final_report
        )

        spec.exit_code(300,"ERROR_CSP_FAILED","The crystal structure prediction failed")
        spec.exit_code(300,"ERROR_GEN_FAILED","The mattergen generator failed")
        spec.exit_code(400,"ERROR_SIMILARITY_FAILED", "The similarity comparison computations failed")

    def setup(self):
        """Setup and report"""
        self.report("Running Similarity WorkChain")
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        # self.ctx.chemical_system = '-'.join(sorted(el.symbol for el in Composition(self.ctx.chemical_formula).elements))
        self.ctx.chemical_system = max(self.inputs.chemical_systems.value, key=len)
        self.ctx.ml_model = self.inputs.ml_model.value
        self.ctx.comparisons = self.inputs.comparisons.value
        self.ctx.generated_structures = list()
        self.ctx.n_csp = settings.inputs["similarity_csp"]["num_runs"]
        self.ctx.n_gen = settings.inputs["similarity_csp"]["num_runs"]
        self.ctx.similarity_zeta = int(settings.inputs["similarity"]["zeta"])
        self.ctx.chunks = None
        self.ctx.success = False

    def run_csp(self):
        for i in range(self.ctx.n_csp):
            builder = self._csp_builder()
            future = self.submit(builder)
            self.to_context(**{f"csp_{i}": future})

    def inspect_csp(self):
        failed_jobs = 0
        for i in range(self.ctx.n_csp):
            wch = self.ctx[f"csp_{i}"]
            if not wch.is_finished_ok:
                failed_jobs += 1
                continue
            try:
                self.ctx.generated_structures.extend(wch.called[-1].outputs.output_dict["structures"])
            except:
                failed_jobs += 1

        if not self.ctx.generated_structures or (failed_jobs / self.ctx.n_csp > 0.5):
            return self.exit_codes.ERROR_CSP_FAILED

    def run_gen(self):


        print(self.ctx.chemical_system)


        for i in range(self.ctx.n_gen):
            builder = self._gen_builder()
            future = self.submit(builder)
            self.to_context(**{f"{self.ctx.chemical_system}_mattergen_{i}": future})

    def inspect_gen(self):
        failed_jobs = 0
        for i in range(self.ctx.n_gen):
            wch = self.ctx[f"{self.ctx.chemical_system}_mattergen_{i}"]
            if not wch.is_finished_ok:
                failed_jobs += 1
                continue
            try:
                self.ctx.generated_structures.extend(wch.called[-1].outputs.output_dict["structures"])
            except:
                failed_jobs += 1
        if failed_jobs == self.ctx.n_gen:
            return self.exit_codes.ERROR_GEN_FAILED

    def run_similarity(self):

        print('ALISUYGDUIYAGSDU')
        print(len(self.ctx.generated_structures), (len(self.ctx.generated_structures)//1000))


        if len(self.ctx.generated_structures) > 1200:
            self.ctx.chunks = (len(self.ctx.generated_structures) // 1000) + 1
            self.report('Splitting similarity {} computations into {} separate jobs of 1000 each'.format(
                len(self.ctx.generated_structures), self.ctx.chunks))
            for chunk in range(self.ctx.chunks):
                if chunk < len(range(self.ctx.chunks)) - 1:
                    structure_list = self.ctx.generated_structures[chunk * 1000:(chunk + 1) * 1000]
                    print('---> ', chunk, len(structure_list))
                else:
                    structure_list = self.ctx.generated_structures[(chunk + 1) * 1000:]
                    print('LAAAAAST ---> ', chunk, len(structure_list), len(self.ctx.generated_structures), len(self.ctx.generated_structures)//100)
                builder = self._similarity_builder(csp_structures=structure_list, comparisons=self.ctx.comparisons, ml_model=self.ctx.ml_model)
                future = self.submit(builder)
                self.to_context(**{f"similarity_{chunk}": future})
        else:
            builder = self._similarity_builder(csp_structures=self.ctx.generated_structures, comparisons=self.ctx.comparisons,
                                               ml_model=self.ctx.ml_model)
            future = self.submit(builder)
            self.to_context(**{f"similarity": future})

    def inspect_similarity(self):
        failed_jobs = 0
        positive_matches = 0
        negative_matches = 0
        if self.ctx.chunks is None:
            wch = self.ctx[f"similarity"]
            if not wch.is_finished_ok:
                failed_jobs += 1
                self.report('Similarity computations in total failed')
                return

            similarities = wch.called[-1].outputs.output_dict['similarities']
            for index in similarities:
                similarity = float(similarities[index]['similarity']) ** self.ctx.similarity_zeta
                if similarity > 0.9:
                    ref_structure_dict = similarities[index]['comparison']
                    generated_structure_dict = similarities[index]['generated']
                    material_id = ref_structure_dict['properties']['material_id']
                    composition = Structure.from_dict(similarities[index]['generated']).composition
                    composition_string = re.sub(" ", "", str(composition))

                    print('NOCHUNK: ', index, self.ctx.chemical_system, similarity)

                    add_similarities(chemical_system=self.ctx.chemical_system, similarity=similarity,
                                     composition=composition_string,
                                     reference_material_id=material_id, reference_structure=ref_structure_dict,
                                     csp_structure=generated_structure_dict)
                    positive_matches += 1
                else:
                    negative_matches += 1
        else:
            for chunk in range(self.ctx.chunks):
                wch = self.ctx[f"similarity_{chunk}"]
                if not wch.is_finished_ok:
                    failed_jobs += 1
                    self.report('Similarity computations in chunk {} failed'.format(chunk))
                    return
                wch = self.ctx[f"similarity_{chunk}"]
                if not wch.is_finished_ok:
                    failed_jobs += 1
                    self.report('Similarity computations in chunk {} failed'.format(chunk))
                    return

                similarities = wch.called[-1].outputs.output_dict['similarities']
                for index in similarities:
                    similarity = float(similarities[index]['similarity']) ** self.ctx.similarity_zeta
                    if similarity > 0.9:
                        ref_structure_dict = similarities[index]['comparison']
                        generated_structure_dict = similarities[index]['generated']
                        material_id = ref_structure_dict['properties']['material_id']
                        composition = Structure.from_dict(similarities[index]['generated']).composition
                        composition_string = re.sub(" ", "", str(composition))

                        print('CHUNK: ', chunk, index, self.ctx.chemical_system, similarity)

                        add_similarities(chemical_system=self.ctx.chemical_system, similarity=similarity,
                                         composition=composition_string,
                                         reference_material_id=material_id, reference_structure=ref_structure_dict,
                                         csp_structure=generated_structure_dict)
                        positive_matches += 1
                    else:
                        negative_matches += 1

        self.ctx.success = True

    def final_report(self):
        if self.ctx.success:
            self.report(f"SimilarityWorkChain for {self.ctx.chemical_formula} completed")
        else:
            self.report(f"SimilarityWorkChain for {self.ctx.chemical_formula} failed")

    def _csp_builder(self):
        Workflow = WorkflowFactory("mattergen.csp")
        builder = Workflow.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.code = get_code("MatterGen")
        _, model_path, _ = get_model_device("MatterGenCSP")
        builder.job_info = Dict({"job_type": "csp",
                                 "model_path": model_path,
                                 "batch_size": settings.inputs["similarity_csp"]["batch_size"],
                                 "num_batches": settings.inputs["similarity_csp"]["num_batches"]})
        builder.max_iterations = Int(2)
        return builder

    def _gen_builder(self):
        Workflow = WorkflowFactory("mattergen.base")
        builder = Workflow.get_builder()
        builder.chemical_system = Str(self.ctx.chemical_system)
        builder.code = get_code("MatterGen")
        model_name, _, _ = get_model_device("MatterGen")
        builder.job_info = Dict({"job_type": "gen",
                                 "model_name": model_name,
                                 "batch_size": settings.inputs["similarity_csp"]["batch_size"],
                                 "num_batches": settings.inputs["similarity_csp"]["num_batches"],
                                 "energy_above_hull": settings.inputs["MatterGen_generate"]["energy_above_hull"]}
        )
        builder.max_iterations = Int(2)
        return builder

    @staticmethod
    def _similarity_builder(csp_structures, comparisons, ml_model):
        Workflow = WorkflowFactory('similarity_calc')
        builder = Workflow.get_builder()
        builder.input_structures = List(csp_structures)
        builder.comparison_structures = comparisons
        builder.code = get_code(ml_model)
        model, model_path, device = get_model_device(ml_model)
        relax_key = "similarity"
        job_info = {"job_type": relax_key, "ml_model": ml_model, "device": device,
                    "fmax": settings.inputs["similarity_relax"]["fmax"],
                    "max_steps": settings.inputs["similarity_relax"]["max_steps"]}
        if ml_model in ["uPET"]:
            job_info.update({"model_name": model})
        else:
            job_info.update({"model_path": model_path})
        builder.job_info = Dict(job_info)
        return builder
