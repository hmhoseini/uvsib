"""
PhotocatWorkChain: stage-1 photocatalysis gap filter after the catalysis
chain.

For one (composition, reaction, reaction_path) whose adsorbates stage is
Done, it:

  1. ranks the slabs by their BEST (lowest) eta and takes the top
     ``n_slabs`` (input.yaml ``photocat.n_slabs``),
  2. predicts experimental-target band gaps for the PARENT BULK structures
     of those slabs (bulk-trained models on a slab-with-vacuum would be
     off-distribution garbage -- the slab gets the tag, the bulk gets the
     prediction) with the ensemble runner (codes/files/photocat_gap.py:
     ALIGNN-MBJ primary, MEGNet/MODNet/ALIGNN-OPT as available),
  3. writes the record back so export_all.py picks it up:
       DBSurface.attributes["photocat"][reaction][reaction_path] =
           {eta, gap ensemble record, failure probability, flags}
     (atomic nested jsonb_set -- parallel reactions tagging one slab never
     clobber each other), plus the bulk-level record on
     DBStructureVersion.attributes["photocat_gap"].

Stage 2 -- HSE on the shortlist -- is the USER's job, never automated here.
"""
from aiida.engine import WorkChain
from aiida.orm import Str, List, Dict
from aiida.plugins import WorkflowFactory
from monty.json import jsanitize
from uvsib.db.tables import DBSurfaceMLAdsorbate, DBSurface
from uvsib.db.utils import (query_by_columns, query_structureversions_by_attributes,
                            update_jsonb_path, update_version_attributes)
from uvsib.workchains.utils import get_code
from uvsib.workflows import settings


def _photocat_cfg():
    """Knobs from input.yaml `photocat:` -- enabled is gated in main.py."""
    raw = settings.inputs.get("photocat") or {}
    gap_max = raw.get("gap_max")
    return {
        "gap_min": float(raw.get("gap", 1.5)),
        "gap_max": None if gap_max in (None, "none", "None") else float(gap_max),
        "n_slabs": int(raw.get("n_slabs", 10)),
        "models": list(raw.get("models",
                               ["alignn_mbj", "alignn_opt", "megnet_pbe",
                                "modnet_expt"])),
        "sigma_model": float(raw.get("sigma_model", 0.5)),
    }


class PhotocatWorkChain(WorkChain):
    """Gap-filter tagging of the best-eta slabs for one (reaction, path)."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)
        spec.input("reaction", valid_type=Str)
        spec.input("reaction_path", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.run_predict,
            cls.collect_and_tag,
            cls.final_report,
        )

        spec.exit_code(301, "ERROR_NO_ETA_ROWS",
                       message="No adsorbate rows with eta for this (reaction, path)")
        spec.exit_code(302, "ERROR_NO_BULK_STRUCTURES",
                       message="No stored bulk structures behind the tagged slabs")
        spec.exit_code(303, "ERROR_PREDICT_FAILED",
                       message="The gap-prediction calcjob failed")
        spec.exit_code(304, "ERROR_NOTHING_TAGGED",
                       message="No slab could be tagged with a gap record")

    def setup(self):
        """Rank slabs by best eta; load the parent bulk structures."""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.reaction = self.inputs.reaction.value
        self.ctx.reaction_path = self.inputs.reaction_path.value
        self.ctx.cfg = _photocat_cfg()
        model = settings.inputs['bulk_relax']['model']

        rows = query_by_columns(DBSurfaceMLAdsorbate,
                                {"composition": self.ctx.chemical_formula,
                                 "reaction": self.ctx.reaction,
                                 "reaction_path": self.ctx.reaction_path})
        best = {}                      # surface_id -> (eta, row)
        for row in rows:
            if row.eta is None:
                continue
            if row.surface_id not in best or row.eta < best[row.surface_id][0]:
                best[row.surface_id] = (row.eta, row)
        if not best:
            self.report(f"no eta rows for {self.ctx.chemical_formula} "
                        f"{self.ctx.reaction}/{self.ctx.reaction_path}")
            return self.exit_codes.ERROR_NO_ETA_ROWS

        ranked = sorted(best.items(), key=lambda kv: kv[1][0])
        ranked = ranked[:self.ctx.cfg["n_slabs"]]

        self.ctx.slab_tags = []        # per tagged slab
        bulk_by_uuid = {}              # one prediction per unique bulk
        for surface_id, (eta, row) in ranked:
            uuid = str(row.structure_uuid)
            self.ctx.slab_tags.append({
                "surface_id": surface_id,
                "structure_uuid": uuid,
                "miller_index": row.surface_miller_index,
                "eta": float(eta),
            })
            if uuid not in bulk_by_uuid:
                versions = query_structureversions_by_attributes(
                    structure_uuid=uuid, method=model)
                if versions:
                    version = min(versions,
                                  key=lambda v: v.energy if v.energy is not None
                                  else float("inf"))
                    bulk_by_uuid[uuid] = version.structure
                else:
                    self.report(f"no {model} bulk version for {uuid}; "
                                "its slabs will carry no gap record")
        if not bulk_by_uuid:
            return self.exit_codes.ERROR_NO_BULK_STRUCTURES

        self.ctx.bulk_model = model
        self.ctx.bundle = [{"structure": s, "tag": uuid}
                           for uuid, s in sorted(bulk_by_uuid.items())]
        self.report(f"photocat {self.ctx.chemical_formula} "
                    f"{self.ctx.reaction}/{self.ctx.reaction_path}: "
                    f"{len(self.ctx.slab_tags)} slab(s) -> "
                    f"{len(self.ctx.bundle)} unique bulk structure(s), "
                    f"window [{self.ctx.cfg['gap_min']}, "
                    f"{self.ctx.cfg['gap_max']}] eV, "
                    f"models {self.ctx.cfg['models']}")

    def run_predict(self):
        """One ensemble calcjob over the unique bulk structures."""
        cfg = self.ctx.cfg
        Workflow = WorkflowFactory("photocat_calc")
        builder = Workflow.get_builder()
        builder.input_structures = List(list(self.ctx.bundle))
        builder.code = get_code("photocat")
        builder.local_label = Str(f"{self.ctx.chemical_formula} "
                                  f"{self.ctx.reaction}/{self.ctx.reaction_path}")
        builder.job_info = Dict({
            "job_type": "photocat",
            "models": cfg["models"],
            "gap_min": cfg["gap_min"],
            "gap_max": cfg["gap_max"],
            "sigma_model": cfg["sigma_model"],
        })
        self.to_context(**{"predict": self.submit(builder)})

    def collect_and_tag(self):
        """Write the gap records back onto the slabs (and their bulks)."""
        wch = self.ctx.predict
        if not wch.is_finished_ok:
            self.report("gap-prediction calcjob failed")
            return self.exit_codes.ERROR_PREDICT_FAILED
        out = wch.outputs.output_dict.get_dict()
        by_uuid = {r["tag"]: r for r in out.get("results", [])}

        tagged = 0
        for slab in self.ctx.slab_tags:
            rec = by_uuid.get(slab["structure_uuid"])
            if rec is None:
                self.report(f"no gap record for bulk {slab['structure_uuid']} "
                            f"(surface {slab['surface_id']}) -- not tagged")
                continue
            payload = jsanitize({**{k: v for k, v in rec.items() if k != "tag"},
                                 "eta": slab["eta"],
                                 "miller_index": slab["miller_index"]})
            # atomic nested write: sibling reactions tagging this slab keep
            # their leaves
            update_jsonb_path(DBSurface, slab["surface_id"],
                              ["photocat", self.ctx.reaction,
                               self.ctx.reaction_path],
                              payload, column="attributes", pk_column="id")
            tagged += 1

        # bulk-level record (once per unique structure)
        for item in self.ctx.bundle:
            rec = by_uuid.get(item["tag"])
            if rec is not None:
                update_version_attributes(
                    item["tag"], self.ctx.bulk_model,
                    {"photocat_gap": jsanitize(
                        {k: v for k, v in rec.items() if k != "tag"})})

        self.ctx.n_tagged = tagged
        if not tagged:
            return self.exit_codes.ERROR_NOTHING_TAGGED
        for slab in self.ctx.slab_tags:
            rec = by_uuid.get(slab["structure_uuid"])
            if rec and rec.get("gap_mean") is not None:
                self.report(
                    f"surface {slab['surface_id']} (eta {slab['eta']:.3f} V): "
                    f"gap {rec['gap_mean']:.2f} eV, p_fail {rec['p_fail']:.2f}"
                    f"{', flags ' + ','.join(rec['flags']) if rec.get('flags') else ''}")

    def final_report(self):
        self.report(f"PhotocatWorkChain {self.ctx.chemical_formula} "
                    f"{self.ctx.reaction}/{self.ctx.reaction_path} finished: "
                    f"{self.ctx.n_tagged} slab(s) tagged "
                    "(stage 2 = HSE on the shortlist, by hand)")
