"""
BatteryNEBWorkChain: tier-2 ion-migration barriers for a battery composition.

Runs after BatteryWorkChain (it consumes the relaxed end members stored in
db_battery_path.configs -- nothing is re-enumerated or re-relaxed):

  1. batt_neb.enumerate_hops on the discharged supercell -> symmetry-distinct
     hops + the full hop graph (per host),
  2. endpoint pairs for the requested migration limits
     (vacancy: ion hops into a vacancy in the discharged cell;
      dilute: single ion in the relaxed empty host),
  3. ONE bundled job_type="neb" calcjob per host (shared engine
     codes/files/neb.py, same MLIP runners as everything else),
  4. barriers -> percolation thresholds (e_m 1D/2D/3D) -> db_battery_neb.

The percolation threshold in the transport-relevant limit is THE battery
number here: the lowest barrier at which the hop network wraps the cell.
"""
import numpy as np
from pymatgen.core import Structure
from monty.json import jsanitize
from aiida.engine import WorkChain
from aiida.orm import Str, List, Dict
from aiida.plugins import WorkflowFactory
from uvsib.db.tables import DBBatteryPath, DBBatteryNEB
from uvsib.db.utils import query_by_columns, add_row
from uvsib.workchains import batt, batt_neb
from uvsib.workchains.utils import get_code, get_model_device
from uvsib.workflows import settings


def _neb_cfg():
    """Knobs from input.yaml `battery: neb:` -- every key optional."""
    battery = settings.inputs.get("battery") or {}
    raw = battery.get("neb") or {}
    return {
        "model": raw.get("model"),          # None -> the battery row's model
        "head": raw.get("head", battery.get("head", "Default")),
        "n_images": int(raw.get("n_images", 5)),
        "fmax": float(raw.get("fmax", 0.05)),
        "max_steps": int(raw.get("max_steps", 400)),
        "spring": float(raw.get("spring", 0.1)),
        "climb": bool(raw.get("climb", True)),
        "prerelax_endpoints": bool(raw.get("prerelax_endpoints", True)),
        "max_hop": float(raw.get("max_hop", 4.5)),
        "max_hops": int(raw.get("max_hops", 6)),
        "limits": list(raw.get("limits", ["vacancy", "dilute"])),
    }


def _end_members(configs, working_ion):
    """(discharged Structure, charged Structure) from a db_battery_path
    configs list -- lowest-energy entry at max/zero ion count."""
    parsed = [(Structure.from_dict(c["structure"]), float(c["energy"]))
              for c in configs]
    counted = [(batt.n_ion(s, working_ion), s, e) for s, e in parsed]
    n_max = max(n for n, _, _ in counted)
    discharged = min((c for c in counted if c[0] == n_max), key=lambda c: c[2])[1]
    charged = min((c for c in counted if c[0] == 0), key=lambda c: c[2])[1]
    return discharged, charged


class BatteryNEBWorkChain(WorkChain):
    """Ion-migration barriers + percolation for one (composition, ion)."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)
        spec.input("working_ion", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.build_pairs,
            cls.run_neb,
            cls.analyze_and_store,
            cls.final_report,
        )

        spec.exit_code(301, "ERROR_NO_BATTERY_ROW",
                       message="No db_battery_path row (run BatteryWorkChain first)")
        spec.exit_code(302, "ERROR_NO_HOPS",
                       message="Hop enumeration produced nothing (raise max_hop)")
        spec.exit_code(303, "ERROR_NEB_FAILED",
                       message="All NEB calcjobs failed")
        spec.exit_code(304, "ERROR_ANALYSIS_FAILED",
                       message="No (host, limit) produced a stored result")

    def setup(self):
        """Load the battery rows (end members + model) for this ion."""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.working_ion = self.inputs.working_ion.value
        self.ctx.cfg = _neb_cfg()

        rows = query_by_columns(DBBatteryPath,
                                {"composition": self.ctx.chemical_formula,
                                 "working_ion": self.ctx.working_ion})
        if not rows:
            self.report(f"no db_battery_path rows for "
                        f"{self.ctx.chemical_formula}/{self.ctx.working_ion}")
            return self.exit_codes.ERROR_NO_BATTERY_ROW
        # newest row per host (retries append; ctime orders them)
        by_host = {}
        for row in rows:
            key = str(row.structure_uuid)
            if key not in by_host or row.ctime > by_host[key].ctime:
                by_host[key] = row
        self.ctx.hosts = [(uuid, row.configs, row.model)
                          for uuid, row in sorted(by_host.items())]
        self.report(f"BatteryNEB {self.ctx.chemical_formula}/"
                    f"{self.ctx.working_ion}: {len(self.ctx.hosts)} host(s), "
                    f"limits {self.ctx.cfg['limits']}")

    def build_pairs(self):
        """Enumerate hops per host; build consistently-ordered endpoint
        pairs for every requested limit."""
        cfg = self.ctx.cfg
        ion = self.ctx.working_ion
        self.ctx.jobs = []      # (host_uuid, model, pairs, graph_meta)
        for host_uuid, configs, model in self.ctx.hosts:
            try:
                discharged, charged = _end_members(configs, ion)
                distinct, edges = batt_neb.enumerate_hops(
                    discharged, ion, max_hop=cfg["max_hop"])
            except (ValueError, KeyError) as exc:
                self.report(f"host {host_uuid}: hop enumeration failed: {exc}")
                continue

            by_dist = sorted(distinct.items(), key=lambda kv: kv[1]["distance"])
            if len(by_dist) > cfg["max_hops"]:
                self.report(f"host {host_uuid}: capping {len(by_dist)} hop "
                            f"classes to the {cfg['max_hops']} shortest")
                by_dist = by_dist[:cfg["max_hops"]]
            run_distinct = dict(by_dist)

            pairs = []
            for limit in cfg["limits"]:
                for key, hop in run_distinct.items():
                    try:
                        if limit == "vacancy":
                            ini, fin, _ = batt_neb.hop_endpoints_vacancy(
                                discharged, hop, ion)
                        elif limit == "dilute":
                            ini, fin, _ = batt_neb.hop_endpoints_dilute(
                                charged,
                                discharged[hop["a"]].frac_coords,
                                discharged[hop["b"]].frac_coords,
                                hop["jimage"], ion)
                        else:
                            self.report(f"unknown limit '{limit}' -- skipped")
                            continue
                    except Exception as exc:
                        self.report(f"host {host_uuid} {limit} {key}: "
                                    f"endpoint construction failed: {exc}")
                        continue
                    pairs.append({"initial": ini.as_dict(),
                                  "final": fin.as_dict(),
                                  "tag": f"{host_uuid}|{limit}|{key}"})
            if not pairs:
                continue
            n_li_sites = len({e["a"] for e in edges})
            self.report(f"host {host_uuid}: {len(run_distinct)} hop class(es) "
                        f"x {len(cfg['limits'])} limit(s) = {len(pairs)} bands "
                        f"({n_li_sites} ion sites in the graph)")
            self.ctx.jobs.append((host_uuid, model,
                                  pairs, {"distinct": run_distinct,
                                          "edges": edges}))
        if not self.ctx.jobs:
            return self.exit_codes.ERROR_NO_HOPS

    def run_neb(self):
        """One bundled NEB calcjob per host (all limits together)."""
        for i, (host_uuid, model, pairs, _) in enumerate(self.ctx.jobs):
            builder = self._construct_neb_builder(pairs, model, host_uuid)
            self.to_context(**{f"neb_{i}": self.submit(builder)})

    def analyze_and_store(self):
        """Barriers -> percolation -> db_battery_neb rows."""
        ion = self.ctx.working_ion
        stored = 0
        failed_jobs = 0
        for i, (host_uuid, model, pairs, meta) in enumerate(self.ctx.jobs):
            wch = self.ctx[f"neb_{i}"]
            if not wch.is_finished_ok:
                self.report(f"NEB job for host {host_uuid} failed")
                failed_jobs += 1
                continue
            out = wch.outputs.output_dict.get_dict()
            by_tag = {r["tag"]: r for r in out.get("results", [])}

            for limit in self.ctx.cfg["limits"]:
                results = {}
                for key in meta["distinct"]:
                    res = by_tag.get(f"{host_uuid}|{limit}|{key}")
                    if res is not None:
                        results[key] = res
                if not results:
                    continue
                rows, barriers = batt_neb.hop_summary(meta["distinct"], results)
                # attach the TS structures for later DFT verification
                for row in rows:
                    res = results.get(row["class_key"])
                    row["images"] = (res or {}).get("images")
                th = batt_neb.percolation_thresholds(meta["edges"], barriers)

                add_row(DBBatteryNEB, {
                    "structure_uuid": host_uuid,
                    "composition": self.ctx.chemical_formula,
                    "working_ion": ion,
                    "hop_limit": limit,
                    "model": model,
                    "e_m_1d": th["e_m_1d"],
                    "e_m_2d": th["e_m_2d"],
                    "e_m_3d": th["e_m_3d"],
                    "hops": jsanitize(rows),
                    "attributes": {"model_head": self.ctx.cfg["head"],
                                   "n_hop_classes": len(meta["distinct"]),
                                   "n_converged": len(barriers),
                                   "cfg": {k: v for k, v in self.ctx.cfg.items()
                                           if k != "model"}},
                })
                stored += 1
                self.report(f"host {host_uuid} [{limit}]: "
                            f"{len(barriers)}/{len(meta['distinct'])} hops "
                            f"converged; e_m 1D/2D/3D = {th['e_m_1d']}/"
                            f"{th['e_m_2d']}/{th['e_m_3d']} eV")

        self.ctx.n_stored = stored
        if failed_jobs == len(self.ctx.jobs):
            return self.exit_codes.ERROR_NEB_FAILED
        if not stored:
            return self.exit_codes.ERROR_ANALYSIS_FAILED

    def final_report(self):
        self.report(f"BatteryNEBWorkChain {self.ctx.chemical_formula}/"
                    f"{self.ctx.working_ion} finished: {self.ctx.n_stored} "
                    "(host, limit) result(s) in db_battery_neb")

    ################################################################################
    def _construct_neb_builder(self, pairs, model, host_uuid):
        """Bundled NEB job on the shared engine -- identical plumbing to the
        relax builders, job_type 'neb'."""
        cfg = self.ctx.cfg
        ml_model = cfg["model"] or model or "MACE"
        Workflow = WorkflowFactory(ml_model.lower())
        builder = Workflow.get_builder()
        builder.input_structures = List(list(pairs))
        builder.code = get_code(ml_model)
        builder.local_label = Str(f"battery NEB {self.ctx.chemical_formula} "
                                  f"{self.ctx.working_ion} host {host_uuid[:8]}")
        model_name, model_path, device = get_model_device(ml_model)
        builder.job_info = Dict({
            "job_type": "neb",
            "ML_model": ml_model,
            "model_name": model_name,
            "model_path": model_path,
            "model_head": cfg["head"],
            "device": device,
            "fmax": cfg["fmax"],
            "max_steps": cfg["max_steps"],
            "n_images": cfg["n_images"],
            "spring": cfg["spring"],
            "climb": cfg["climb"],
            "prerelax_endpoints": cfg["prerelax_endpoints"],
        })
        return builder
