"""Electrode|electrolyte half-cell interfaces for a solid-state cell.

Two stages, cheap one first (docs/interfaces.md):

  1. THERMODYNAMIC SCREEN -- pseudo-binary interface reaction energy over the
     potential window (workchains/interface_stability.py, after Richards,
     Miara, Wang, Kim & Ceder, Chem. Mater. 28, 266 (2016)). Runs in-daemon:
     it is a phase-diagram query, not a simulation. Most pairs decompose here.
  2. GEOMETRIC BUILD -- Zur-McGill lattice matching for the survivors
     (codes/files/interface_build.py, submitted as a CalcJob).

Half-cells only. A single periodic cell holding `anode|electrolyte` and
`electrolyte|cathode` needs two simultaneous lattice matches and is polarised
by construction; the halves are coupled through mu_ion instead, exactly as the
CHE catalysis path couples through the electrode potential.

BOTH OUTCOMES ARE STORED. A pair that reacts gets a DBInterface row with
``built=False`` and its decomposition products: "this junction decomposes" is a
result, not an absence, and it is the answer for the interlayer/coating
question later.

Submission: the electrolytes ride in on the `interfaces` request block, e.g.
    {"electrolytes": ["Li7La3Zr2O12", "Li3PS4"], "half_cell": "cathode"}
"""
import json
import os
import tempfile

from pymatgen.core import Composition
from monty.json import jsanitize
from aiida.engine import WorkChain
from aiida.orm import Str, Dict, SinglefileData
from aiida.plugins import CalculationFactory

from uvsib.db.tables import DBComposition, DBInterface
from uvsib.db.utils import query_by_columns, get_structs_by_uuids, add_row
from uvsib.workchains import interface_stability, interface_frames
from uvsib.workchains.utils import get_code, get_model_device
from uvsib.workchains.phase_diagram import get_entries_from_db
from uvsib.workflows import settings

GNoMECalculation = CalculationFactory("gnome")


MODES = ("active_learning", "production")


def _interface_cfg():
    """`battery: interfaces:` block of input.yaml.

    Nested under `battery:` because interfaces are a branch of the battery
    tree -- the working ion, the electrode structures and the mu convention
    all come from there. Every key optional; MLIP settings fall back to the
    battery block, then bulk_relax.
    """
    batt = settings.inputs.get("battery") or {}
    raw = batt.get("interfaces") or {}
    bulk = settings.inputs.get("bulk_relax", {})
    mode = raw.get("mode", "active_learning")
    if mode not in MODES:
        # loud: a typo here would silently pick the wrong half of the path
        raise ValueError(f"battery: interfaces: mode must be one of {MODES}, "
                         f"got {mode!r}")
    return {
        "mode": mode,
        "model": raw.get("model", batt.get("model", bulk.get("model"))),
        "head": raw.get("head", batt.get("head", bulk.get("head"))),
        "build_code": raw.get("build_code", "sqs_cpu"),
        "relax_code": raw.get("relax_code", raw.get("model",
                                                    batt.get("model", "MACE"))),
        "film_layers": int(raw.get("film_layers", 3)),
        "substrate_layers": int(raw.get("substrate_layers", 2)),
        "max_atoms": int(raw.get("max_atoms", 2500)),
        "mu_grid": raw.get("mu_grid", [0.0, -1.0, -2.0, -3.0, -4.0]),
        "fmax": float(raw.get("fmax", 0.02)),
        "max_steps": int(raw.get("max_steps", 400)),
        "relax_all": bool(raw.get("relax_all", False)),
        "n_frames": int(raw.get("n_frames", 10)),
        "generation": int(raw.get("generation", 0)),
        "mongo": raw.get("mongo") or {},
    }


def _json_file(payload, name):
    tmp = os.path.join(tempfile.mkdtemp(), name)
    with open(tmp, "w") as fh:
        json.dump(jsanitize(payload, strict=True), fh)
    return SinglefileData(file=tmp)


class InterfaceWorkChain(WorkChain):
    """Half-cell junctions between one electrode composition and N electrolytes."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)
        spec.input("working_ion", valid_type=Str)
        spec.input("request", valid_type=Dict)

        spec.outline(
            cls.setup,
            cls.screen_pairs,
            cls.build_interfaces,
            cls.relax_interfaces,
            cls.store_results,
            cls.harvest_frames,
        )

        spec.exit_code(311, "ERROR_NO_ELECTRODE",
                       message="No relaxed electrode structure for this composition")
        spec.exit_code(312, "ERROR_NO_ELECTROLYTE",
                       message="The request names no electrolytes")
        spec.exit_code(313, "ERROR_SCREEN_FAILED",
                       message="The thermodynamic screen could not be evaluated")
        spec.exit_code(314, "ERROR_FRAME_PUSH_FAILED",
                       message="Harvested frames could not be written to MongoDB")

    def setup(self):
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.working_ion = self.inputs.working_ion.value
        self.ctx.request = self.inputs.request.get_dict()
        self.ctx.cfg = _interface_cfg()
        self.ctx.half_cell = self.ctx.request.get("half_cell", "cathode")

        self.ctx.electrolytes = list(self.ctx.request.get("electrolytes", []))
        if not self.ctx.electrolytes:
            self.report("interfaces request carries no 'electrolytes'")
            return self.exit_codes.ERROR_NO_ELECTROLYTE

        # electrode structures: same stable_struct manifest the battery and
        # surface-builder stages read, so an SQS sweep is covered for free
        rows = query_by_columns(DBComposition,
                                {"composition": self.ctx.chemical_formula})
        ss = rows[0].stable_struct if rows else None
        if not (ss and ss.get("ml_uuid_list")):
            self.report(f"no stable_struct manifest for "
                        f"{self.ctx.chemical_formula}")
            return self.exit_codes.ERROR_NO_ELECTRODE
        self.ctx.electrodes = get_structs_by_uuids(ss["ml_uuid_list"],
                                                   self.ctx.cfg["model"])
        if not self.ctx.electrodes:
            return self.exit_codes.ERROR_NO_ELECTRODE
        self.report(f"InterfaceWorkChain {self.ctx.chemical_formula} / "
                    f"{self.ctx.working_ion}: {len(self.ctx.electrodes)} "
                    f"electrode(s) x {len(self.ctx.electrolytes)} electrolyte(s)")

    def screen_pairs(self):
        """Stage 1. In-daemon: a phase-diagram query, not a simulation."""
        model = self.ctx.cfg["model"]
        self.ctx.screen = []
        for electrolyte in self.ctx.electrolytes:
            # ONE energy reference per hull: entries come from our own MLIP
            # relaxations, never mixed with MP's DFT entries. interface_
            # stability.assert_single_reference enforces it.
            # get_entries_from_db takes a FORMULA and expands to the relevant
            # chemical systems itself; the joint electrode+electrolyte formula
            # is what pulls in both sides plus the shared elements.
            joint = (Composition(self.ctx.chemical_formula)
                     + Composition(electrolyte)).reduced_formula
            entries = get_entries_from_db(joint, model) or []
            # Heads differ across stages by design (bulk_relax/battery on
            # matpes_r2scan, adsorbates/SQS on Default), so a method query
            # returns a mix even from a clean DB. Select this stage's head
            # rather than refusing the mix.
            entries, n_other = interface_stability.select_head(
                entries, self.ctx.cfg["head"])
            if n_other:
                self.report(f"{joint}: dropped {n_other} entr(ies) from other "
                            f"model heads, kept {len(entries)} on "
                            f"'{self.ctx.cfg['head']}'")
            if not entries:
                self.report(f"no MLIP entries for {joint} on head "
                            f"'{self.ctx.cfg['head']}'; the hull cannot be "
                            f"built, skipping {electrolyte}")
                continue
            try:
                res = interface_stability.screen_pair(
                    self.ctx.chemical_formula, electrolyte, entries,
                    self.ctx.working_ion, mu_grid=self.ctx.cfg["mu_grid"])
            except Exception as exc:
                # loud: a screen that cannot be evaluated must not look like a
                # pair that passed
                self.report(f"screen failed for {electrolyte}: "
                            f"{type(exc).__name__}: {exc}")
                continue
            self.ctx.screen.append(res)
            w = res["worst"]
            self.report(f"{self.ctx.chemical_formula}|{electrolyte}: "
                        f"dE_D={w['energy']:+.3f} eV/atom at mu={w['mu']} "
                        f"-> {'BUILD' if res['build_interface'] else 'reacts, skip'}")
        if not self.ctx.screen:
            return self.exit_codes.ERROR_SCREEN_FAILED

    def build_interfaces(self):
        """Stage 2. Only the survivors, as one bundled CalcJob."""
        survivors = [r for r in self.ctx.screen if r["build_interface"]]
        self.ctx.build_job = None
        if not survivors:
            self.report("no pair survived the thermodynamic screen; nothing "
                        "to build (the reacting pairs are still recorded)")
            return

        cfg = self.ctx.cfg
        pairs = []
        for res in survivors:
            electrolyte = res["electrolyte"]
            se_rows = query_by_columns(DBComposition, {"composition": electrolyte})
            se_ss = se_rows[0].stable_struct if se_rows else None
            if not (se_ss and se_ss.get("ml_uuid_list")):
                self.report(f"electrolyte {electrolyte} has no relaxed "
                            f"structure in the DB -- skipping its build")
                continue
            se_structs = get_structs_by_uuids(se_ss["ml_uuid_list"], cfg["model"])
            if not se_structs:
                continue
            se_dict, se_uuid = se_structs[0]
            for el_dict, el_uuid in self.ctx.electrodes:
                pairs.append({
                    "label": f"{self.ctx.chemical_formula}|{electrolyte}",
                    "film": {"uuid": str(el_uuid), "structure": el_dict},
                    "substrate": {"uuid": str(se_uuid), "structure": se_dict},
                    "film_layers": cfg["film_layers"],
                    "substrate_layers": cfg["substrate_layers"],
                    "max_atoms": cfg["max_atoms"],
                })
        if not pairs:
            self.report("survivors had no relaxed electrolyte structures")
            return

        inputs = {
            "code": get_code(cfg["build_code"]),
            "parameters": Dict(dict={
                "staged_files": [["interface_build.py", "aiida.py"]],
                "retrieve_list": ["output.json"],
            }),
            "file": {"request": _json_file({"pairs": pairs},
                                           "input_structures.json")},
            "metadata": {"label": f"interface build: "
                                  f"{self.ctx.chemical_formula} "
                                  f"({len(pairs)} pair(s))"},
        }
        self.to_context(build_job=self.submit(GNoMECalculation, **inputs))
        self.report(f"submitted {len(pairs)} interface build(s) on "
                    f"'{cfg['build_code']}'")

    def _collect_built(self):
        """Flatten the build output into the relax runner's input shape."""
        built, by_label = [], {}
        job = self.ctx.get("build_job")
        if job is None or not job.is_finished_ok:
            return built, by_label
        try:
            out = job.outputs.output_dict.get_dict()
        except Exception as exc:
            self.report(f"cannot read interface build output: "
                        f"{type(exc).__name__}: {exc}")
            return built, by_label
        for res in out.get("results", []):
            for k, d in enumerate(res.get("interfaces", [])):
                uid = f"{res['label']}#{k}"
                built.append({"uuid": uid, "label": res["label"],
                              "structure": d["structure"],
                              "active_mask": d["active_mask"]})
                by_label[uid] = (res, d)
        return built, by_label

    def _prepare_built(self):
        built, by_label = self._collect_built()
        self.ctx.built_interfaces = built
        self.ctx.built_index = by_label
        return built

    def relax_interfaces(self):
        """Stage 3. MLIP relaxation at FIXED CELL, frozen bulk.

        Both modes run it -- production needs the converged endpoint for the
        NEB, active_learning needs the trajectory. The runner decides what to
        emit from `params.mode`.
        """
        self.ctx.relax_job = None
        built = self._prepare_built()
        if not built:
            self.report("no interface was built; nothing to relax")
            return

        cfg = self.ctx.cfg
        payload = {
            "params": {"fmax": cfg["fmax"], "max_steps": cfg["max_steps"],
                       "n_frames": cfg["n_frames"],
                       "relax_all": cfg["relax_all"], "mode": cfg["mode"]},
            "interfaces": built,
        }
        model = cfg["relax_code"]
        mdl, mdl_path, device = get_model_device(model)
        inputs = {
            "code": get_code(model),
            "parameters": Dict(dict={
                "cmdline_params": [f"--ML_model={model}", f"--model={mdl}",
                                   f"--model_path={mdl_path}",
                                   f"--device={device}",
                                   f"--task_name={cfg['head']}"],
                "staged_files": [["interface_relax.py", "aiida.py"],
                                 ["_calculators.py", "_calculators.py"]],
                "retrieve_list": ["output.json", "opt.log"],
            }),
            "file": {"request": _json_file(payload, "input_structures.json")},
            "metadata": {"label": f"interface relax [{cfg['mode']}]: "
                                  f"{self.ctx.chemical_formula} "
                                  f"({len(built)} junction(s))"},
        }
        self.to_context(relax_job=self.submit(GNoMECalculation, **inputs))
        self.report(f"submitted relaxation of {len(built)} junction(s) in "
                    f"{cfg['mode']} mode on '{model}'")

    def store_results(self):
        """One DBInterface row per outcome -- reacting pairs included."""
        model = self.ctx.cfg["model"]
        built = {}
        job = self.ctx.get("build_job")
        if job is not None and job.is_finished_ok:
            try:
                out = job.outputs.output_dict.get_dict()
                for res in out.get("results", []):
                    built.setdefault(res["label"], []).extend(res["interfaces"])
            except Exception as exc:
                self.report(f"cannot read interface build output: "
                            f"{type(exc).__name__}: {exc}")
        elif job is not None:
            self.report(f"interface build failed (exit {job.exit_status}); "
                        f"the screen results are still stored")

        # relaxed geometries, keyed by the build uuid
        relaxed = {}
        rjob = self.ctx.get("relax_job")
        if rjob is not None and rjob.is_finished_ok:
            try:
                for r in rjob.outputs.output_dict.get_dict().get("results", []):
                    relaxed[r.get("uuid")] = r
            except Exception as exc:
                self.report(f"cannot read interface relax output: "
                            f"{type(exc).__name__}: {exc}")
        elif rjob is not None:
            self.report(f"interface relaxation failed (exit {rjob.exit_status}); "
                        f"storing the UNRELAXED geometries")
        self.ctx.relaxed = relaxed

        n_rows = 0
        for res in self.ctx.screen:
            w = res["worst"]
            base = {
                "composition": self.ctx.chemical_formula,
                "electrode": res["electrode"],
                "electrolyte": res["electrolyte"],
                "working_ion": self.ctx.working_ion,
                "half_cell": self.ctx.half_cell,
                "model": model,
                "reaction_energy": w["energy"],
                "reaction_products": jsanitize(w["products"]),
                "reacts": w["reacts"],
                "severe": w["severe"],
                "mu_worst": w["mu"],
                "reaction_scan": jsanitize(res["scan"]),
            }
            label = f"{self.ctx.chemical_formula}|{res['electrolyte']}"
            ifaces = built.get(label, [])
            if not ifaces:
                add_row(DBInterface, dict(base, built=False))
                n_rows += 1
                continue
            for k, d in enumerate(ifaces):
                uid = f"{label}#{k}"
                rel = relaxed.get(uid)
                # structure_uuid stays NULL: even relaxed, this is an MLIP
                # geometry with an MLIP energy, and a db_structure row implies
                # a computed (reference) energy.
                add_row(DBInterface, dict(
                    base, built=True,
                    structure=jsanitize((rel or {}).get("structure")
                                        or d["structure"], strict=True),
                    film_miller=d["film_miller"],
                    substrate_miller=d["substrate_miller"],
                    termination=jsanitize(d["termination"]),
                    n_atoms=d["n_atoms"], area=d["area"],
                    strain_percent=d["strain_percent"],
                    active_mask=d["active_mask"],
                    attributes={
                        "n_active": int(sum(d["active_mask"])),
                        "build_uuid": uid,
                        "relaxed": bool(rel),
                        "relax_converged": (rel or {}).get("converged"),
                        "relax_steps": (rel or {}).get("n_steps"),
                        "fmax_final": (rel or {}).get("fmax_final"),
                        "mlip_energy": (rel or {}).get("energy"),
                        "mode": self.ctx.cfg["mode"]}))
                n_rows += 1
        self.report(f"stored {n_rows} DBInterface row(s)")

    def harvest_frames(self):
        """Active-learning mode only: push relaxation frames to MongoDB.

        In `production` mode this is a no-op -- the point there is the
        converged endpoint and the NEB that follows, not training data.

        A failed push is an ERROR, not a warning: the frames exist only in
        this workchain's outputs, and a silent failure means a generation of
        DFT input is quietly lost.
        """
        cfg = self.ctx.cfg
        if cfg["mode"] != "active_learning":
            self.report(f"mode={cfg['mode']}: no frames harvested")
            return

        relaxed = self.ctx.get("relaxed") or {}
        index = self.ctx.get("built_index") or {}
        if not relaxed:
            self.report("no relaxation results; nothing to harvest")
            return

        mongo = cfg.get("mongo") or {}
        if not mongo.get("host") or not mongo.get("db_name"):
            self.report("battery: interfaces: mongo: is not configured "
                        "(host/db_name); cannot push frames")
            return self.exit_codes.ERROR_FRAME_PUSH_FAILED

        docs, n_unconv = [], 0
        for uid, res in relaxed.items():
            if res.get("error"):
                continue
            if not res.get("converged"):
                # KEEP them: an unconverged interface relaxation is still a
                # legitimate set of geometries, and its high-force frames are
                # exactly what the model is worst at. Only the ENDPOINT would
                # be misleading, and that is not what we label.
                n_unconv += 1
            built = index.get(uid)
            if not built:
                continue
            screen_res, d = built
            base = {
                "composition": self.ctx.chemical_formula,
                "electrode": screen_res["electrode"],
                "electrolyte": screen_res["electrolyte"],
                "working_ion": self.ctx.working_ion,
                "half_cell": self.ctx.half_cell,
                "label": res.get("label"),
                "interface_uuid": uid,
                "film_miller": d["film_miller"],
                "substrate_miller": d["substrate_miller"],
                "termination": jsanitize(d["termination"]),
            }
            docs.extend(interface_frames.frame_documents(
                base, res, cfg["generation"], cfg["model"], cfg["head"]))

        if not docs:
            self.report("relaxations produced no frames to harvest")
            return

        try:
            n = interface_frames.push_frames(mongo, jsanitize(docs, strict=True))
        except Exception as exc:
            self.report(f"MongoDB push FAILED: {type(exc).__name__}: {exc}")
            return self.exit_codes.ERROR_FRAME_PUSH_FAILED

        self.report(f"pushed {n} frame(s) of generation {cfg['generation']} to "
                    f"{mongo['host']}/{mongo['db_name']}."
                    f"{mongo.get('collection', 'interface_frames')}"
                    + (f" ({n_unconv} junction(s) had not converged; their "
                       f"frames are kept deliberately)" if n_unconv else ""))
