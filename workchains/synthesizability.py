"""SynthesizabilityWorkChain -- classify all generated structures.

Runs three synthesizability views (thermo / reaction / PU) over every generated
structure of a composition, and -- when finite-T scoring is enabled -- first
computes phonon-derived vibrational free energies for the near-hull candidates
(and the hull-defining phases) so the classification is done on the Gibbs hull
at the synthesis temperature instead of the 0 K hull.

Finite-T sub-step (the one this stage adds):

    near-hull candidates + hull vertices
        -> PhononWorkChain (tight relax -> phonopy displacements -> MLIP forces
           -> force constants -> F_vib(T) -> QHA over volumes -> G(T))
        -> per-atom free-energy correction G(T)-E0 at the target temperature
        -> finite-T (Gibbs) convex hull
        -> classify candidates on the finite-T hull.

Heavy science is in ``_synth_classifiers`` (pure, unit-tested) and the phonon
runner ``codes/files/phonon.py`` (end-to-end tested with EMT); this module does
DB I/O, candidate selection, and the AiiDA wiring.

Phonons need a *conservative* MLIP (MACE / MatterSim); see input.yaml
``synthesizability.finite_T.model``.
"""

import os
import json
import tempfile

import numpy as np

from pymatgen.core import Composition
from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram

from aiida.orm import Str, Dict, List, Int, Float, SinglefileData
from aiida.engine import WorkChain, if_, calcfunction
from aiida.plugins import WorkflowFactory, CalculationFactory

from uvsib.db.tables import DBStructure, DBStructureVersion, DBComposition
from uvsib.db.session import get_session
from uvsib.db.utils import (
    update_row,
    query_by_columns,
    get_chemical_systems,
    update_version_attributes,
)
from uvsib.codes.utils import get_element_entries
from uvsib.workchains.utils import get_code, get_model_device, element_reference_entries
from uvsib.workchains.phase_diagram import get_entries_from_db
from uvsib.workchains._synth_classifiers import (
    classify_entries,
    summarize,
    load_pu_model,
    finite_T_entry,
    DEFAULTS,
)
from uvsib.workflows import settings

DFT_FUNC = settings.DFT_FUNC
_GENERATED_SOURCES = ("generated", "csp")

# The disordered-SQS generator runs remotely on the gnome@v100 code (reused as
# the generic v100 Python code), driven through the GNoME CalcJob with the
# runner + parser overridden via parameters/options. No new entry point needed.
GNoMECalculation = CalculationFactory("gnome")


# --------------------------------------------------------------------------- #
# DB / entry helpers
# --------------------------------------------------------------------------- #
def get_generated_versions(chemical_formula, method, sources=_GENERATED_SOURCES):
    """Return (structure_uuid, ComputedStructureEntry) for generated structures."""
    chemical_systems = get_chemical_systems(chemical_formula, new=False)
    out = []
    with get_session() as session:
        rows = (
            session.query(DBStructureVersion)
            .join(DBStructure)
            .filter(DBStructure.chemsys.in_(chemical_systems))
            .filter(DBStructureVersion.method == method)
            .filter(DBStructureVersion.source.in_(sources))
            .all()
        )
        for row in rows:
            struct = Structure.from_dict(row.structure)
            out.append((row.structure_uuid, ComputedStructureEntry(
                composition=struct.composition, structure=struct,
                energy=row.energy, data={"struct_uuid": row.structure_uuid})))
    return out


def gather_entries(chemical_formula, method):
    """Hull entries for this composition, all on the MLIP ``method``.

    The DB entries already carry the on-method elemental endpoints -- the
    references relaxed with the same MLIP (source='reference') and any generated
    elemental structures -- so the hull is method-consistent (no DFT mixing,
    which is what made pure elements sit above their own hull). Only elements
    still lacking an on-method elemental entry are filled (MLIP reference if
    present, else a logged DFT fallback) so the phase diagram can always build.
    """
    entries = get_entries_from_db(chemical_formula, method)
    if not entries:
        return None
    elements = [el.symbol for el in Composition(chemical_formula).elements]
    have_elem = {e.composition.chemical_system for e in entries
                 if len(e.composition.elements) == 1}
    missing = [el for el in elements if el not in have_elem]
    if missing:
        refs, _ = element_reference_entries(missing, method)
        entries = list(entries) + refs
    return entries


def _entry_key(entry):
    """Stable key for an entry: its struct_uuid, else comp:<reduced formula>."""
    su = entry.data.get("struct_uuid") if getattr(entry, "data", None) else None
    return str(su) if su else "comp:" + entry.composition.reduced_formula


def build_phase_diagram(entries):
    try:
        return PhaseDiagram(entries)
    except Exception:
        return None


def build_finite_T_pd(entries, corr):
    """Gibbs hull at T: shift each entry by its per-atom free-energy correction."""
    corrected = []
    for e in entries:
        c = corr.get(_entry_key(e), 0.0)
        corrected.append(finite_T_entry(e, c) if c else e)
    return build_phase_diagram(corrected)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def _classifier_params(job_info):
    params = dict(DEFAULTS)
    for src in (settings.inputs.get("synthesizability", {}) or {}, job_info or {}):
        for key in DEFAULTS:
            if key in src:
                params[key] = src[key]
    pu_model_path = (job_info or {}).get("pu_model_path") \
        or (settings.inputs.get("synthesizability", {}) or {}).get("pu_model_path")
    return params, pu_model_path


def _finite_T_config():
    return (settings.inputs.get("synthesizability", {}) or {}).get("finite_T", {}) or {}


def _finite_T_sqs_request(formula, ft):
    """Build the request dict for the remote disordered-SQS generator."""
    return {
        "formula": formula,
        "sqs_size": int(ft.get("sqs_size", 32)),
        "sqs_steps": int(ft.get("sqs_steps", 2000)),
        "n_realizations": max(1, int(ft.get("sqs_realizations", 3))),
        "lattice": ft.get("sqs_lattice", "fcc"),
        "a": float(ft.get("sqs_a", 3.8)),
        "cutoffs": list(ft.get("sqs_cutoffs", [7.0, 4.0])),
    }


def _sqs_request_file(request):
    """Stage the SQS request as sqs_request.json (SinglefileData)."""
    path = os.path.join(tempfile.gettempdir(), "sqs_request.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(request, f)
    return SinglefileData(file=path)


def _sqs_options():
    """Scheduler options for the remote SQS generator on the gnome@v100 code."""
    js = settings.configs["codes"]["GNoME"]["job_script"]
    options = {
        "resources": {
            "num_machines": js["nodes"],
            "num_mpiprocs_per_machine": js["ntasks"],
            "num_cores_per_mpiproc": js["cpus"],
        },
        "max_wallclock_seconds": js["time"],
        "parser_name": "sqs_parser",   # generic output.json -> Dict
    }
    if js.get("exclusive"):
        options["custom_scheduler_commands"] = "#SBATCH --exclusive"
    return options


def make_disordered_sqs(formula, supercell_atoms=32, lattice="fcc", n_steps=2000,
                        random_seed=None):
    """Disordered SQS at the target composition (icet) -- the disordered partner
    for the vibrational entropy of ordering. ``random_seed`` selects the
    realization (vary it to average over independent SQS). Returns a pymatgen
    Structure dict, or None if icet/ase are unavailable or generation fails.

    Reference / local-fallback implementation: the production path generates the
    SQS *remotely* on the gnome@v100 code via ``run_sqs`` (the icet search is too
    slow to run inline in a daemon step). ``codes/files/sqs_disordered.py`` is the
    remote twin of this routine."""
    try:
        from ase.build import bulk
        from icet import ClusterSpace
        from icet.tools.structure_generation import generate_sqs
        from pymatgen.io.ase import AseAtomsAdaptor
    except Exception:
        return None
    comp = Composition(formula)
    els = [el.symbol for el in comp.elements]
    if len(els) < 2:
        return None
    fracs = comp.fractional_composition.get_el_amt_dict()
    try:
        prim = bulk(els[0], lattice, a=3.8)
        cs = ClusterSpace(prim, [7.0, 4.0], [els])
        sqs = generate_sqs(cluster_space=cs, max_size=int(supercell_atoms),
                           target_concentrations={el: float(fracs[el]) for el in els},
                           n_steps=int(n_steps), T_start=5.0, T_stop=0.01,
                           random_seed=random_seed)
        return AseAtomsAdaptor().get_structure(sqs).as_dict()
    except Exception:
        return None


@calcfunction
def _synthesizability_output(payload):
    """Wrap the classification summary in a provenance-tracked Dict node.

    A WorkChain may only RETURN Data that already has a creator (a calcfunction
    output or one of its own inputs); a Dict built in the workchain body has no
    provenance and AiiDA rejects it. Re-emitting it from this calcfunction gives
    the returned node a creator so ``self.out`` is accepted."""
    return Dict(dict=payload.get_dict())


class SynthesizabilityWorkChain(WorkChain):
    """Classify all generated structures by synthesizability (0 K or finite-T)."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)
        spec.input("job_info", valid_type=Dict, required=False)
        spec.output("synthesizability", valid_type=Dict, required=False)

        spec.outline(
            cls.setup,
            cls.build_hull,
            if_(cls.should_run_phonons)(
                cls.run_sqs,        # remote disordered-SQS generation (gnome@v100)
                cls.inspect_sqs,    # fold the SQS realizations into the phonon set
                cls.run_phonons,
                cls.inspect_phonons,
            ),
            cls.classify,
            cls.results,
            cls.final_report,
        )

        spec.exit_code(301, "ERROR_NO_HULL",
                       message="Could not build a phase diagram for this composition")

    def setup(self):
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        job_info = self.inputs.job_info.get_dict() if "job_info" in self.inputs else {}
        self.ctx.params, self.ctx.pu_model_path = _classifier_params(job_info)
        self.ctx.model = settings.inputs["bulk_relax"]["model"]
        self.ctx.ft = _finite_T_config()
        self.ctx.corr = {}
        self.ctx.svib = {}            # uuid -> S_vib(T) [kB/atom]
        self.ctx.imag = {}            # uuid -> imaginary_modes flag
        self.ctx.dS_info = {}         # vibrational-entropy-of-ordering result
        self.ctx.ordered_uuid = None  # lowest-energy ordered polymorph at the target comp
        self.ctx.sqs_uuids = []       # disordered SQS realizations
        self.ctx.sqsgen = None        # remote SQS-generation CalcJob node
        self.ctx.sqs_request = None   # request dict for that job (None -> skip)
        self.report(f"Classifying synthesizability for {self.ctx.chemical_formula}")

    def build_hull(self):
        print('hull build')
        """Build the 0 K hull and select the near-hull set for phonons."""
        entries = gather_entries(self.ctx.chemical_formula, self.ctx.model)
        if not entries:
            self.report("Synthesizability: no entries to build the hull")
            return self.exit_codes.ERROR_NO_HULL
        pd = build_phase_diagram(entries)
        if pd is None:
            self.report("Synthesizability: could not build the phase diagram")
            return self.exit_codes.ERROR_NO_HULL

        self.ctx.entries = entries
        self.ctx.pd0 = pd
        self.ctx.generated = get_generated_versions(self.ctx.chemical_formula, self.ctx.model)

        # candidate set for phonons: near-hull generated structures + hull vertices
        window = float(self.ctx.ft.get("ehull_window", 0.10))
        cap = int(self.ctx.ft.get("max_phonon_structures", 40))
        phonon_items, seen = [], set()

        def _add(key, entry):
            if key in seen:
                return
            seen.add(key)
            phonon_items.append({"uuid": key, "structure": entry.structure.as_dict()})

        for uuid, entry in self.ctx.generated:
            try:
                if pd.get_e_above_hull(entry) <= window:
                    _add(str(uuid), entry)
            except Exception:
                continue
        for entry in pd.stable_entries:
            if getattr(entry, "structure", None) is not None:
                _add(_entry_key(entry), entry)

        self.ctx.phonon_items = phonon_items[:cap]

        # For the vibrational entropy of ordering, ensure both partners get
        # phonons: the lowest-energy *ordered* polymorph at the target
        # composition, and a *disordered* SQS at the same composition.
        if bool(self.ctx.ft.get("enabled", False)):
            tgt_rf = Composition(self.ctx.chemical_formula).reduced_formula
            ordered = None
            for uuid, entry in self.ctx.generated:
                if entry.composition.reduced_formula != tgt_rf:
                    continue
                if ordered is None or entry.energy_per_atom < ordered[2]:
                    ordered = (str(uuid), entry, entry.energy_per_atom)
            if ordered is not None:
                self.ctx.ordered_uuid = ordered[0]
                if not any(it["uuid"] == ordered[0] for it in self.ctx.phonon_items):
                    self.ctx.phonon_items.append(
                        {"uuid": ordered[0], "structure": ordered[1].structure.as_dict()})

            # Disordered partner(s) for the vibrational entropy of ordering.
            # The icet SQS search is too slow to run inline in a daemon step, so
            # it is generated *remotely* (run_sqs) on the gnome@v100 code and
            # folded into the phonon set afterwards (inspect_sqs). Only
            # meaningful for multi-element compositions.
            if len(Composition(self.ctx.chemical_formula).elements) >= 2:
                self.ctx.sqs_request = _finite_T_sqs_request(
                    self.ctx.chemical_formula, self.ctx.ft)
                self.report(
                    f"Synthesizability: will generate "
                    f"{self.ctx.sqs_request['n_realizations']} disordered SQS "
                    f"realization(s) ({self.ctx.sqs_request['sqs_size']} atoms) "
                    "remotely on gnome@v100 for the vibrational entropy of ordering")
            else:
                self.report("Synthesizability: single-element composition; no "
                            "disordered SQS / vibrational entropy of ordering")

    def should_run_phonons(self):
        return bool(self.ctx.ft.get("enabled", False)) and bool(self.ctx.phonon_items)

    def run_sqs(self):
        """Submit the disordered-SQS generation remotely on the gnome@v100 code.

        Reuses the GNoME CalcJob (runner + parser overridden via parameters /
        options) so no new entry point is needed. No-op when no SQS is requested
        (single-element composition)."""
        if not self.ctx.sqs_request:
            return
        req = self.ctx.sqs_request
        inputs = {
            "code": get_code("SQS_gen_phonon"),
            "parameters": Dict(dict={
                "cmdline_params": ["--request=sqs_request.json"],
                "staged_files": [["sqs_disordered.py", "aiida.py"]],
                "retrieve_list": ["output.json"],
            }),
            "file": {"sqs_request_file": _sqs_request_file(req)},
            "metadata": {
                "options": _sqs_options(),
                "label": f"SQS(disordered): {req['formula']}",
            },
        }
        self.to_context(sqsgen=self.submit(GNoMECalculation, **inputs))
        self.report(
            f"Synthesizability: submitted remote SQS generation "
            f"({req['n_realizations']} realizations x {req['sqs_size']} atoms, "
            f"{req['sqs_steps']} MC steps) on gnome@v100")

    def inspect_sqs(self):
        """Fold the remote SQS realizations into the phonon set."""
        node = self.ctx.sqsgen
        if node is None:
            return  # no SQS requested
        if not node.is_finished_ok:
            self.report("Synthesizability: remote SQS generation failed; "
                        "vibrational entropy of ordering will be skipped")
            return
        results = node.outputs.output_dict.get_dict().get("results", [])
        for rec in results:
            uid, struct = rec.get("uuid"), rec.get("structure")
            if not uid or struct is None:
                continue
            self.ctx.sqs_uuids.append(uid)
            self.ctx.phonon_items.append({"uuid": uid, "structure": struct})
        if self.ctx.sqs_uuids:
            self.report(f"Synthesizability: received {len(self.ctx.sqs_uuids)} "
                        "disordered SQS realization(s) from gnome@v100 for the "
                        "vibrational entropy of ordering")
        else:
            self.report("Synthesizability: remote SQS returned no structures; "
                        "vibrational entropy of ordering will be skipped")

    def run_phonons(self):
        """Submit one PhononWorkChain for the selected structures."""
        ft = self.ctx.ft
        model = ft.get("model", "MACE")              # conservative MLIP for phonons
        mname, mpath, device = get_model_device(model)
        job_info = {
            "ML_model": model, "model_name": mname, "model_path": mpath,
            "model_head": ft.get("head"), "device": device,
            "fmax": ft.get("fmax", 1e-3), "max_steps": ft.get("max_steps", 500),
            "min_supercell_length": ft.get("min_supercell_length", 8.0),
            "displacement": ft.get("displacement", 0.01),
            "mesh": ft.get("mesh", 20),
            "t_min": ft.get("t_min", 0.0), "t_max": ft.get("t_max", 1500.0),
            "t_step": ft.get("t_step", 50.0),
            "volume_scales": ft.get("volume_scales", [0.97, 0.99, 1.0, 1.01, 1.03]),
        }
        Workflow = WorkflowFactory("phonon")
        builder = Workflow.get_builder()
        builder.input_structure = List(self.ctx.phonon_items)
        builder.code = get_code(model)
        builder.job_info = Dict(dict=job_info)
        builder.local_label = Str(self.ctx.chemical_formula)
        builder.max_iterations = Int(2)
        self.to_context(phonon=self.submit(builder))

    def inspect_phonons(self):
        """Collect per-atom free-energy corrections at the target temperature."""
        wch = self.ctx.phonon
        if not wch.is_finished_ok:
            self.report("Synthesizability: phonon job failed; falling back to 0 K hull")
            return
        target_T = float(self.ctx.ft.get("temperature", 1000.0))
        results = wch.outputs.output_dict.get_dict().get("results", [])
        corr, svib, imag = {}, {}, {}
        for r in results:
            if "error" in r or "temperatures" not in r:
                continue
            uid = str(r["uuid"])
            temps = np.asarray(r["temperatures"], dtype=float)
            cvec = np.asarray(r["free_energy_correction_eV_per_atom"], dtype=float)
            corr[uid] = float(np.interp(target_T, temps, cvec))
            if "entropy_kB_per_atom" in r:
                svec = np.asarray(r["entropy_kB_per_atom"], dtype=float)
                svib[uid] = float(np.interp(target_T, temps, svec))
            imag[uid] = bool(r.get("imaginary_modes", False))
        self.ctx.corr = corr
        self.ctx.svib = svib
        self.ctx.imag = imag
        self.report(f"Synthesizability: collected {len(corr)} phonon free energies "
                    f"at {target_T:.0f} K")

        # Vibrational entropy of ordering, averaged over the SQS realizations:
        # S_vib(disordered) - S_vib(ordered). Realizations with imaginary modes
        # have unreliable S_vib, so prefer the clean ones (fall back to all if
        # none are clean), and report the spread + imaginary counts so the number
        # is trust-flagged.
        o = self.ctx.ordered_uuid
        sqs = [(u, svib[u], imag.get(u, False)) for u in self.ctx.sqs_uuids if u in svib]
        if o in svib and sqs:
            clean = [s for _, s, im in sqs if not im]
            used = clean if clean else [s for _, s, _ in sqs]
            s_dis_mean, s_dis_std = float(np.mean(used)), float(np.std(used))
            n_imag = sum(1 for _, _, im in sqs if im)
            self.ctx.dS_info = {
                "dS_vib": s_dis_mean - svib[o],
                "T_K": target_T,
                "S_ordered": round(svib[o], 4),
                "S_disordered_mean": round(s_dis_mean, 4),
                "S_disordered_std": round(s_dis_std, 4),
                "n_sqs_total": len(sqs),
                "n_sqs_used": len(used),
                "n_sqs_imaginary": n_imag,
                "ordered_imaginary": bool(imag.get(o, False)),
                "ordered_uuid": o,
                "sqs_uuids": list(self.ctx.sqs_uuids),
            }
            flag = ""
            if self.ctx.dS_info["ordered_imaginary"]:
                flag += " [ordered has imag modes]"
            if n_imag:
                flag += f" [{n_imag}/{len(sqs)} SQS have imag modes]"
            self.report(
                f"Synthesizability: dS_vib(disordered-ordered) = {self.ctx.dS_info['dS_vib']:+.3f} "
                f"kB/atom at {target_T:.0f} K (ordered S={svib[o]:.3f}, disordered "
                f"S={s_dis_mean:.3f}+/-{s_dis_std:.3f} over {len(used)}/{len(sqs)} SQS){flag}")
        elif o is not None or self.ctx.sqs_uuids:
            self.report("Synthesizability: could not form the ordered/disordered S_vib pair "
                        f"(ordered phononed: {o in svib}, SQS phononed: {len(sqs)})")

    def classify(self):
        formula, model = self.ctx.chemical_formula, self.ctx.model
        pd0 = self.ctx.pd0
        generated = self.ctx.generated
        if not generated:
            self.report("Synthesizability: no generated structures to classify")
            self.ctx.output = {"per_structure": [], "summary": {"n_total": 0, "counts": {}}}
            return

        pu_model = load_pu_model(self.ctx.pu_model_path)

        finite_T = None
        if self.ctx.corr:
            target_T = float(self.ctx.ft.get("temperature", 1000.0))
            pdT = build_finite_T_pd(self.ctx.entries, self.ctx.corr)
            if pdT is not None:
                gen_corr = {str(u): self.ctx.corr.get(str(u), 0.0) for u, _ in generated}
                finite_T = {"T": target_T, "pd": pdT, "corr": gen_corr}

        results = classify_entries(generated, pd0, params=self.ctx.params,
                                   pu_model=pu_model, finite_T=finite_T)

        target_T = float(self.ctx.ft.get("temperature", 1000.0))
        for r in results:
            if "combined" not in r:
                continue
            rec = {k: r[k] for k in ("thermo", "reaction", "pu", "combined")}
            uid = str(r["uuid"])
            if uid in self.ctx.svib:
                rec["S_vib_kB_per_atom"] = round(self.ctx.svib[uid], 4)
                rec["S_vib_T_K"] = target_T
                rec["imaginary_modes"] = bool(self.ctx.imag.get(uid, False))
            update_version_attributes(r["uuid"], model, {"synthesizability": rec})

        summary = summarize(results)
        summary["finite_T"] = None if finite_T is None else finite_T["T"]
        # vibrational entropy of ordering (Fultz): disordered SQS - ordered ground state,
        # averaged over SQS realizations with imaginary-mode/spread diagnostics.
        info = self.ctx.dS_info
        summary["dS_vib_ordering_kB_per_atom"] = info.get("dS_vib") if info else None
        if info:
            summary["dS_vib_T_K"] = info["T_K"]
            summary["S_vib_ordered_kB_per_atom"] = info["S_ordered"]
            summary["S_vib_disordered_kB_per_atom"] = info["S_disordered_mean"]
            summary["S_vib_disordered_std_kB_per_atom"] = info["S_disordered_std"]
            summary["n_sqs_realizations"] = info["n_sqs_total"]
            summary["n_sqs_used"] = info["n_sqs_used"]
            summary["n_sqs_imaginary"] = info["n_sqs_imaginary"]
            summary["ordered_imaginary_modes"] = info["ordered_imaginary"]
            summary["ordering_pair"] = {"ordered_uuid": info["ordered_uuid"],
                                        "sqs_uuids": info["sqs_uuids"]}
        self.ctx.output = {"per_structure": results, "summary": summary}

        try:
            row = query_by_columns(DBComposition, {"composition": formula})[0]
            attrs = dict(row.attributes or {})
            attrs["synthesizability"] = summary
            update_row(DBComposition, row.uuid, {"attributes": attrs})
        except Exception:
            self.report("Synthesizability: could not update DBComposition attributes")

        self.report(f"Synthesizability: {summary['counts']} over {summary['n_total']} "
                    f"structures (finite_T={summary['finite_T']})")

    def results(self):
        if "output" in self.ctx:
            self.out("synthesizability", _synthesizability_output(Dict(dict=self.ctx.output)))

    def final_report(self):
        self.report("SynthesizabilityWorkChain finished successfully")
