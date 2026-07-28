"""
BatteryWorkChain: deintercalation electrode characteristics for a composition
whose gen/csp + phase-diagram stages have finished.

Bulk analogue of the catalysis branch (surface builder + adsorbates are
skipped for battery submissions). For every stable host structure containing
the working ion it:

  1. enumerates vacancy orderings on the ion sublattice across the x grid
     on the REMOTE CPU queue named by input.yaml ``battery: enum_code:``
     (default sqs_cpu, the same code the synthesizability finite_T SQS
     generator uses) -- the Ewald ranking is far too CPU-heavy for the
     daemon host; one bundled job covers all hosts, results are collected
     by echoed host uuid (in-daemon fallback, loudly, only when the code
     is not registered so local smoke setups keep working),
  2. relaxes all configurations in ONE bundled MLIP relax per host (the same
     relax workchains the gen/csp paths use; the elemental working-ion
     reference is bundled into the first host's job via the standard
     missing_element_references machinery, so the anode reference is
     on-method),
  3. computes voltage profile, capacities, energy density, volume change and
     the framework-integrity / endpoint-stability flags (batt.py, pure
     python),
  4. stores one DBBatteryPath row per host (Postgres -- results never live
     only in the AiiDA storage).

Submission: {"composition": "LiFePO4", "reaction": "BATTERY",
             "reaction_path": "Li"} -- the reaction path carries the ion.
"""
import json
import os
import tempfile

from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from monty.json import jsanitize
from aiida.engine import WorkChain
from aiida.orm import Str, List, Dict, SinglefileData
from aiida.plugins import WorkflowFactory, CalculationFactory
from uvsib.db.tables import DBComposition, DBBatteryPath
from uvsib.db.utils import (query_by_columns, query_structure,
                            get_structs_by_uuids, add_row, add_structures)
from uvsib.workchains import batt, battery_enum
from uvsib.workchains.utils import (get_code, get_model_device,
                                    split_relax_output,
                                    element_reference_entries,
                                    missing_element_references)
from uvsib.codes.utils import get_mp_element_structures
from uvsib.workflows import settings

GNoMECalculation = CalculationFactory("gnome")


def _enum_options(code_name):
    """Scheduler options for the remote enumeration job (mirrors the
    synthesizability finite_T _sqs_options; generic sqs_parser)."""
    js = settings.configs["codes"][code_name]["job_script"]
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


def _json_file(payload, filename):
    """Stage a dict as a named SinglefileData (own tmpdir: no clobber when
    several BatteryWorkChains submit concurrently)."""
    path = os.path.join(tempfile.mkdtemp(prefix="battery_enum_"), filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return SinglefileData(file=path)


def _battery_cfg():
    """Knobs from input.yaml `battery:` -- every key optional, MLIP settings
    default to bulk_relax so the module runs without the block."""
    raw = settings.inputs.get("battery") or {}
    bulk = settings.inputs.get("bulk_relax", {})
    return {
        "model": raw.get("model", bulk.get("model", "MACE")),
        "head": raw.get("head", bulk.get("head", "Default")),
        "fmax": raw.get("fmax", bulk.get("fmax", 0.05)),
        "max_steps": raw.get("max_steps", bulk.get("max_steps", 200)),
        "n_x_steps": int(raw.get("n_x_steps", 4)),
        "max_configs_per_x": int(raw.get("max_configs_per_x", 8)),
        "supercell_max_atoms": int(raw.get("supercell_max_atoms", 128)),
        "max_hosts": int(raw.get("max_hosts", 3)),
        "enum_code": raw.get("enum_code", "sqs_cpu"),
    }


def _host_structs(chemical_formula, model, max_hosts, working_ion):
    """(structure_dict, uuid) pairs to sweep: the stable_struct manifest when
    present (same convention as the surface builder), else the legacy
    composition+method query; filtered to hosts that contain the ion."""
    rows = query_by_columns(DBComposition, {"composition": chemical_formula})
    ss = rows[0].stable_struct if rows else None
    if ss and ss.get("ml_uuid_list"):
        pairs = get_structs_by_uuids(ss["ml_uuid_list"], model)
    else:
        results = query_structure({"composition": chemical_formula}, method=model)
        pairs = sorted(((row.structure, str(row.structure_uuid))
                        for row in results), key=lambda x: x[1])
    hosts = []
    for struct_dict, uuid in pairs:
        struct = Structure.from_dict(struct_dict)
        if any(site.specie.symbol == working_ion for site in struct):
            hosts.append((struct_dict, uuid))
        if len(hosts) == max_hosts:
            break
    return hosts


class BatteryWorkChain(WorkChain):
    """Deintercalation battery characteristics for one (composition, ion)."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)
        spec.input("working_ion", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.run_enum,
            cls.inspect_enum,
            cls.relax_configs,
            cls.analyze_and_store,
            cls.final_report,
        )

        spec.exit_code(301, "ERROR_NO_HOST",
                       message="No stable host containing the working ion (or unsupported ion)")
        spec.exit_code(302, "ERROR_ENUMERATION_FAILED",
                       message="Vacancy-ordering enumeration produced no configurations")
        spec.exit_code(303, "ERROR_RELAX_FAILED",
                       message="All configuration relaxations failed")
        spec.exit_code(304, "ERROR_ANALYSIS_FAILED",
                       message="No host produced a valid battery summary")

    def setup(self):
        """Load config and the stable host structures containing the ion."""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.working_ion = self.inputs.working_ion.value
        self.ctx.cfg = _battery_cfg()

        if self.ctx.working_ion not in batt.ION_Z:
            self.report(f"Unsupported working ion '{self.ctx.working_ion}' "
                        f"(known: {sorted(batt.ION_Z)})")
            return self.exit_codes.ERROR_NO_HOST

        self.ctx.hosts = _host_structs(
            self.ctx.chemical_formula, self.ctx.cfg["model"],
            self.ctx.cfg["max_hosts"], self.ctx.working_ion)
        if not self.ctx.hosts:
            self.report(f"No stable structure of {self.ctx.chemical_formula} "
                        f"contains {self.ctx.working_ion}")
            return self.exit_codes.ERROR_NO_HOST
        self.report(f"BatteryWorkChain {self.ctx.chemical_formula} / "
                    f"{self.ctx.working_ion}: {len(self.ctx.hosts)} host(s)")

    def run_enum(self):
        """Submit the vacancy-ordering enumeration for ALL hosts as one
        bundled job on the remote CPU queue (battery.enum_code, default
        sqs_cpu). The pure module workchains/battery_enum.py is shipped
        verbatim via the file namespace -- single canonical copy, nothing
        to keep in sync. In-daemon fallback (loud) only when the code is
        not registered, so local smoke setups keep working."""
        cfg = self.ctx.cfg
        code_name = cfg["enum_code"]
        self.ctx.enum_remote = code_name in (settings.configs.get("codes") or {})
        if not self.ctx.enum_remote:
            self.report(f"battery enum_code '{code_name}' is not registered "
                        "in config.yaml codes -- enumerating IN-DAEMON "
                        "(CPU-heavy on the host; register the code for "
                        "production)")
            return

        request = {
            "working_ion": self.ctx.working_ion,
            "params": {"n_x_steps": cfg["n_x_steps"],
                       "max_configs_per_x": cfg["max_configs_per_x"],
                       "supercell_max_atoms": cfg["supercell_max_atoms"]},
            "hosts": [{"uuid": uuid, "structure": struct_dict}
                      for struct_dict, uuid in self.ctx.hosts],
        }
        inputs = {
            "code": get_code(code_name),
            "parameters": Dict(dict={
                "cmdline_params": ["--request=battery_enum_request.json"],
                "staged_files": [["battery_enum_job.py", "aiida.py"]],
                "retrieve_list": ["output.json"],
            }),
            "file": {
                "request": _json_file(request, "battery_enum_request.json"),
                "module": SinglefileData(file=battery_enum.__file__),
            },
            "metadata": {
                "options": _enum_options(code_name),
                "label": f"battery enum: {self.ctx.chemical_formula} "
                         f"{self.ctx.working_ion}",
            },
        }
        self.to_context(battery_enum_job=self.submit(GNoMECalculation, **inputs))
        self.report(f"submitted remote enumeration of {len(self.ctx.hosts)} "
                    f"host(s) on '{code_name}'")

    def inspect_enum(self):
        """Collect the enumeration into ctx.plans = [(uuid, bundle), ...],
        attributed by ECHOED host uuid (never by position)."""
        self.ctx.plans = []
        if not self.ctx.enum_remote:
            self._enumerate_in_daemon()
        else:
            node = self.ctx.battery_enum_job
            if not node.is_finished_ok:
                self.report("remote enumeration job failed (see the "
                            "calculation node for the scheduler output)")
                return self.exit_codes.ERROR_ENUMERATION_FAILED
            out = node.outputs.output_dict.get_dict()
            for rec in out.get("failed", []):
                self.report(f"enumeration failed for host {rec.get('uuid')}: "
                            f"{rec.get('reason')}")
            known = {uuid for _, uuid in self.ctx.hosts}
            for rec in out.get("hosts", []):
                uuid = rec.get("uuid")
                if uuid not in known:
                    self.report(f"remote enumeration returned unknown host "
                                f"uuid {uuid} -- dropped (never guessed)")
                    continue
                bundle = rec.get("configs") or []
                if not bundle:
                    self.report(f"host {uuid}: enumeration returned no "
                                "configurations")
                    continue
                self.report(
                    f"host {uuid}: N={rec.get('n_sites')} ion sites, grid "
                    f"{rec.get('counts')}, {len(bundle)} configs "
                    f"({'Ewald-ranked' if rec.get('ewald_ranked') else 'random sampling'})")
                self.ctx.plans.append((uuid, bundle))
        if not self.ctx.plans:
            return self.exit_codes.ERROR_ENUMERATION_FAILED

    def _enumerate_in_daemon(self):
        """Legacy host-side enumeration -- fallback for unregistered codes."""
        cfg = self.ctx.cfg
        for struct_dict, uuid in self.ctx.hosts:
            try:
                plan = battery_enum.enumerate_deintercalation(
                    struct_dict, self.ctx.working_ion,
                    n_x_steps=cfg["n_x_steps"],
                    max_configs_per_x=cfg["max_configs_per_x"],
                    supercell_max_atoms=cfg["supercell_max_atoms"])
            except Exception as exc:
                self.report(f"enumeration failed for host {uuid}: {exc}")
                continue
            bundle = [s.as_dict() for k in plan["counts"]
                      for s in plan["configs"][k]]
            self.report(f"host {uuid}: N={plan['n_sites']} ion sites, grid "
                        f"{plan['counts']}, {len(bundle)} configs "
                        f"({'Ewald-ranked' if plan['ewald_ranked'] else 'random sampling'})")
            self.ctx.plans.append((uuid, bundle))

    def relax_configs(self):
        """One bundled MLIP relax per host; the working-ion elemental
        reference (if not stored yet) rides with the first host's job."""
        model = self.ctx.cfg["model"]
        ref_structs = []
        if missing_element_references([self.ctx.working_ion], model,
                              head=self.ctx.cfg["head"]):
            structs = get_mp_element_structures([self.ctx.working_ion])
            ref_structs = list(structs.values())
            if not ref_structs:
                self.report(f"Warning: no MP elemental structure for "
                            f"{self.ctx.working_ion}; voltage reference will "
                            "fall back to the DFT bundle (offset risk)")

        self.ctx.n_main = {}
        for i, (uuid, bundle) in enumerate(self.ctx.plans):
            refs = ref_structs if i == 0 else []
            self.ctx.n_main[uuid] = len(bundle)
            builder = self._construct_relax_builder(bundle + refs, uuid)
            self.to_context(**{f"battery_relax_{i}": self.submit(builder)})

    def analyze_and_store(self):
        """Collect the relaxes, run the pure calculator, store DB rows."""
        model = self.ctx.cfg["model"]
        ion = self.ctx.working_ion

        # split the relaxes first so a first-host reference is stored before
        # the ion chemical potential is read
        collected = []               # (uuid, main_entries)
        for i, (uuid, _) in enumerate(self.ctx.plans):
            wch = self.ctx[f"battery_relax_{i}"]
            if not wch.is_finished_ok:
                self.report(f"relax for host {uuid} failed")
                continue
            try:
                main, refs = split_relax_output(wch, self.ctx.n_main[uuid])
            except Exception as exc:
                self.report(f"cannot read relax output for host {uuid}: {exc}")
                continue
            if refs:
                pairs = [(e.structure.as_dict(), e.energy) for e in refs]
                add_structures("reference", model, pairs, head=self.ctx.cfg["head"])
                self.report(f"Stored MLIP elemental reference for {ion}")
            collected.append((uuid, main))
        if not collected:
            return self.exit_codes.ERROR_RELAX_FAILED

        ref_entries, missing = element_reference_entries([ion], model,
                                                 head=self.ctx.cfg["head"])
        if not ref_entries:
            self.report(f"no elemental reference for {ion} on {model}")
            return self.exit_codes.ERROR_ANALYSIS_FAILED
        if missing:
            self.report(f"Warning: DFT-fallback reference for {missing} "
                        "(per-element offset risk in the voltages)")
        ref = ref_entries[0]
        mu_ion = ref.energy / ref.composition.num_atoms

        stored = 0
        for uuid, entries in collected:
            configs = [{"structure": e.structure, "energy": e.energy}
                       for e in entries]
            try:
                summary = batt.battery_summary(configs, ion, mu_ion)
            except (ValueError, KeyError) as exc:
                # e.g. an end member did not survive the relax -- a voltage
                # without its end members is not a result (fail loudly)
                self.report(f"battery summary failed for host {uuid}: {exc}")
                continue

            endpoint_ehull = self._charged_endpoint_ehull(entries, ion, model)
            if endpoint_ehull is None:
                summary["flags"]["endpoint_ehull_failed"] = True

            add_row(DBBatteryPath, {
                "structure_uuid": uuid,
                "composition": self.ctx.chemical_formula,
                "working_ion": ion,
                "model": model,
                "avg_voltage": summary["avg_voltage"],
                "capacity_grav": summary["capacity_grav"],
                "capacity_vol": summary["capacity_vol"],
                "energy_density": summary["energy_density"],
                "volume_change_pct": summary["volume_change_pct"],
                "endpoint_ehull": endpoint_ehull,
                "voltage_profile": jsanitize(summary["voltage_profile"]),
                "configs": jsanitize(
                    [{"structure": e.structure.as_dict(), "energy": e.energy,
                      "n_ion": batt.n_ion(e.structure, ion)}
                     for e in entries]),
                "flags": jsanitize(summary["flags"]),
                "attributes": {"model_head": self.ctx.cfg["head"],
               "n_sites": summary["n_sites"],
                               "z": summary["z"], "mu_ion": mu_ion},
            })
            stored += 1
            active_flags = [k for k, v in summary["flags"].items() if v]
            self.report(
                f"host {uuid}: V_avg={summary['avg_voltage']:.3f} V, "
                f"Q={summary['capacity_grav']:.1f} mAh/g, "
                f"{summary['energy_density']:.0f} Wh/kg, "
                f"dV={summary['volume_change_pct']:+.1f}%, "
                f"flags={active_flags if active_flags else 'none'}")

        self.ctx.n_stored = stored
        if not stored:
            return self.exit_codes.ERROR_ANALYSIS_FAILED

    def _charged_endpoint_ehull(self, entries, ion, model):
        """e_above_hull (eV/atom) of the charged (empty) host against the
        chemsys hull the gen path already populated; None when it cannot be
        computed (recorded as a flag, never a silent zero)."""
        try:
            from uvsib.workchains.phase_diagram import get_entries_from_db
            charged = min((e for e in entries
                           if batt.n_ion(e.structure, ion) == 0),
                          key=lambda e: e.energy)
            db_entries = get_entries_from_db(self.ctx.chemical_formula, model) or []
            elements = [el.symbol for el in
                        Structure.from_dict(self.ctx.hosts[0][0]).composition.elements]
            # the chemsys hull was built by the gen path on the BULK head --
            # reference it on that head, and flag the comparison as
            # cross-head when the battery head differs (probe energy and
            # hull energies then carry different task heads)
            bulk_head = settings.inputs.get("bulk_relax", {}).get("head")
            el_entries, _ = element_reference_entries(elements, model,
                                                      head=bulk_head)
            if bulk_head != self.ctx.cfg["head"]:
                self.report(f"charged-endpoint ehull is CROSS-HEAD: probe on "
                            f"'{self.ctx.cfg['head']}', hull on '{bulk_head}' "
                            "-- treat the value as approximate")
            pd = PhaseDiagram(db_entries + el_entries)
            probe = ComputedStructureEntry(structure=charged.structure,
                                           energy=charged.energy)
            return float(pd.get_e_above_hull(probe))
        except Exception as exc:
            self.report(f"charged-endpoint e_above_hull failed: {exc}")
            return None

    def final_report(self):
        self.report(f"BatteryWorkChain for {self.ctx.chemical_formula} / "
                    f"{self.ctx.working_ion} finished: {self.ctx.n_stored} "
                    "host(s) stored in db_battery_path")

    ################################################################################
    def _construct_relax_builder(self, structures, uuid):
        """Bundled MLIP relax, identical plumbing to the gen/csp relaxes but
        on the battery knobs (which default to bulk_relax)."""
        cfg = self.ctx.cfg
        ml_model = cfg["model"]
        Workflow = WorkflowFactory(ml_model.lower())
        builder = Workflow.get_builder()
        builder.input_structures = List(list(structures))
        builder.code = get_code(ml_model)
        builder.local_label = Str(f"battery {self.ctx.chemical_formula} "
                                  f"{self.ctx.working_ion} host {uuid[:8]}")
        model, model_path, device = get_model_device(ml_model)
        builder.job_info = Dict({
            "job_type": "relax",
            "ML_model": ml_model,
            "model_name": model,
            "model_path": model_path,
            "model_head": cfg["head"],
            "device": device,
            "fmax": cfg["fmax"],
            "max_steps": cfg["max_steps"],
        })
        return builder
