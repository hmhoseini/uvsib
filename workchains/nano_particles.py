"""NanoParticleWorkChain -- adsorbates on nano-particles.

The particle counterpart of AdsorbatesWorkChain (docs/nano_particle_path.md):
slabs go through pymatgen's AdsorbateSiteFinder, which is z-height based and
cannot handle free particles, so particles get their own chain end to end:

  DBNanoParticles rows (imported via run_dir/import_nano_particles.py,
  status='Created')
      -> NanoParticle CalcJobs in batches of BATCH_SIZE particles
         (codes/files/nano_particles.py: per particle a bare relax with the SAME
         MLIP as the adsorbate relaxations, local-normal site finding,
         X-dummy adsorbate placement, per-site relax + validation -- the
         adsorbate registry/pathways/validators are the slab runner's own,
         staged alongside as slab_adsorbates.py)
      -> CHE overpotentials with the SAME calculate_*_overpotential
         functions as the slab chain
      -> results into DBNanoParticles.attributes["adsorbates"]["<reaction>/
         <path>"] (energetics per site, structures of the best set) and the
         relaxed bare particle back onto the row (structure, energy, model).

Nothing is materialized into DBStructure/DBSurface -- this chain is fully
separate from the slab pipeline by design.

NOTE for main.py wiring: ``_construct_particle_builder`` (currently dead
behind a NotImplementedError) must additionally pass ``reaction`` and
``reaction_path`` when it is enabled -- both are available on main's ctx.
"""
from ase.io import jsonio
from pymatgen.io.ase import AseAtomsAdaptor

from aiida.engine import WorkChain
from aiida.orm import Str

from uvsib.codes.nano_particles.calculation import NanoParticles
from uvsib.codes.nano_particles.workchain import (batch_payload_file,
                                                  nano_cmdline,
                                                  nano_job_options,
                                                  nano_parameters)
from uvsib.db.tables import DBNanoParticles
from uvsib.db.utils import query_by_columns, update_row
from uvsib.workchains.cer import calculate_cer_overpotential
from uvsib.workchains.co2rr import calculate_co2rr_overpotential
from uvsib.workchains.her import calculate_her_overpotential
from uvsib.workchains.noxrr import calculate_noxrr_overpotential
from uvsib.workchains.nrr import calculate_nrr_overpotential
from uvsib.workchains.oer import calculate_oer_overpotential
from uvsib.workchains.orr import calculate_orr_overpotential
from uvsib.workchains.utils import get_code, get_model_device
from uvsib.workflows import settings

_REACTION_MAP = {
    "OER": calculate_oer_overpotential,
    "CO2RR": calculate_co2rr_overpotential,
    "CER": calculate_cer_overpotential,
    "NRR": calculate_nrr_overpotential,
    "NOXRR": calculate_noxrr_overpotential,
    "HER": calculate_her_overpotential,
    "ORR": calculate_orr_overpotential,
}

# particles per NanoParticles CalcJob: the model and gas references
# are loaded/relaxed once per job, and one bad particle only fails its own
# result entry -- but the job walltime must cover the whole batch
BATCH_SIZE = 50


def _chunk(seq, size):
    return [seq[i:i + size] for i in range(0, len(seq), size)]


# The context is YAML-serialized into the process node's ``checkpoints``
# attribute -- a SINGLE jsonb string, capped by postgres at 268435455 bytes --
# and rewritten on every step transition. AiiDA nodes serialize by reference
# (!aiida_node <uuid>), plain python objects by value. So the ctx must only
# ever carry uuids/counters/node references: structures and job payloads are
# fetched fresh in the step that needs them and dropped again.


class NanoParticleWorkChain(WorkChain):
    """Adsorbates + CHE energetics on the DBNanoParticles inventory."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("elements", valid_type=Str)
        spec.input("ml_model", valid_type=Str)
        spec.input("reaction", valid_type=Str)
        spec.input("reaction_path", valid_type=Str)
        # accepted for main.py compatibility; particles_range filters the
        # inventory by atom count ("40-160"), generator is ignored
        spec.input("particles_range", valid_type=Str, required=False)
        # spec.input("generator", valid_type=Str, required=False)

        spec.outline(
            cls.setup,
            cls.run_adsorbs,
            cls.inspect_adsorbs,
            cls.store_results,
            cls.final_report,
        )

        spec.exit_code(300, "ERROR_CALCULATION_FAILED", message="The calculation did not finish successfully")
        spec.exit_code(301, "ERROR_NO_STRUCTURES_FOUND", message="No particles matched in DBNanoParticles.")
        spec.exit_code(302, "NO_CANDIDATES_WITHIN_ETA_LIMIT", message="No candidates produced an overpotential")

    # ------------------------------------------------------------------ #
    def setup(self):
        self.ctx.ml_model = self.inputs.ml_model.value
        self.ctx.reaction = self.inputs.reaction.value
        self.ctx.reaction_path = self.inputs.reaction_path.value
        self.ctx.result_key = f"{self.ctx.reaction}/{self.ctx.reaction_path}"
        # normalized sorted form, matching import_nano_particles / add_nano_particles
        self.ctx.elements = "-".join(sorted(self.inputs.elements.value.split("-")))

        if self.ctx.reaction not in _REACTION_MAP:
            self.report(f"The reaction {self.ctx.reaction} is not known")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        n_min, n_max = 0, 10**3
        if "particles_range" in self.inputs:
            rng = self.inputs.particles_range.value
            if rng and len(rng.split("-")) == 2:
                n_min, n_max = (int(x) for x in rng.split("-"))

        selected, skipped_done = [], 0
        for row in self._inventory():
            if row.status == "Failed":
                continue
            if not (n_min <= (row.num_atoms or 0) <= n_max):
                continue
            attrs = row.attributes or {}
            if self.ctx.result_key in (attrs.get("adsorbates") or {}):
                skipped_done += 1          # idempotent re-run: already done
                continue
            selected.append(str(row.uuid))
        if skipped_done:
            self.report(f"{skipped_done} particle(s) already carry "
                        f"{self.ctx.result_key} results; skipping them")
        if not selected:
            self.report(f"No particles to process for elements="
                        f"{self.ctx.elements} (range {n_min}-{n_max})")
            return self.exit_codes.ERROR_NO_STRUCTURES_FOUND

        # uuids only -- the structures are re-read in run_adsorbs
        self.ctx.particle_uuids = selected
        self.ctx.candidates = 0
        self.ctx.stored = 0
        self.ctx.failed = 0
        self.report(f"NanoParticle adsorbates: {len(selected)} particle(s), "
                    f"{self.ctx.reaction}/{self.ctx.reaction_path} with "
                    f"{self.ctx.ml_model}")

    def _inventory(self):
        """The DBNanoParticles rows for this chain's elements, always fetched
        fresh -- never cached in the ctx (see the note above _chunk)."""
        return query_by_columns(DBNanoParticles, {"elements": self.ctx.elements})

    # ------------------------------------------------------------------ #
    def run_adsorbs(self):
        """NanoParticles jobs in batches of BATCH_SIZE particles
        (model + gas references loaded/relaxed once per job; bare relax +
        sites + adsorbate relaxations happen remotely)."""
        ml_model = self.ctx.ml_model
        job_cfg = settings.inputs["adsorbates"]
        model_name, model_path, device = get_model_device(ml_model)
        cmdline = nano_cmdline(ml_model, model_name, model_path,
                               job_cfg["head"], device, job_cfg,
                               self.ctx.reaction, self.ctx.reaction_path)

        # structures live only for the duration of this step; the payload goes
        # to the job as a SinglefileData, not through the ctx
        structures = {str(row.uuid): row.structure for row in self._inventory()}

        self.ctx.batch_uuids = _chunk(self.ctx.particle_uuids, BATCH_SIZE)
        for b_idx, batch in enumerate(self.ctx.batch_uuids):
            payload = batch_payload_file([(u, structures[u]) for u in batch])
            inputs = {
                "code": get_code(ml_model),
                "parameters": nano_parameters("nano_adsorbates", cmdline),
                "file": {"input_structures_file": payload},
                "metadata": {
                    "options": nano_job_options(ml_model),
                    "label": f"NanoAdsorbates: {self.ctx.reaction}/"
                             f"{self.ctx.reaction_path} batch "
                             f"{b_idx + 1}/{len(self.ctx.batch_uuids)} "
                             f"({len(batch)} particles)",
                },
            }
            for uuid_str in batch:
                update_row(DBNanoParticles, uuid_str, {"status": "Running"})
            self.to_context(**{f"np_batch_{b_idx}":
                               self.submit(NanoParticles, **inputs)})
        self.report(f"submitted {len(self.ctx.batch_uuids)} batch job(s) for "
                    f"{len(self.ctx.particle_uuids)} particle(s) "
                    f"(batch size {BATCH_SIZE})")

    # ------------------------------------------------------------------ #
    def inspect_adsorbs(self):
        """Job-level triage only. The batch outputs are deliberately NOT
        carried in the ctx -- one batch of relaxed adsorbate geometries is
        O(100 MB), and the ctx is re-serialized into a single jsonb string on
        every step transition. store_results re-reads them per batch instead."""
        self.ctx.ok_batches = []
        for b_idx, batch in enumerate(self.ctx.batch_uuids):
            job = self.ctx[f"np_batch_{b_idx}"]
            if not job.is_finished_ok:
                self.report(f"WARNING: batch {b_idx + 1} failed entirely "
                            f"(exit status {job.exit_status}); marking its "
                            f"{len(batch)} particle(s) Failed")
                for uuid_str in batch:
                    self.ctx.failed += 1
                    update_row(DBNanoParticles, uuid_str, {"status": "Failed"})
                continue
            self.ctx.ok_batches.append(b_idx)

        if not self.ctx.ok_batches:
            self.report("ERROR: all batches failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

    # ------------------------------------------------------------------ #
    def store_results(self):
        """CHE overpotentials per particle/site; results onto the row.

        One batch payload is held at a time and released again -- nothing from
        the outputs enters the ctx."""
        calc_method = _REACTION_MAP[self.ctx.reaction]
        model_key = f"{self.ctx.ml_model.lower()}_energy"

        for b_idx in self.ctx.ok_batches:
            job = self.ctx[f"np_batch_{b_idx}"]
            by_uuid = {r["uuid"]: r for r in
                       job.outputs.output_dict.get_dict().get("results", [])}
            for uuid_str in self.ctx.batch_uuids[b_idx]:
                result = by_uuid.get(uuid_str)
                if result is None or "error" in result:
                    reason = (result["error"] if result
                              else "missing from batch output")
                    self.report(f"particle {uuid_str[:8]} failed in batch "
                                f"{b_idx + 1}: {reason}")
                    self.ctx.failed += 1
                    update_row(DBNanoParticles, uuid_str, {"status": "Failed"})
                    continue
                self._store_particle(uuid_str, result, calc_method, model_key)
                self.ctx.stored += 1
            del by_uuid          # release the batch before loading the next

        if self.ctx.failed:
            self.report(f"WARNING: {self.ctx.failed}/"
                        f"{len(self.ctx.particle_uuids)} particle(s) failed")
        if not self.ctx.stored:
            self.report("ERROR: all particles failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def _store_particle(self, uuid_str, output, calc_method, model_key):
        """Energetics of one particle onto its DBNanoParticles row."""
        particle = output["particle"]
        sets_out = []
        best = None
        for adsorb_set in output["structures"]:
            energy_set = {}
            for ads_json in adsorb_set["structures"]:
                atoms = jsonio.decode(ads_json)
                energy_set[atoms.info["adsorbate"]] = atoms.info[model_key]
            try:
                eta, dG_steps, dG_cumulative = calc_method(energy_set, self.ctx.reaction_path)
            except KeyError as missing:
                # an intermediate was dropped by the validator (e.g.
                # dissociated during relaxation) -- skip this site only
                self.report(
                    f"Skipping {self.ctx.result_key} on particle "
                    f"{uuid_str[:8]} ({adsorb_set['site_type']}): missing "
                    f"intermediate {missing.args[0]} "
                    f"(have: {sorted(energy_set)})")
                continue
            record = {
                "site_type": adsorb_set["site_type"],
                "ads_coord": adsorb_set["ads_coord"],
                "site_anchors": adsorb_set.get("site_anchors"),
                "eta": eta,
                "dG_steps": dG_steps,
                "dG_cumulative": dG_cumulative,
                "energies": energy_set,
            }
            sets_out.append(record)
            self.ctx.candidates += 1
            if best is None or eta < best[0]:
                best = (eta, adsorb_set)

        row = query_by_columns(DBNanoParticles, {"uuid": uuid_str})[0]
        attrs = dict(row.attributes or {})
        adsorbates_attr = dict(attrs.get("adsorbates") or {})
        adsorbates_attr[self.ctx.result_key] = {
            "model": self.ctx.ml_model,
            "particle_energy": particle["energy"],
            "n_sets": len(sets_out),
            "sets": sets_out,
            # geometries only for the best (lowest-eta) site: bounded size
            "best_structures": (best[1]["structures"] if best else None),
        }
        attrs["adsorbates"] = adsorbates_attr
        relaxed = AseAtomsAdaptor.get_structure(
            jsonio.decode(particle["structure"]))
        update_row(DBNanoParticles, uuid_str, {
            "attributes": attrs,
            "structure": relaxed.as_dict(),
            "energy": particle["energy"],
            "model": self.ctx.ml_model,
            "status": "Done",
        })

    # ------------------------------------------------------------------ #
    def final_report(self):
        if self.ctx.candidates == 0:
            self.report(f"NanoParticleWorkChain {self.ctx.elements}: no "
                        f"candidates for {self.ctx.result_key}.")
            return self.exit_codes.NO_CANDIDATES_WITHIN_ETA_LIMIT
        self.report(f"NanoParticleWorkChain {self.ctx.elements}: "
                    f"{self.ctx.candidates} site energetics stored for "
                    f"{self.ctx.result_key}.")
