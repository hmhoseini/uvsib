"""User-facing SQS WorkChain.

Submits one SQS request (generate + MLIP-relax, see ``codes/files/sqs.py``)
to the wrapped CalcJob, then analyses the returned structures via
``workchains/sqs_analysis.py`` -- bulk convex hull, slab surface energies and
surface O-vacancy formation energies in one pass.

Inputs
------
- ``request``    (Dict)  : SQS request payload (parent_label, structure,
                            sublattices, composition_grid, supercell,
                            n_per_comp, sqs, surfaces, defects).
- ``mu_O2``      (Float, optional) : eV per O2 molecule, MLIP-relaxed
                            molecular reference. Override only -- when not
                            supplied, the SQS CalcJob relaxes
                            ``molecular_references/o2.vasp`` against the same
                            MLIP and the per-molecule value is read from its
                            output, so absolute O-vacancy formation energies
                            are available by default. If both the override is
                            missing AND the CalcJob did not emit ``mu_O2``,
                            ``analysis["ovacancy"]`` entries fall back to
                            ``(E_vac - E_pristine) / n_vac`` and are tagged
                            ``reference="no O2 ref"``.
- ``functional`` (Str, optional)   : elemental reference set for the bulk
                            hull; defaults to ``settings.DFT_FUNC``.
- ``local_label`` (Str, optional)  : label propagated to the CalcJob node.

Outputs
-------
- ``analysis`` (Dict) : ``{"bulk": [...], "surface": [...], "ovacancy": [...]}``
"""
from aiida.engine import WorkChain
from aiida.plugins import WorkflowFactory
from aiida.orm import Str, Dict, List, Float
from uvsib.db.utils import query_by_columns, update_row, add_structures, add_row
from uvsib.db.tables import DBComposition
from uvsib.codes.utils import get_mp_element_structures
from uvsib.workflows import settings
from uvsib.workchains.utils import (get_code, get_model_device,
                                    element_reference_entries, missing_element_references)
from uvsib.workchains.sqs_analysis import split_by_kind, analyse_bulk, analyse_surface, analyse_ovac
from pymatgen.core.structure import Structure


class SQSWorkChain(WorkChain):
    """Run one SQS request and build the bulk/slab/vacancy analysis."""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("request", valid_type=Dict)
        spec.input("mu_O2", valid_type=Float, required=False)
        spec.input("local_label", valid_type=Str, required=True)

        spec.outline(
            cls.setup,
            cls.run_sqs,
            cls.inspect_sqs,
            cls.create_hull,
            cls.store_structures,
            cls.final_report,
        )

        spec.exit_code(300, "ERROR_SQS_FAILED", message="The SQS generation/relax calculation failed")
        spec.exit_code(400, "ERROR_ANALYSIS_FAILED", message="No usable structures returned from SQS")

    def setup(self):
        self.report("Running SQS WorkChain")
        self.ctx.success = False
        self.ctx.mu_O2 = (float(self.inputs.mu_O2.value) if "mu_O2" in self.inputs else None)
        self.ctx.local_label = self.inputs.local_label.value
        self.ctx.functional = settings.inputs["SQS"]["model"]

    def run_sqs(self):
        request = self.inputs.request.get_dict()
        request["element_refs"] = self._element_ref_payload(request)
        builder = self._sqs_builder(request)
        future = self.submit(builder)
        self.to_context(sqs=future)

    @staticmethod
    def _request_elements(request):
        """All elements the request can produce: parent structure + sublattice species."""
        struct = Structure.from_dict(request["structure"])
        elems = {str(el.symbol) for el in struct.composition.elements}
        for spec in (request.get("sublattices") or {}).values():
            elems.update(spec.get("species", []))
        return sorted(elems)

    def _element_ref_payload(self, request):
        """MP ground-state seeds for elements that lack an on-method reference.

        The CalcJob relaxes them (variable-cell) with the SAME calculator as
        the SQS structures; create_hull stores the results under
        source='reference' so element_reference_entries serves them as hull
        endpoints here and in the gen path alike. Elements already cached in
        the DB are skipped.
        """
        model_key = settings.inputs["bulk_relax"]["model"]
        if settings.inputs["SQS"]["model"] != model_key:
            self.report(f"WARNING: SQS model ({settings.inputs['SQS']['model']}) != "
                        f"bulk_relax model ({model_key}); references are stored "
                        "under the bulk_relax method key")
        elements = self._request_elements(request)
        self.ctx.elements = elements
        missing = missing_element_references(elements, model_key)
        if not missing:
            return []
        seeds = get_mp_element_structures(sorted(missing))
        lost = [el for el in missing if el not in seeds]
        if lost:
            self.report(f"Warning: no MP elemental structure for {lost}; "
                        "hull falls back to DFT references for them")
        if seeds:
            self.report(f"element references: {sorted(seeds)}")
        return [{"element": el, "structure": sd} for el, sd in sorted(seeds.items())]

    def inspect_sqs(self):
        wch = self.ctx.sqs
        if not wch.is_finished_ok:
            return self.exit_codes.ERROR_SQS_FAILED

        output_dict = wch.outputs.output_dict.get_dict()

        self.ctx.structures = output_dict.get("structures", [])
        self.ctx.metadata = output_dict.get("metadata", [])

        # Explicit mu_O2 input wins as an override; otherwise pick up the
        # value the SQS CalcJob computed inline against the same MLIP
        # (codes/files/sqs.py relaxes molecular_references/o2.vasp and
        # writes the per-molecule energy under "mu_O2").
        if self.ctx.mu_O2 is None and output_dict.get("mu_O2") is not None:
            self.ctx.mu_O2 = float(output_dict["mu_O2"])
            self.report(f"mu_O2 from MLIP: {self.ctx.mu_O2:.4f} eV/molecule")

        if not self.ctx.structures:
            return self.exit_codes.ERROR_ANALYSIS_FAILED
        self.report(f"SQS returned {len(self.ctx.structures)} relaxed structures")

    def create_hull(self):
        self.report("Building convex hull + surface analysis")
        model_key = settings.inputs["bulk_relax"]["model"]
        self.store_element_refs(model_key)
        bulk, slab, slab_vac = split_by_kind(self.ctx.structures, self.ctx.metadata)

        el_entries, missing = element_reference_entries(self.ctx.elements, model_key)
        if missing:
            self.report(f"Warning: DFT-fallback elemental refs for {missing} "
                        "(per-element offset risk)")
        bulk_results = analyse_bulk(bulk, self.ctx.functional, element_entries=el_entries)
        surf_results = analyse_surface(slab, bulk_results)
        vac_results = analyse_ovac(slab_vac, surf_results, self.ctx.mu_O2)

        analysis = {
            "bulk":     bulk_results,
            "surface":  surf_results,
            "ovacancy": vac_results,
            "summary":  {
                "n_bulk":     len(bulk_results),
                "n_surface":  len(surf_results),
                "n_ovacancy": len(vac_results),
                "n_ground_states": sum(1 for r in bulk_results if r["is_ground_state"]),
                "functional": self.ctx.functional,
                "mu_O2_eV":   self.ctx.mu_O2,
            },
        }

        self.ctx.bulk_results = bulk_results
        self.ctx.success = True

    def store_element_refs(self, model_key):
        """Cache the in-job MLIP-relaxed elemental references (source='reference').

        Only elements that were missing from the cache were shipped, so this
        adds each reference once; unconverged refs never made it into the
        output and simply stay missing (DFT fallback in create_hull).
        """
        pairs = []
        for s, m in zip(self.ctx.structures, self.ctx.metadata):
            if m.get("kind") != "element_ref":
                continue
            energy = Structure.from_dict(s).properties.get("predicted_energy")
            if energy is not None:
                pairs.append((s, float(energy)))
        if pairs:
            add_structures("reference", model_key, pairs)
            self.report(f"stored {len(pairs)} MLIP elemental reference(s)")

    def store_structures(self):
          """Persist SQS ground-state bulks so SurfaceBuilder.get_struct_uuid
          (query on composition + method=bulk_relax model) picks them up.

          Each grid composition gets its own DBComposition row + ml_uuid_list;
          the PARENT formula's row is written last with the union over all
          grid compositions, so a surface-builder run on the parent processes
          the whole sweep (get_struct_uuid follows the manifest)."""
          method = settings.inputs["bulk_relax"]["model"]   # MUST match get_struct_uuid's query key
          source = f"sqs_{self.ctx.functional}"             # true provenance kept here
          parent_formula = Structure.from_dict(
              self.inputs.request["structure"]).composition.reduced_formula

          # lowest-energy SQS sample per (parent, composition point)
          best = {}
          for r in self.ctx.bulk_results:
              key = (r["parent_label"], r["comp_key"])
              if key not in best or r["energy_per_atom"] < best[key]["energy_per_atom"]:
                  best[key] = r

          by_formula = {}
          for r in best.values():
              formula = Structure.from_dict(r["structure"]).composition.reduced_formula
              total_e = r["energy_per_atom"] * r["n_atoms"]   # DBStructureVersion.energy is TOTAL energy
              by_formula.setdefault(formula, []).append((r["structure"], total_e))

          per_formula = {}
          for formula, pairs in by_formula.items():
              uuids = add_structures(source, method, pairs)
              per_formula[formula] = uuids
              rows = query_by_columns(DBComposition, {"composition": formula})
              if not rows:
                  add_row(DBComposition, {"composition": formula})
                  rows = query_by_columns(DBComposition, {"composition": formula})
              update_row(DBComposition, rows[0].uuid, {"stable_struct": {
                  "origin": "sqs", "parent": parent_formula, "ml_uuid_list": uuids}})

              self.report(f"stored {len(uuids)} SQS bulk(s) for {formula}")

          # parent aggregate LAST so it wins on the parent row even when the
          # parent formula is itself a grid point
          all_uuids = [u for uuids in per_formula.values() for u in uuids]
          rows = query_by_columns(DBComposition, {"composition": parent_formula})
          if not rows:
              add_row(DBComposition, {"composition": parent_formula})
              rows = query_by_columns(DBComposition, {"composition": parent_formula})
          update_row(DBComposition, rows[0].uuid, {"stable_struct": {
              "origin": "sqs", "parent": parent_formula,
              "ml_uuid_list": all_uuids, "per_formula": per_formula}})
          self.report(f"SQS manifest on {parent_formula}: {len(all_uuids)} structure(s) "
                      f"across {len(per_formula)} composition(s)")

    def final_report(self):
        label = self.ctx.local_label
        if self.ctx.success:
            self.report(f"SQSWorkChain for {label} completed")
        else:
            self.report(f"SQSWorkChain for {label} failed")

    # ------------------------------------------------------------------
    # builder
    # ------------------------------------------------------------------
    def _sqs_builder(self, request: dict):
        ML_model = settings.inputs["SQS"]["model"]
        Workflow = WorkflowFactory("sqs_calc")
        builder = Workflow.get_builder()
        builder.input_structure = List(list=[request])
        builder.code = get_code(ML_model)
        builder.local_label = Str(self.ctx.local_label)
        model, model_path, device = get_model_device(ML_model)
        job_info = {
            "job_type":   "sqs",
            "ML_model":   ML_model,
            "model_name": model,
            "model_path": model_path,
            "model_head": settings.inputs["SQS"]["head"],
            "device":     device,
            "fmax":       settings.inputs["SQS"]["fmax"],
            "max_steps":  settings.inputs["SQS"]["max_steps"],
        }
        builder.job_info = Dict(dict=job_info)
        return builder
