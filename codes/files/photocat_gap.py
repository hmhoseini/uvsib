"""
Photocatalysis gap filter -- ensemble band-gap prediction runner (job_type
"photocat").

Predicts experimental-target band gaps for the BULK structures behind the
best-eta slabs, with an ensemble of pretrained property models, and turns
model disagreement into a failure probability against the requested gap
window. Stage 1 of the 2-stage filter -- stage 2 (HSE on the shortlist) is
done by the user, never automated.

Backends (each optional at runtime -- whatever imports, runs; everything
else is reported as unavailable):
    alignn_mbj  : ALIGNN pretrained on JARVIS TBmBJ gaps  (near-experimental)
    alignn_opt  : ALIGNN pretrained on JARVIS OPT (PBE-level) gaps
    megnet_pbe  : MEGNet MP-2018 PBE gaps
    modnet_expt : MODNet model trained on experimental gaps; needs the env
                  var MODNET_EXPT_GAP_MODEL pointing at a saved model

Fidelities: "expt" and "mbj" members form the experimental-target ensemble
(mean + spread -> failure probability). "pbe" members are consistency
checks only: a PBE gap ABOVE the experimental-target mean is physically
backwards and raises the fidelity_inversion flag.

Failure probability: P(true gap outside [gap_min, gap_max]) under
Normal(gap_mean, sigma_eff), with
    sigma_eff^2 = sigma_model^2 + (spread/2)^2 + (0.25 * n_suspicion_flags)^2
sigma_model (default 0.5 eV) is the honest model-chain error vs experiment;
suspicion flags are d10_cu_ag (mBJ systematically fails Cu(I)/Ag d10
compounds) and fidelity_inversion.

Input (input_structures.json): [{"structure": <pmg as_dict>, "tag": str}]
Output (output.json): {"results": [record per structure], "indices": [...]}
plus total.txt / failed.txt (failed = no experimental-target model produced
a gap for that structure).
"""
import argparse
import json
import math
import os
import traceback

FIDELITY = {
    "alignn_mbj": "mbj",
    "alignn_opt": "pbe",
    "megnet_pbe": "pbe",
    "modnet_expt": "expt",
}
EXPT_TARGET = ("expt", "mbj")

# mBJ (and to a degree the PBE-trained members) systematically underestimate
# gaps of d10 Cu(I)/Ag(I) compounds (Cu2O ~1 eV low) -- flag, widen sigma
D10_FLAG_ELEMENTS = {"Cu", "Ag"}
SUSPICION_SIGMA = 0.25          # eV per raised suspicion flag
METAL_GAP = 0.05                # eV; below -> predicted_metal flag


# --------------------------------------------------------------------------- #
# pure ensemble math (unit-tested in tests/test_photocat_gap.py)
# --------------------------------------------------------------------------- #
def _phi(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def ensemble_stats(gaps, fidelities):
    """Mean and spread over the experimental-target ensemble members.

    gaps: {model_name: gap_eV}; fidelities: {model_name: "expt"|"mbj"|"pbe"}.
    Returns (gap_mean, spread, primary_names); (None, None, []) when no
    experimental-target member delivered.
    """
    primary = sorted(name for name in gaps
                     if fidelities.get(name) in EXPT_TARGET
                     and gaps[name] is not None)
    if not primary:
        return None, None, []
    values = [gaps[name] for name in primary]
    gap_mean = sum(values) / len(values)
    spread = (max(values) - min(values)) if len(values) >= 2 else None
    return gap_mean, spread, primary


def failure_probability(gap_mean, spread, gap_min, gap_max, sigma_model,
                        n_suspicion):
    """P(true gap outside [gap_min, gap_max]) under the effective normal."""
    sigma_eff = math.sqrt(sigma_model ** 2
                          + ((spread or 0.0) / 2.0) ** 2
                          + (SUSPICION_SIGMA * n_suspicion) ** 2)
    p = _phi((gap_min - gap_mean) / sigma_eff)
    if gap_max is not None:
        p += 1.0 - _phi((gap_max - gap_mean) / sigma_eff)
    return min(p, 1.0), sigma_eff


def assess(gaps, fidelities, elements, gap_min, gap_max, sigma_model,
           errors=None):
    """Full per-structure record: ensemble stats, flags, failure probability.

    Returns a JSON-ready dict; gap_mean is None (p_fail 1.0, flag
    no_expt_target_model) when no experimental-target member delivered.
    """
    gap_mean, spread, primary = ensemble_stats(gaps, fidelities)

    flags = []
    if D10_FLAG_ELEMENTS & set(elements):
        flags.append("d10_cu_ag_mbj_unreliable")
    if gap_mean is not None:
        if len(primary) < 2:
            flags.append("single_expt_target_model")
        pbe_gaps = [g for name, g in gaps.items()
                    if fidelities.get(name) == "pbe" and g is not None]
        if any(g > gap_mean + 0.1 for g in pbe_gaps):
            flags.append("fidelity_inversion")   # PBE above expt-target: backwards
        if gap_mean < METAL_GAP:
            flags.append("predicted_metal")

    if gap_mean is None:
        flags.append("no_expt_target_model")
        p_fail, sigma_eff = 1.0, None
    else:
        n_susp = sum(f in ("d10_cu_ag_mbj_unreliable", "fidelity_inversion")
                     for f in flags)
        p_fail, sigma_eff = failure_probability(
            gap_mean, spread, gap_min, gap_max, sigma_model, n_susp)

    return {
        "gaps": gaps,
        "fidelities": {k: fidelities.get(k) for k in gaps},
        "errors": errors or {},
        "gap_mean": gap_mean,
        "spread": spread,
        "sigma_eff": sigma_eff,
        "p_fail": p_fail,
        "flags": flags,
        "window": [gap_min, gap_max],
        "sigma_model": sigma_model,
    }


# --------------------------------------------------------------------------- #
# model backends (import lazily; anything missing is reported, not fatal)
# --------------------------------------------------------------------------- #
def _to_jarvis(structure):
    from jarvis.core.atoms import Atoms as JarvisAtoms
    return JarvisAtoms(lattice_mat=structure.lattice.matrix.tolist(),
                       coords=structure.frac_coords.tolist(),
                       elements=[str(site.specie) for site in structure],
                       cartesian=False)


def _make_alignn(model_name):
    from alignn.pretrained import get_prediction   # availability probe

    def predict(structure):
        out = get_prediction(model_name=model_name,
                             atoms=_to_jarvis(structure))
        try:
            return float(out[0])
        except (TypeError, IndexError):
            return float(out)
    return predict


def _make_megnet():
    from megnet.utils.models import load_model
    model = load_model("Bandgap_MP_2018")

    def predict(structure):
        return float(model.predict_structure(structure).ravel()[0])
    return predict


def _make_modnet():
    path = os.environ.get("MODNET_EXPT_GAP_MODEL")
    if not path:
        raise RuntimeError("set MODNET_EXPT_GAP_MODEL to a saved MODNetModel")
    from modnet.models import MODNetModel
    from modnet.preprocessing import MODData
    model = MODNetModel.load(path)

    def predict(structure):
        data = MODData(materials=[structure])
        data.featurize()
        return float(model.predict(data).iloc[0, 0])
    return predict


BACKEND_FACTORIES = {
    "alignn_mbj": lambda: _make_alignn("jv_mbj_bandgap_alignn"),
    "alignn_opt": lambda: _make_alignn("jv_optb88vdw_bandgap_alignn"),
    "megnet_pbe": _make_megnet,
    "modnet_expt": _make_modnet,
}


def load_backends(names):
    """{name: predict_fn}, {name: reason} for everything that failed to load."""
    loaded, unavailable = {}, {}
    for name in names:
        factory = BACKEND_FACTORIES.get(name)
        if factory is None:
            unavailable[name] = "unknown backend"
            continue
        try:
            loaded[name] = factory()
        except Exception as exc:
            unavailable[name] = f"{type(exc).__name__}: {exc}"
    return loaded, unavailable


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def run(models, gap_min, gap_max, sigma_model):
    from pymatgen.core import Structure

    with open("input_structures.json", "r") as f:
        items = json.load(f)

    backends, unavailable = load_backends(models)
    for name, why in unavailable.items():
        print(f"backend {name} unavailable: {why}")
    if not backends:
        raise RuntimeError(f"no gap backend could be loaded (asked: {models})")

    results, indices = [], []
    num_failed = 0
    for i, item in enumerate(items):
        tag = item.get("tag", f"structure_{i}")
        try:
            structure = Structure.from_dict(item["structure"])
            gaps, errors = {}, dict(unavailable)
            for name, predict in backends.items():
                try:
                    gaps[name] = float(predict(structure))
                except Exception as exc:
                    gaps[name] = None
                    errors[name] = f"{type(exc).__name__}: {exc}"
            elements = {site.specie.symbol for site in structure}
            rec = assess(gaps, FIDELITY, elements, gap_min, gap_max,
                         sigma_model, errors=errors)
            rec["tag"] = tag
            rec["formula"] = structure.composition.reduced_formula
            if rec["gap_mean"] is None:
                num_failed += 1
        except Exception as exc:
            rec = {"tag": tag, "gap_mean": None, "p_fail": 1.0,
                   "flags": ["runner_error"],
                   "errors": {"runner": f"{type(exc).__name__}: {exc}"}}
            traceback.print_exc()
            num_failed += 1
        results.append(rec)
        indices.append(i)

    with open("output.json", "w") as f:
        json.dump({"results": results, "indices": indices}, f)
    with open("total.txt", "w") as f:
        f.write(str(len(items)))
    with open("failed.txt", "w") as f:
        f.write(str(num_failed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str,
                        default="alignn_mbj,alignn_opt,megnet_pbe,modnet_expt")
    parser.add_argument("--gap_min", type=float, default=1.5)
    parser.add_argument("--gap_max", type=str, default="none",
                        help="eV or 'none' (no upper window edge)")
    parser.add_argument("--sigma_model", type=float, default=0.5)
    args = parser.parse_args()

    gap_max = None if str(args.gap_max).lower() == "none" else float(args.gap_max)
    run([m.strip() for m in args.models.split(",") if m.strip()],
        args.gap_min, gap_max, args.sigma_model)
