import argparse
import json
import math
import random

import numpy as np
from ase.io import jsonio
from ase.optimize import BFGSLineSearch


KB_EV_PER_K = 8.617333262145e-5


def _mobile_indices(atoms):
    fixed = set()
    for constraint in atoms.constraints:
        indices = getattr(constraint, "index", None)
        if indices is not None:
            fixed.update(int(i) for i in indices)
    return [i for i in range(len(atoms)) if i not in fixed]


def _displace_mobile_atoms(atoms, amplitude, rng):
    mobile = _mobile_indices(atoms)
    if not mobile:
        mobile = list(range(len(atoms)))
    direction = rng.normal(size=(len(mobile), 3))
    norms = np.linalg.norm(direction, axis=1)
    norms[norms == 0.0] = 1.0
    direction = direction / norms[:, None]
    for row, atom_index in enumerate(mobile):
        atoms.positions[atom_index] += amplitude * direction[row]


def _canonical_state_key(atoms, decimals=2):
    symbols = atoms.get_chemical_symbols()
    scaled = atoms.get_scaled_positions(wrap=True)
    rows = sorted(
        (sym, round(float(pos[0]), decimals), round(float(pos[1]), decimals),
         round(float(pos[2]), decimals))
        for sym, pos in zip(symbols, scaled)
    )
    return rows


def _rate_from_barrier(barrier, temperature, prefactor):
    if barrier < 0.0:
        barrier = 0.0
    return prefactor * math.exp(-barrier / (KB_EV_PER_K * temperature))


def _run_single_dimer_event(record, calc, args, rng):
    try:
        try:
            from ase.mep.dimer import DimerControl, MinModeAtoms, MinModeTranslate
        except ImportError:
            from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate
    except Exception as exc:
        return None, f"ASE dimer API unavailable: {exc}"

    initial = jsonio.decode(record["atoms_json"])
    initial.calc = calc

    relax = BFGSLineSearch(initial, maxstep=args.maxstep, logfile="akmc_minimum.log")
    converged = relax.run(fmax=args.fmax, steps=args.relax_steps)
    if not converged:
        return None, "initial minimum relaxation did not converge"

    initial_energy = float(initial.get_potential_energy())
    saddle = initial.copy()
    saddle.calc = calc
    _displace_mobile_atoms(saddle, args.dimer_displacement, rng)

    try:
        control = DimerControl(
            initial_eigenmode_method="displacement",
            displacement_method="vector",
            logfile="akmc_dimer_control.log",
            mask=_mobile_indices(saddle),
        )
    except TypeError:
        control = DimerControl(
            initial_eigenmode_method="displacement",
            displacement_method="vector",
            logfile="akmc_dimer_control.log",
        )
    minmode_atoms = MinModeAtoms(saddle, control=control)
    minmode_atoms.displace(method="gauss", displacement_vector=None)
    try:
        translator = MinModeTranslate(
            minmode_atoms,
            trajectory=None,
            logfile="akmc_dimer_translate.log",
            maximum_translation=args.maxstep,
        )
    except TypeError:
        translator = MinModeTranslate(
            minmode_atoms,
            trajectory=None,
            logfile="akmc_dimer_translate.log",
        )
    converged = translator.run(fmax=args.dimer_fmax, steps=args.dimer_steps)
    if not converged:
        return None, "dimer search did not converge"

    saddle_atoms = minmode_atoms.atoms
    saddle_atoms.calc = calc
    saddle_energy = float(saddle_atoms.get_potential_energy())

    final_states = []
    for sign in (-1.0, 1.0):
        product = saddle_atoms.copy()
        product.calc = calc
        _displace_mobile_atoms(product, sign * args.product_displacement, rng)
        opt = BFGSLineSearch(product, maxstep=args.maxstep, logfile="akmc_product.log")
        converged = opt.run(fmax=args.fmax, steps=args.relax_steps)
        if not converged:
            continue
        product_energy = float(product.get_potential_energy())
        final_states.append({
            "energy": product_energy,
            "state_key": _canonical_state_key(product),
            "atoms_json": jsonio.encode(product),
        })

    if not final_states:
        return None, "product relaxation did not converge on either side"

    final_states.sort(key=lambda item: item["energy"])
    final_state = final_states[0]
    barrier = saddle_energy - initial_energy
    reverse_barrier = saddle_energy - final_state["energy"]
    event = {
        "source_row_id": record["row_id"],
        "source_eta": record["eta"],
        "source_surface_id": record["surface_id"],
        "source_adsorbate": record["adsorbate"],
        "source_site_type": record["site_type"],
        "initial_energy": initial_energy,
        "saddle_energy": saddle_energy,
        "final_energy": final_state["energy"],
        "barrier": barrier,
        "reverse_barrier": reverse_barrier,
        "rate": _rate_from_barrier(barrier, args.temperature, args.prefactor),
        "temperature": args.temperature,
        "prefactor": args.prefactor,
        "initial_atoms_json": jsonio.encode(initial),
        "saddle_atoms_json": jsonio.encode(saddle_atoms),
        "final_atoms_json": final_state["atoms_json"],
        "final_state_key": final_state["state_key"],
    }
    return event, None


def run_akmc(calc, args):
    with open("input_structures.json", "r") as handle:
        records = json.load(handle)

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    events = []
    rejected = []

    for record in records:
        local_events = []
        for attempt in range(args.searches_per_minimum):
            try:
                event, reason = _run_single_dimer_event(record, calc, args, rng)
            except Exception as exc:
                event, reason = None, str(exc)
            if event is None:
                rejected.append({
                    "row_id": record.get("row_id"),
                    "adsorbate": record.get("adsorbate"),
                    "attempt": attempt,
                    "reason": reason,
                })
                continue
            event["attempt"] = attempt
            local_events.append(event)
            events.append(event)

        if local_events:
            total_rate = sum(max(0.0, item["rate"]) for item in local_events)
            if total_rate > 0.0:
                threshold = random.random() * total_rate
                cumulative = 0.0
                selected = local_events[-1]
                for item in local_events:
                    cumulative += item["rate"]
                    if cumulative >= threshold:
                        selected = item
                        break
                selected["kmc_selected"] = True
                selected["kmc_time_increment"] = -math.log(max(random.random(), 1e-16)) / total_rate

    output = {
        "events": events,
        "num_minima": len(records),
        "num_events": len(events),
        "num_rejected": len(rejected),
    }

    with open("output.json", "w") as handle:
        json.dump(output, handle)
    with open("total.txt", "w") as handle:
        handle.write(str(len(records)))
    with open("failed.txt", "w") as handle:
        handle.write(str(len(rejected)))
    with open("rejected.json", "w") as handle:
        json.dump(rejected, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ML_model", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--dimer_fmax", type=float, default=0.08)
    parser.add_argument("--relax_steps", type=int, default=200)
    parser.add_argument("--dimer_steps", type=int, default=120)
    parser.add_argument("--searches_per_minimum", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--prefactor", type=float, default=1.0e13)
    parser.add_argument("--dimer_displacement", type=float, default=0.05)
    parser.add_argument("--product_displacement", type=float, default=0.15)
    parser.add_argument("--maxstep", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parsed = parser.parse_args()

    from _calculators import make_calculator

    calculator = make_calculator(
        parsed.ML_model,
        model=parsed.model,
        model_path=parsed.model_path,
        device=parsed.device,
        task_name=parsed.task_name,
    )
    run_akmc(calculator, parsed)
