"""Generate disordered Special Quasirandom Structures at a target composition.

Entry-point script run as ``aiida.py`` by a CalcJob on the gnome@v100 code (the
generic v100 Python code; see config ``codes.GNoME``). This is the *remote*
counterpart of ``workchains.synthesizability.make_disordered_sqs`` -- the icet
``generate_sqs`` Monte-Carlo search is expensive (minutes at 32 atoms, much
worse beyond), so the SynthesizabilityWorkChain offloads it here instead of
running it inline in a daemon step.

It produces the *disordered* partner(s) for the vibrational entropy of ordering
(Fultz): one SQS per realization (different ``random_seed``), at the same
composition as the ordered ground state. Each SQS is later phononed alongside
the ordered polymorph by the PhononWorkChain.

Request (``sqs_request.json``, staged in via the CalcJob ``file`` namespace)
---------------------------------------------------------------------------

    {
      "formula":        "Cu3Au",
      "sqs_size":       64,            # atoms in the SQS supercell (icet max_size)
      "sqs_steps":      2000,          # icet MC steps per realization
      "n_realizations": 3,             # independent SQS (seeds 0..N-1)
      "lattice":        "fcc",         # ase.build.bulk lattice for the primitive
      "a":              3.8,           # nominal lattice constant (irrelevant to
                                       #   the search; the cell is relaxed later)
      "cutoffs":        [7.0, 4.0]     # icet ClusterSpace pair, triplet cutoffs (A)
    }

Output (``output.json``, parsed by the generic ``sqs_parser`` into output_dict)
-------------------------------------------------------------------------------

    {
      "results": [
        {"uuid": "sqs:Cu3Au#0", "seed": 0, "structure": <pymatgen Structure.as_dict()>},
        ...
      ],
      "formula":         "Cu3Au",
      "reduced_formula": "Cu3Au",
      "n_requested":     3,
      "n_generated":     3
    }

The ``uuid`` keys ("sqs:<reduced_formula>#<i>") match the ids the workchain
expects so the phonon results map straight back onto the SQS realizations.

Needs ``icet`` (plus ``ase`` and ``pymatgen``) in the v100 Python environment.
"""

import sys
import json
import argparse

from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import bulk
from icet import ClusterSpace
from icet.tools.structure_generation import generate_sqs


def make_disordered_sqs(formula, supercell_atoms, lattice, a, cutoffs, n_steps, random_seed):
    """One disordered SQS at the target composition; pymatgen dict or None."""
    comp = Composition(formula)
    els = [el.symbol for el in comp.elements]
    if len(els) < 2:
        return None
    fracs = comp.fractional_composition.get_el_amt_dict()
    prim = bulk(els[0], lattice, a=float(a))
    cs = ClusterSpace(prim, list(cutoffs), [els])
    sqs = generate_sqs(
        cluster_space=cs,
        max_size=int(supercell_atoms),
        target_concentrations={el: float(fracs[el]) for el in els},
        n_steps=int(n_steps), T_start=5.0, T_stop=0.01,
        random_seed=random_seed)
    return AseAtomsAdaptor().get_structure(sqs).as_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", type=str, default="sqs_request.json")
    # The shared cmdline helper may append --ML_model/--device etc.; tolerate
    # and ignore any extras (SQS generation has no calculator).
    args, _unknown = ap.parse_known_args()

    with open(args.request, "r", encoding="utf-8") as f:
        req = json.load(f)

    formula = req["formula"]
    reduced = Composition(formula).reduced_formula
    n_real = max(1, int(req.get("n_realizations", 3)))
    supercell_atoms = int(req.get("sqs_size", 32))
    n_steps = int(req.get("sqs_steps", 2000))
    lattice = req.get("lattice", "fcc")
    a = float(req.get("a", 3.8))
    cutoffs = req.get("cutoffs", [7.0, 4.0])

    results = []
    for i in range(n_real):
        try:
            sqs_dict = make_disordered_sqs(
                formula, supercell_atoms, lattice, a, cutoffs, n_steps,
                random_seed=i)
        except Exception as exc:  # noqa: BLE001 -- log + continue per realization
            sys.stderr.write(f"SQS realization {i} failed: {exc}\n")
            sqs_dict = None
        if sqs_dict is None:
            continue
        results.append({"uuid": f"sqs:{reduced}#{i}", "seed": i,
                        "structure": sqs_dict})

    output = {
        "results": results,
        "formula": formula,
        "reduced_formula": reduced,
        "n_requested": n_real,
        "n_generated": len(results),
    }
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
