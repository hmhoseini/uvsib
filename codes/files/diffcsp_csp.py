"""DiffCSP CSP runner (compute-node side).

Shipped to the remote folder as ``aiida.py`` by ``DiffCSPCalculation`` together
with ``refine.py`` (charge-neutral + primitive-cell dedup, shared with the
MatterGen/GNoME paths).

Unlike DiffCSP's ab-initio ``scripts/generation.py`` (unconditioned, samples
compositions from a fixed training-set atom-count distribution),
``scripts/sample.py`` (Jiao *et al.*, https://github.com/jiaor17/DiffCSP) takes
an explicit ``--formula`` and fixes the atom types to that composition,
diffusing only the lattice and fractional coordinates -- true
composition-conditioned crystal structure prediction, the same role
MatterGen's CSP mode and GNoME's SAPS-csp mode play in this pipeline. No
chemical-system filtering approximation is needed here.

Pipeline:
  1. shell out to ``<repo_path>/scripts/sample.py --model_path ... --formula
     ... --num_evals ... --batch_size ... --step_lr ... --save_path <this
     job's own working directory>``, with ``cwd=repo_path`` (its ``eval_utils``
     helper resolves the ``diffcsp`` package relative to cwd, so it must run
     from the repo root -- this is why ``--save_path`` is passed explicitly
     rather than relying on the default relative path);
  2. sample.py writes ``<save_path>/<formula>/<formula>_<i>.cif`` for each of
     the ``num_evals`` attempts (some may be silently skipped by sample.py
     itself if it failed to build a valid structure);
  3. read every CIF back into a pymatgen ``Structure``;
  4. charge-neutrality filter + primitive-cell de-duplication (``refine.py``).

No MLIP energy screen runs here, matching the DiffCSP gen-path design: energy
scoring happens once, downstream, in the CSP path's shared ML relax
(+ MinimaHopping) stage that already handles MatterGen's and GNoME's CSP
candidates.

The output contract is identical to MatterGen's/GNoME's: ``output.json`` is a
list of pymatgen structure dicts, consumed unchanged by ``DiffCSPParser``.
"""

import os
import json
import glob
import argparse
import subprocess
from pymatgen.core import Structure
from refine import select_charge_neutral, refine_and_filter_structures


def run_diffcsp(args, save_path):
    """Invoke DiffCSP's own composition-conditioned sampler."""
    cmd = [
        "python", os.path.join(args.repo_path, "scripts", "sample.py"),
        "--model_path", args.model_path,
        "--save_path", save_path,
        "--formula", args.formula,
        "--num_evals", str(args.num_evals),
        "--batch_size", str(args.batch_size),
        "--step_lr", str(args.step_lr),
    ]
    subprocess.run(cmd, cwd=args.repo_path, check=True)

def structures_from_cifs(save_path, formula):
    cif_dir = os.path.join(save_path, formula)
    structures = []
    for cif_path in sorted(glob.glob(os.path.join(cif_dir, "*.cif"))):
        try:
            structures.append(Structure.from_file(cif_path))
        except Exception:
            continue
    return structures

def main():
    parser = argparse.ArgumentParser(description="DiffCSP composition-conditioned CSP")
    parser.add_argument("--repo_path", required=True, help="path to the DiffCSP checkout")
    parser.add_argument("--model_path", required=True, help="DiffCSP CSP training run / checkpoint dir")
    parser.add_argument("--chemical_formula", required=True, help="target composition, e.g. 'Y2Ru2O7'")
    parser.add_argument("--num_evals", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--step_lr", type=float, default=1e-5)
    args = parser.parse_args()
    args.formula = args.chemical_formula

    save_path = os.getcwd()
    run_diffcsp(args, save_path)
    structures = structures_from_cifs(save_path, args.formula)

    struct_dicts = [s.as_dict() for s in structures]
    struct_dicts = select_charge_neutral(struct_dicts)
    struct_dicts = refine_and_filter_structures(struct_dicts)

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(struct_dicts, f)

if __name__ == "__main__":
    main()
