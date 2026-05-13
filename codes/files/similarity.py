import json
import argparse
import numpy as np
from ase import Atoms
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list
from upet.calculator import UPETCalculator
from dscribe.descriptors import SOAP
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from scipy.special import sph_harm_y as _sph_harm_fn
    def _ylm(m, l, phi, theta):
        return _sph_harm_fn(l, m, theta, phi)   # new scipy API: (n, m, theta, phi)
except ImportError:
    from scipy.special import sph_harm as _sph_harm_legacy
    def _ylm(m, l, phi, theta):
        return _sph_harm_legacy(m, l, phi, theta)  # old scipy API: (m, l, phi, theta)


# ---------------------------------------------------------------------------
# Module-level SOAP calculator (constructed once, reused across calls)
# ---------------------------------------------------------------------------
_SOAP_R_CUT = 6.0
_SOAP_SPECIES = ["H"]   # dummy species — all atoms treated identically

_soap_calc = SOAP(species=_SOAP_SPECIES, r_cut=_SOAP_R_CUT, n_max=8, l_max=6, periodic=True,
    average="outer",    # mean over atoms → one vector per structure
    sparse=False)

# Steinhardt l-values to include
_STEINHARDT_L = [4, 6, 8, 10]
_STEINHARDT_R_CUT = 6.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _to_dummy(atoms: Atoms) -> Atoms:
    """Return a copy with all species replaced by H (geometry preserved)."""
    dummy = atoms.copy()
    dummy.set_chemical_symbols(["H"] * len(atoms))
    return dummy


def _soap_fp(atoms: Atoms) -> np.ndarray:
    """Unit-normalised SOAP fingerprint (averaged over atoms)."""
    vec = _soap_calc.create(_to_dummy(atoms))   # shape (n_soap_features,)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _steinhardt_fp(atoms: Atoms) -> np.ndarray:
    """
    Unit-normalised vector of averaged Steinhardt Q_l values for l in _STEINHARDT_L.

    Uses ASE's neighbor_list for PBC-correct bond vectors (works for any
    cell shape — orthorhombic, triclinic, hexagonal), then computes the
    standard Steinhardt invariant:

        Q_l(i) = sqrt( 4π/(2l+1) * sum_m |<Y_l^m>_i|^2 )

    averaged over all atoms.
    """
    i_idx, _, D = neighbor_list("ijD", atoms, _STEINHARDT_R_CUT)
    n_atoms = len(atoms)

    # Normalise bond vectors → spherical angles
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    D_hat = D / np.where(norms > 0, norms, 1.0)
    theta = np.arccos(np.clip(D_hat[:, 2], -1.0, 1.0))   # polar   [0, π]
    phi   = np.arctan2(D_hat[:, 1], D_hat[:, 0])          # azimuth [-π, π]

    # Count neighbours per atom (for normalisation)
    counts = np.zeros(n_atoms)
    np.add.at(counts, i_idx, 1)
    safe_counts = np.where(counts > 0, counts, 1)

    ql_means = []
    for l in _STEINHARDT_L:
        # Accumulate sum_j Y_l^m(r_ij) per atom for each m
        qlm = np.zeros((n_atoms, 2 * l + 1), dtype=complex)
        for idx_m, m in enumerate(range(-l, l + 1)):
            ylm = _ylm(m, l, phi, theta)             # shape (n_bonds,)
            np.add.at(qlm[:, idx_m], i_idx, ylm)

        qlm /= safe_counts[:, np.newaxis]           # per-atom average over bonds
        ql_per_atom = np.sqrt(
            4 * np.pi / (2 * l + 1) * np.sum(np.abs(qlm) ** 2, axis=1)
        )
        ql_means.append(float(np.mean(ql_per_atom)))

    vec  = np.array(ql_means)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, clipped to [0, 1] (SOAP/Steinhardt values are non-negative)."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.clip(a @ b / denom, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fingerprint(atoms: Atoms, w_soap: float = 0.7, w_stein: float = 0.3) -> np.ndarray:
    """
    Compute a combined species-agnostic fingerprint for a periodic structure.

    Parameters
    ----------
    atoms   : ASE Atoms object (must have a periodic cell)
    w_soap  : weight for the SOAP component (default 0.7)
    w_stein : weight for the Steinhardt component (default 0.3)

    Returns
    -------
    1-D numpy array — concatenation of the two weighted, unit-normed components.
    """
    soap_vec  = _soap_fp(atoms)
    stein_vec = _steinhardt_fp(atoms)
    return np.concatenate([w_soap * soap_vec, w_stein * stein_vec])


def similarity(atoms_a: Atoms, atoms_b: Atoms, w_soap: float = 0.7, w_stein: float = 0.3) -> float:
    """
    Species-agnostic structural similarity between two periodic crystals.

    Score = w_soap  * cosine(SOAP_a,  SOAP_b)
          + w_stein * cosine(Stein_a, Stein_b)

    Components are computed independently and summed so each contribution
    is always interpretable on the same [0, 1] scale regardless of the
    difference in descriptor dimensionality.

    Parameters
    ----------
    atoms_a, atoms_b : ASE Atoms objects with periodic cells
    w_soap           : weight for SOAP similarity contribution  (default 0.7)
    w_stein          : weight for Steinhardt similarity contribution (default 0.3)

    Returns
    -------
    float in [0, 1].  1.0 = geometrically identical, 0.0 = no similarity.
    """
    if abs(w_soap + w_stein - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {w_soap + w_stein:.4f}")

    soap_a,  soap_b  = _soap_fp(atoms_a),      _soap_fp(atoms_b)
    stein_a, stein_b = _steinhardt_fp(atoms_a), _steinhardt_fp(atoms_b)

    sim_soap  = _cosine(soap_a,  soap_b)
    sim_stein = _cosine(stein_a, stein_b)

    return w_soap * sim_soap + w_stein * sim_stein


def run(calc, fmax, max_steps):
    fail_relax = 0
    relaxed_generated = list()
    with open('input_structures.json', 'r') as f:
        structure_list = json.loads(f.read())
        for structure_dict in structure_list:
            atoms = AseAtomsAdaptor.get_atoms(Structure.from_dict(structure_dict))
            atoms.calc = calc
            relax = FIRE(atoms)
            relax.run(fmax=fmax, steps=max_steps)
            if not relax.converged:
                fail_relax += 1
                continue
            relaxed_generated.append(atoms)

    relaxed_comparisons = list()
    with open('comparison_structures.json', 'r') as f:
        structure_list = json.loads(f.read())
        for structure_dict in structure_list:
            atoms = AseAtomsAdaptor.get_atoms(Structure.from_dict(structure_dict))
            atoms.calc = calc
            relax = FIRE(atoms)
            relax.run(fmax=fmax, steps=max_steps)
            if not relax.converged:
                fail_relax += 1
                continue
            relaxed_comparisons.append(atoms)

    count = 0
    similarities = dict()
    for rel_comp in relaxed_comparisons:
        for rel_generated in relaxed_generated:
            sim_val = similarity(rel_comp, rel_generated, w_soap=0.5, w_stein=0.5)
            comp_dict = AseAtomsAdaptor.get_structure(rel_comp).as_dict()
            generated_dict = AseAtomsAdaptor.get_structure(rel_generated).as_dict()
            similarities[count] = {'similarity': sim_val, 'comparison': comp_dict, 'generated': generated_dict}
            count += 1

    output = dict({'similarities': similarities})
    with open('output.json', 'w') as f:
        json.dump(output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml_model", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--fmax", type=float)
    parser.add_argument("--max_steps", type=int)
    args = parser.parse_args()

    calc = UPETCalculator(model=args.model, device=args.device)

    run(calc, args.fmax, args.max_steps)
