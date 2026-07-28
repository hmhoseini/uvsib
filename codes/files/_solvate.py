"""Pure solvation geometry for the catalysis NEB fine-tune machinery.

No calculators, no AiiDA -- deterministic (seeded) construction of explicit
water films on adsorbed slabs plus the H-transfer endpoint building blocks:

  pack_water                water film above the top surface of a slab
  water_units               intact H2O molecules present in a structure
  find_h_transfer_pairs     (donor H, water O, acceptor) candidates near *O
  make_h_transfer_endpoints NEB endpoint pair built by editing ONE parent
  freeze_far_atoms          indices outside the reactive region (for NEB)
  nearest_index             locate e.g. the *O acceptor from ads_coord

Endpoint discipline (same as the battery NEB driver): the final state is the
SAME structure with ONLY the transferred H moved -- never two independently
built states -- so atom ordering is identical by construction and any energy
difference is chemistry, not solvent-configuration hysteresis.

Everything works on plain ase.Atoms with periodic in-plane cells (pymatgen
Slab convention: a, b in-plane, c along the surface normal).
"""
import numpy as np
from ase import Atoms
from ase.geometry import find_mic

# TIP3P-like rigid geometry used only for initial placement; the MLIP
# pre-relax / MD owns the real geometry afterwards.
WATER_OH = 0.9572          # A
WATER_HOH = 104.52         # deg
# 0.997 g/cm3 -> molecules per A^3
WATER_NUMBER_DENSITY = 0.997 * 6.02214076e23 / 18.01528 / 1.0e24

MAX_OH_BOND = 1.25         # A; H belongs to this O (water bookkeeping)


def _random_rotation(rng):
    """Uniform random rotation matrix (Shoemake quaternion method)."""
    u1, u2, u3 = rng.random(3)
    q = np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                  np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                  np.sqrt(u1) * np.sin(2 * np.pi * u3),
                  np.sqrt(u1) * np.cos(2 * np.pi * u3)])
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _water_template():
    """O at origin, H's in the xz plane with the TIP3P-like geometry."""
    half = np.radians(WATER_HOH / 2.0)
    h1 = WATER_OH * np.array([np.sin(half), 0.0, np.cos(half)])
    h2 = WATER_OH * np.array([-np.sin(half), 0.0, np.cos(half)])
    return np.array([[0.0, 0.0, 0.0], h1, h2])


def _mic_dists(points, targets, cell, pbc):
    """Minimum-image distances between every point and every target."""
    diffs = points[:, None, :] - targets[None, :, :]
    flat = diffs.reshape(-1, 3)
    _, d = find_mic(flat, cell, pbc)
    return d.reshape(len(points), len(targets))


def pack_water(slab, thickness=6.0, gap=2.3, n_waters=None, seed=0,
               min_contact=1.8, min_oo=2.6, top_clearance=3.0,
               max_tries_per_water=2000):
    """Return slab + a water film packed above its top surface.

    The film occupies z in [z_top + gap, z_top + gap + thickness]. n_waters
    defaults to liquid density for that volume. Placement is seeded rejection
    sampling (uniform in-plane fractional coordinates, uniform z, random
    orientation) with minimum-image contact checks against everything already
    placed. Raises ValueError when the film does not fit (insufficient
    vacuum) or cannot be packed (fail loudly, never a thinner film silently).
    """
    rng = np.random.default_rng(seed)
    cell = np.array(slab.get_cell())
    z_top = slab.positions[:, 2].max()
    z0, z1 = z_top + gap, z_top + gap + thickness
    cell_height = cell[2, 2]
    if z1 + top_clearance > cell_height:
        raise ValueError(
            f"water film [{z0:.1f}, {z1:.1f}] A + clearance {top_clearance} A "
            f"does not fit below the cell top ({cell_height:.1f} A); "
            "increase the slab vacuum")

    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    if n_waters is None:
        n_waters = int(round(WATER_NUMBER_DENSITY * area * thickness))
    if n_waters < 1:
        raise ValueError(f"n_waters={n_waters}: film too thin/small to hold "
                         "a single water; increase thickness or the cell")

    template = _water_template()
    solvated = slab.copy()
    solvated.set_constraint()          # constraints are the caller's business
    existing = solvated.positions.copy()
    o_positions = []
    pbc = [True, True, True]

    for w in range(n_waters):
        placed = False
        for _ in range(max_tries_per_water):
            fa, fb = rng.random(2)
            origin = fa * cell[0] + fb * cell[1]
            origin[2] = rng.uniform(z0, z1)
            mol = template @ _random_rotation(rng).T + origin
            if mol[:, 2].min() < z0 - 0.5 or mol[:, 2].max() > z1 + 0.5:
                continue
            if _mic_dists(mol, existing, cell, pbc).min() < min_contact:
                continue
            if o_positions and _mic_dists(
                    mol[:1], np.array(o_positions), cell, pbc).min() < min_oo:
                continue
            solvated += Atoms("OH2", positions=mol)
            existing = np.vstack([existing, mol])
            o_positions.append(mol[0])
            placed = True
            break
        if not placed:
            raise ValueError(
                f"packed only {w}/{n_waters} waters (film "
                f"[{z0:.1f}, {z1:.1f}] A, area {area:.1f} A^2); lower "
                "n_waters/density or grow the cell")
    return solvated


def nearest_index(atoms, point, symbol=None, max_dist=None):
    """Index of the atom nearest to a cartesian point (minimum image).

    Restricted to `symbol` if given; raises ValueError when nothing is
    within max_dist (when given) -- the caller wanted a specific atom.
    """
    idx = [i for i, s in enumerate(atoms.get_chemical_symbols())
           if symbol is None or s == symbol]
    if not idx:
        raise ValueError(f"no atoms of symbol {symbol}")
    d = _mic_dists(atoms.positions[idx], np.array([point], dtype=float),
                   np.array(atoms.get_cell()), [True, True, True])[:, 0]
    best = int(np.argmin(d))
    if max_dist is not None and d[best] > max_dist:
        raise ValueError(f"nearest {symbol or 'atom'} is {d[best]:.2f} A from "
                         f"{np.round(point, 2)} (> {max_dist} A)")
    return idx[best]


def water_units(atoms):
    """{O index: [H index, H index]} for every intact water in the structure.

    An O counts as a water oxygen iff exactly two H sit within MAX_OH_BOND.
    Lattice/adsorbate O (0 or 1 H) and hydroxide fragments are excluded --
    donors must come from intact solvent molecules.
    """
    symbols = atoms.get_chemical_symbols()
    o_idx = [i for i, s in enumerate(symbols) if s == "O"]
    h_idx = [i for i, s in enumerate(symbols) if s == "H"]
    if not o_idx or not h_idx:
        return {}
    cell = np.array(atoms.get_cell())
    d = _mic_dists(atoms.positions[o_idx], atoms.positions[h_idx],
                   cell, [True, True, True])
    units = {}
    for k, oi in enumerate(o_idx):
        bound = [h_idx[j] for j in np.nonzero(d[k] <= MAX_OH_BOND)[0]]
        if len(bound) == 2:
            units[oi] = bound
    return units


def find_h_transfer_pairs(atoms, acceptor_index, max_dist=3.5, k=3):
    """Donor candidates for *O + H -> *OH: the k nearest water H's.

    Returns [{"h", "water_o", "acceptor", "d_h_acc"}, ...] sorted by the
    H..acceptor minimum-image distance, restricted to H's belonging to
    intact waters (water_units) within max_dist.
    """
    units = water_units(atoms)
    cell = np.array(atoms.get_cell())
    acc = atoms.positions[acceptor_index][None, :]
    cands = []
    for oi, (h1, h2) in units.items():
        if oi == acceptor_index:
            continue
        for h in (h1, h2):
            d = float(_mic_dists(atoms.positions[h][None, :], acc,
                                 cell, [True, True, True])[0, 0])
            if d <= max_dist:
                cands.append({"h": h, "water_o": oi,
                              "acceptor": acceptor_index, "d_h_acc": d})
    cands.sort(key=lambda c: c["d_h_acc"])
    return cands[:k]


def make_h_transfer_endpoints(atoms, h_index, acceptor_index, bond=0.98):
    """(initial, final) NEB endpoints: ONLY the H moves onto the acceptor.

    final places the H at `bond` A from the acceptor along the minimum-image
    acceptor -> H direction (so the transfer approaches from the donor side).
    Both copies are calculator- and constraint-free; atom order is untouched.
    """
    initial = atoms.copy()
    initial.calc = None
    initial.set_constraint()
    final = initial.copy()
    cell = np.array(atoms.get_cell())
    v = atoms.positions[h_index] - atoms.positions[acceptor_index]
    v_mic, d = find_mic(v[None, :], cell, [True, True, True])
    v_mic = v_mic[0]
    if d[0] < 1e-6:
        raise ValueError("H and acceptor coincide")
    final.positions[h_index] = (atoms.positions[acceptor_index]
                                + bond * v_mic / d[0])
    return initial, final


def freeze_far_atoms(atoms, centers, free_radius=6.0):
    """Indices to freeze in the NEB: everything farther than free_radius
    (minimum image) from ALL center atoms. Centers are never frozen."""
    centers = list(centers)
    cell = np.array(atoms.get_cell())
    d = _mic_dists(atoms.positions, atoms.positions[centers],
                   cell, [True, True, True]).min(axis=1)
    fixed = [i for i in range(len(atoms))
             if d[i] > free_radius and i not in set(centers)]
    return sorted(fixed)
