"""
Battery NEB driver: hop enumeration, endpoint construction, percolation.

Pure python (pymatgen + numpy, no AiiDA) -- the domain layer above the shared
NEB engine (codes/files/neb.py). The BatteryNEBWorkChain uses it to turn a
db_battery_path row into a bundle of consistently-ordered endpoint pairs, and
to turn the computed barriers into THE battery transport number: the lowest
barrier at which the hop network percolates (1D / 2D / 3D).

Two migration limits, same machinery:
  vacancy : discharged supercell minus one ion; a neighboring ion hops into
            the vacancy (dominant transport picture near full lithiation).
  dilute  : one ion in the empty host hopping between the (mapped) ion sites
            (transport near full charge; often rate-limiting).

Endpoints are always built by EDITING one parent structure (remove a site /
move one atom), so initial and final share atom ordering by construction --
the engine's hard requirement.

Symmetry handling: hops are deduplicated by the invariant (site class of A,
site class of B, hop length). Site classes come from the symmetrized
structure. Distinct hops with identical invariants would be merged -- rare in
practice and conservative for screening (upgrade path: full orbit analysis).
"""
from __future__ import annotations

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def _as_structure(struct):
    if isinstance(struct, Structure):
        return struct
    return Structure.from_dict(struct)


def ion_site_classes(structure, working_ion, symprec=0.1):
    """{site_index: class_id} for the working-ion sites (symmetry orbits).

    Falls back to one-class-per-site when symmetry detection fails (P1 cells
    after relaxation noise): dedup then keeps every hop, which is safe."""
    ion_idx = [i for i, s in enumerate(structure)
               if s.specie.symbol == working_ion]
    try:
        sym = SpacegroupAnalyzer(structure, symprec=symprec) \
            .get_symmetrized_structure()
        classes = {}
        for class_id, group in enumerate(sym.equivalent_indices):
            for idx in group:
                if idx in set(ion_idx):
                    classes[idx] = class_id
        if set(classes) == set(ion_idx):
            return classes
    except Exception:
        pass
    return {idx: idx for idx in ion_idx}


def enumerate_hops(structure, working_ion, max_hop=4.5, symprec=0.1,
                   round_dist=2):
    """Ion-ion hop graph of a (discharged) supercell.

    Returns (distinct, edges):
      distinct : {class_key: hop} -- one representative hop per symmetry
                 class, the ones that need a NEB. hop = {"a": site index,
                 "b": site index, "jimage": (3 ints, periodic image of b),
                 "distance": float, "class_key": str}
      edges    : [hop, ...] -- EVERY directed a->b hop with its class_key;
                 the percolation analysis needs the full graph, each edge
                 inheriting the barrier computed for its class.
    """
    struct = _as_structure(structure)
    classes = ion_site_classes(struct, working_ion, symprec)
    ion_idx = sorted(classes)
    if not ion_idx:
        raise ValueError(f"no {working_ion} sites in the structure")

    distinct, edges = {}, []
    for a in ion_idx:
        for nbr in struct.get_neighbors(struct[a], max_hop):
            b = nbr.index
            if b not in classes:
                continue
            jimage = tuple(int(x) for x in nbr.image)
            if b == a and jimage == (0, 0, 0):
                continue
            dist = round(float(nbr.nn_distance), 6)
            # unordered pair invariant: A->B and B->A are the same class
            lo, hi = sorted((classes[a], classes[b]))
            class_key = f"{lo}-{hi}-{round(dist, round_dist)}"
            hop = {"a": a, "b": b, "jimage": jimage, "distance": dist,
                   "class_key": class_key}
            edges.append(hop)
            if class_key not in distinct or dist < distinct[class_key]["distance"]:
                distinct[class_key] = hop
    if not distinct:
        raise ValueError(f"no {working_ion}-{working_ion} hops within "
                         f"{max_hop} A -- raise max_hop")
    return distinct, edges


def hop_endpoints_vacancy(structure, hop, working_ion):
    """(initial, final) for a vacancy hop, identical atom ordering.

    initial: supercell with the ion at site b removed (vacancy at B).
    final  : same cell, but the ion from site a moved INTO the B position
             (minimum-image aware -- the target is B's periodic image seen
             from A). The vacancy is now at A.
    """
    struct = _as_structure(structure)
    target = struct.lattice.get_cartesian_coords(
        struct[hop["b"]].frac_coords + np.array(hop["jimage"]))

    initial = struct.copy()
    initial.remove_sites([hop["b"]])
    # removing b shifts indices above it down by one
    a_new = hop["a"] if hop["a"] < hop["b"] else hop["a"] - 1
    if initial[a_new].specie.symbol != working_ion:
        raise RuntimeError("index bookkeeping broke: moving site is not "
                           f"{working_ion}")
    final = initial.copy()
    final.translate_sites([a_new], target - final[a_new].coords,
                          frac_coords=False, to_unit_cell=False)
    return initial, final, a_new


def hop_endpoints_dilute(host, site_a_frac, site_b_frac, jimage, working_ion):
    """(initial, final) for a single ion in the empty host.

    The ion is APPENDED (last index) at fractional site A / at B + jimage --
    same host ordering, same moving-atom index, by construction. Site
    fractional coordinates come from the discharged supercell (mapped onto
    the host lattice); the engine's endpoint pre-relax cleans up the mismatch
    with the relaxed empty framework.
    """
    host = _as_structure(host)
    initial = host.copy()
    initial.append(working_ion, np.asarray(site_a_frac), coords_are_cartesian=False)
    final = host.copy()
    final.append(working_ion, np.asarray(site_b_frac) + np.array(jimage),
                 coords_are_cartesian=False)
    return initial, final, len(host)


class _OffsetUnionFind:
    """Union-find over arbitrary node ids carrying integer cell offsets, so
    a cycle with a nonzero net offset == a path that wraps the periodic cell
    (the textbook percolation-on-a-torus detector)."""

    def __init__(self):
        self.parent = {}
        self.offset = {}

    def _add(self, i):
        if i not in self.parent:
            self.parent[i] = i
            self.offset[i] = np.zeros(3, dtype=int)

    def find(self, i):
        """(root, offset of i relative to root), with path compression."""
        self._add(i)
        path = []
        while self.parent[i] != i:
            path.append(i)
            i = self.parent[i]
        root = i
        total = np.zeros(3, dtype=int)
        for node in reversed(path):          # root-adjacent first
            total = total + self.offset[node]
            self.parent[node] = root
            self.offset[node] = total.copy()
        off = self.offset[path[0]] if path else np.zeros(3, dtype=int)
        return root, off.copy()

    def union(self, i, j, jimage):
        """Connect i (home cell) with j (cell + jimage).
        Returns None if a new tree link was made, else the cycle vector."""
        ri, oi = self.find(i)
        rj, oj = self.find(j)
        shift = np.asarray(jimage, dtype=int)
        if ri != rj:
            self.parent[rj] = ri
            self.offset[rj] = oi + shift - oj
            return None
        return oi + shift - oj


def percolation_thresholds(edges, barriers):
    """Barrier at which the hop network first percolates in 1 / 2 / 3
    independent directions.

    Parameters
    ----------
    edges : list[dict]
        The FULL edge list from enumerate_hops (a, b, jimage, class_key);
        a/b are used as opaque node ids -- no remapping needed.
    barriers : dict
        class_key -> barrier (eV). Use max(fwd, rev) of the computed NEB for
        a conservative percolation number. Classes missing here (failed NEB)
        simply never open -- fail loudly upstream, degrade gracefully here.

    Returns
    -------
    {"e_m_1d": float|None, "e_m_2d": ..., "e_m_3d": ...} -- None when that
    dimensionality is never reached with the available edges.
    """
    weighted = sorted(
        ((barriers[e["class_key"]], e) for e in edges
         if e["class_key"] in barriers and barriers[e["class_key"]] is not None),
        key=lambda t: t[0])

    uf = _OffsetUnionFind()
    cycles = []
    thresholds = {"e_m_1d": None, "e_m_2d": None, "e_m_3d": None}
    rank = 0
    for barrier, edge in weighted:
        cycle = uf.union(edge["a"], edge["b"], edge["jimage"])
        if cycle is not None and np.any(cycle):
            cycles.append(cycle)
            new_rank = int(np.linalg.matrix_rank(np.array(cycles)))
            while rank < new_rank:
                rank += 1
                thresholds[f"e_m_{rank}d"] = float(barrier)
        if rank == 3:
            break
    return thresholds


def hop_summary(distinct, results):
    """Merge NEB results back onto the distinct hops.

    ``results``: {class_key: engine result dict (barrier_fwd/rev, converged,
    energies, ...)}. Returns a JSON-ready list sorted by barrier and the
    barriers map for percolation (max(fwd, rev), converged hops only).
    """
    rows, barriers = [], {}
    for key, hop in distinct.items():
        res = results.get(key)
        row = {"class_key": key, "distance": hop["distance"],
               "a": hop["a"], "b": hop["b"]}
        if res is None:
            row.update({"barrier_fwd": None, "barrier_rev": None,
                        "converged": False, "error": "no NEB result"})
        else:
            row.update({"barrier_fwd": res.get("barrier_fwd"),
                        "barrier_rev": res.get("barrier_rev"),
                        "converged": bool(res.get("converged")),
                        "error": res.get("error")})
            if row["converged"] and row["barrier_fwd"] is not None:
                barriers[key] = max(row["barrier_fwd"], row["barrier_rev"])
        rows.append(row)
    rows.sort(key=lambda r: (r["barrier_fwd"] is None,
                             r["barrier_fwd"] or 0.0))
    return rows, barriers
