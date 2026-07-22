"""
Experimental-structure injection helpers -- pure python (pymatgen only, no
AiiDA), unit-testable standalone.

Part of the anti-"MLIP-energy-lottery" fix: experimentally-known MP
structures are (a) injected verbatim into the csp/gen relax bundles (tagged
source="mp_experimental" in the DB) and (b) force-included into the
stable_struct manifest even when the ML hull window would drop them. The
helpers here do the two fiddly, testable parts:

  split_output_slices : split one bundled relax output back into its input
                        groups (generated | injected | references) by the
                        runner's original-input ``indices`` -- robust to
                        non-converged structures being dropped.
  dedup_forced        : the force-include filter -- keep only experimental
                        structures NOT structurally equivalent to anything
                        the ML selection already picked (or to each other),
                        so the manifest never carries the same host twice.
"""
from __future__ import annotations

from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.entries.computed_entries import ComputedStructureEntry

# DELIBERATELY TIGHTER than the selection dedup in workchains/utils.py
# (ltol=0.3/stol=0.5/angle_tol=7): the errors are asymmetric. A false merge
# here silently drops an experimentally-known polymorph -- the exact failure
# this module exists to kill (verified: the loose tolerances merge e.g.
# rocksalt with zincblende; pymatgen defaults keep them apart while still
# merging volume-scaled copies, and both keep maricite/olivine NaFePO4
# distinct). A missed merge merely leaves one redundant host in the manifest.
# Consequence: a borderline pair the loose selection matcher merged may
# reappear via force-include -- redundant, cheap, and safe.
MATCHER = StructureMatcher(
    ltol=0.2,
    stol=0.3,
    angle_tol=5,
    scale=True,
    attempt_supercell=False,
    allow_subset=False,
    primitive_cell=True,
)


def split_output_slices(output_dict, sizes):
    """Split a bundled relax output into consecutive input slices.

    Parameters
    ----------
    output_dict : dict
        The runner's output.json payload: ``structures`` / ``energies`` and
        ``indices`` (original input position of each surviving structure;
        missing/short indices fall back to output order, legacy jobs).
    sizes : list[int]
        Input-bundle group sizes in submission order, e.g.
        ``[n_generated, n_injected, n_references]``. Must cover every index.

    Returns
    -------
    list[list[ComputedStructureEntry]] -- one list per group, dropped
    (non-converged) structures simply absent from their group.
    """
    structs = output_dict.get("structures", [])
    energies = output_dict.get("energies", [])
    indices = output_dict.get("indices")
    if not indices or len(indices) != len(structs):
        indices = list(range(len(structs)))

    boundaries = []
    total = 0
    for size in sizes:
        total += size
        boundaries.append(total)
    if indices and max(indices) >= total:
        raise ValueError(f"index {max(indices)} outside the declared bundle "
                         f"(sizes {sizes})")

    groups = [[] for _ in sizes]
    for struct, energy, idx in zip(structs, energies, indices):
        entry = ComputedStructureEntry(
            structure=Structure.from_dict(struct), energy=energy)
        for g, bound in enumerate(boundaries):
            if idx < bound:
                groups[g].append(entry)
                break
    return groups


def dedup_forced(selected_structures, candidates, matcher=None):
    """Filter force-include candidates against the ML selection (and each
    other): only structurally NEW experimental hosts survive.

    Parameters
    ----------
    selected_structures : list[Structure | dict]
        What unique_low_energy_comp already put into the manifest.
    candidates : list[(uuid_str, Structure | dict)]
        The stored mp_experimental rows for this composition, in the order
        they should be considered (e.g. by ehull).
    matcher : StructureMatcher, optional
        Defaults to MATCHER (the selection-dedup tolerances).

    Returns
    -------
    list[(uuid_str, Structure)] -- candidates to force-include, deduplicated.
    """
    matcher = matcher or MATCHER

    def _as_struct(s):
        return s if isinstance(s, Structure) else Structure.from_dict(s)

    seen = [_as_struct(s) for s in selected_structures]
    kept = []
    for uuid, struct in candidates:
        struct = _as_struct(struct)
        if any(matcher.fit(struct, other) for other in seen):
            continue
        kept.append((uuid, struct))
        seen.append(struct)
    return kept
