"""One-off backfill of ``DBStructure.mp_id`` / ``DBStructureVersion.mp_id`` for
MPDB-sourced structures that were added before the mp_id column was tracked.

``add_from_mpdb`` only fetches from the Materials Project when a composition has
no ``MPDB_stb`` / ``MPDB_exp`` rows yet, so re-running the pipeline will NOT
backfill ids onto structures that are already stored. This script re-queries the
MPDB for each affected composition, matches the returned structures against the
stored ones with the same ``StructureMatcher`` the pipeline uses, and writes the
id onto every version of the matched structure plus its parent ``DBStructure``.

Run it from an environment with an AiiDA profile loaded (the same one the
pipeline runs in):

    python -m uvsib.db.backfill_mp_id            # every affected composition
    python -m uvsib.db.backfill_mp_id Ti2O3 ...  # only the named compositions
"""
import sys

from pymatgen.core import Structure

from uvsib.codes.utils import get_structures_from_mpdb_by_composition
from uvsib.db.session import get_session
from uvsib.db.tables import DBStructure, DBStructureVersion
from uvsib.workchains.utils import EHULL_SCAN, get_primitive_cell, matcher

MPDB_SOURCES = ("MPDB_stb", "MPDB_exp")


def _compositions_needing_backfill(session, only=None):
    rows = (
        session.query(DBStructureVersion.composition)
        .filter(DBStructureVersion.source.in_(MPDB_SOURCES))
        .filter(DBStructureVersion.mp_id.is_(None))
        .distinct()
        .all()
    )
    comps = {c for (c,) in rows}
    if only:
        comps &= set(only)
    return sorted(comps)


def backfill(only=None):
    with get_session() as session:
        comps = _compositions_needing_backfill(session, only)
        if not comps:
            print("nothing to backfill")
            return

        for comp in comps:
            print(f"{comp}:")
            stable, experimental = get_structures_from_mpdb_by_composition(comp, EHULL_SCAN)
            fetched = [
                (get_primitive_cell(struct_dict), mp_id)
                for struct_dict, mp_id in (stable + experimental)
                if mp_id
            ]

            versions = (
                session.query(DBStructureVersion)
                .filter(DBStructureVersion.composition == comp)
                .filter(DBStructureVersion.source.in_(MPDB_SOURCES))
                .all()
            )
            by_uuid = {}
            for v in versions:
                by_uuid.setdefault(v.structure_uuid, []).append(v)

            for struct_uuid, uuid_versions in by_uuid.items():
                if all(v.mp_id for v in uuid_versions):
                    continue
                stored = Structure.from_dict(uuid_versions[0].structure)
                match = next(
                    (mp_id for prim, mp_id in fetched if matcher.fit(prim, stored)),
                    None,
                )
                if not match:
                    print(f"  {struct_uuid}: no MPDB match, skipped")
                    continue
                for v in uuid_versions:
                    v.mp_id = match
                parent = session.get(DBStructure, struct_uuid)
                if parent is not None:
                    parent.mp_id = match
                print(f"  {struct_uuid} -> {match}")

        session.commit()


if __name__ == "__main__":
    backfill(sys.argv[1:] or None)
