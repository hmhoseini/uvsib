"""Ingest solvation_frames.py output into db_finetune_frames.

    python -m uvsib.db.ingest_frames output.json --batch cuau-gen0 [--generation 0]

One row per frame, carrying the full attribution (composition, surface_id,
bulk_uuid, reaction) and the harvest metadata (barriers, convergence, ...)
so the DFT batch export (run_dir/export_all.py --finetune-frames) and the
later labeling stage never have to touch the harvest files again.

Refuses to ingest into an existing batch unless --append is given -- a
re-run of the same harvest must not silently duplicate frames.
"""
import argparse
import json
import sys

from uvsib.db.session import get_session
from uvsib.db.tables import DBFinetuneFrame


def ingest(path, batch, generation, append=False):
    with open(path) as f:
        data = json.load(f)
    frames = data.get("frames", [])
    if not frames:
        raise SystemExit(f"{path}: no frames to ingest")
    model = data.get("model")

    with get_session() as session:
        existing = (session.query(DBFinetuneFrame)
                    .filter(DBFinetuneFrame.batch == batch).count())
        if existing and not append:
            raise SystemExit(
                f"batch '{batch}' already holds {existing} frames; "
                "use --append (or a new batch name) to add more")

        for fr in frames:
            task = fr.get("task") or {}
            session.add(DBFinetuneFrame(
                batch=batch,
                generation=generation,
                kind=fr["kind"],
                composition=task.get("composition"),
                model=model,
                surface_id=task.get("surface_id"),
                bulk_uuid=task.get("bulk_uuid"),
                reaction=task.get("reaction"),
                reaction_path=task.get("reaction_path"),
                structure=fr["structure"],
                energy_model=fr.get("energy_model"),
                status="new",
                attributes={"meta": fr.get("meta"),
                            "miller_index": task.get("miller_index"),
                            "tag": task.get("tag")},
            ))
        session.commit()
        total = (session.query(DBFinetuneFrame)
                 .filter(DBFinetuneFrame.batch == batch).count())
    print(f"ingested {len(frames)} frames into batch '{batch}' "
          f"(generation {generation}); batch now holds {total}")
    if data.get("failed_tasks"):
        print(f"NOTE: the harvest recorded {len(data['failed_tasks'])} "
              f"failed task(s): {data['failed_tasks']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_json", help="solvation_frames.py output.json")
    ap.add_argument("--batch", required=True,
                    help="batch name, e.g. cuau-gen0")
    ap.add_argument("--generation", type=int, default=0,
                    help="active-learning generation (default 0)")
    ap.add_argument("--append", action="store_true",
                    help="allow adding frames to an existing batch")
    args = ap.parse_args()
    ingest(args.output_json, args.batch, args.generation, args.append)


if __name__ == "__main__":
    main()
