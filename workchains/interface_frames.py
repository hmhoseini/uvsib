"""Write MLIP-relaxed interface frames to MongoDB for DFT labelling.

The active-learning half of the solid-state cell path (docs/interfaces.md).
Frames harvested along an interface relaxation are pushed to a MongoDB
collection; a DFT consumer picks them up on the HPC side and returns r2SCAN
energy/forces/stress that feed the next fine-tune generation.

WHY A PLAIN COLLECTION AND NOT A FIREWORKS LAUNCHPAD. uvsib does not depend on
FireWorks, and hand-writing FireWorks documents from outside the library is
brittle across versions. The schema here is ours, flat, and documented below;
turning it into VASP jobs is one small consumer script that can live wherever
the DFT is run. Provenance travels WITH each frame, so a label can always be
traced back to the junction and the generation that produced it.

Connection settings come from ``input.yaml`` under ``battery: interfaces:
mongo:`` -- interfaces are a branch of the battery tree, so they are configured
there rather than at top level.

Document schema (one per frame):

    {
      "generation":   int,          # AL generation, from input.yaml
      "status":       "pending",    # pending -> running -> done | failed
      "composition":  str,          # parent electrode formula
      "electrode":    str,
      "electrolyte":  str,
      "working_ion":  str,
      "half_cell":    "anode" | "cathode",
      "label":        str,          # "<electrode>|<electrolyte>"
      "interface_uuid": str,        # DBInterface.structure_uuid or build uuid
      "film_miller":  [int,int,int],
      "substrate_miller": [int,int,int],
      "termination":  [str, str],
      "step":         int,          # position along the relaxation
      "fmax":         float,        # eV/A at that step (active atoms only)
      "mlip_energy":  float,        # the MLIP's own value -- NOT a label
      "n_atoms":      int,
      "structure":    {...},        # pymatgen Structure.as_dict()
      "model":        str,          # MLIP that produced the geometry
      "model_head":   str,
      "dft":          {...} | None, # filled by the consumer: energy/forces/stress
      "ctime":        datetime,
    }

``mlip_energy`` is stored for bookkeeping only. It is the geometry generator's
own prediction and must never be used as a training label -- training a model
on its own output teaches it nothing and hardens its errors.
"""
import datetime


def _client(cfg):
    """pymongo client from the ``battery: interfaces: mongo:`` block.

    Imported lazily so uvsib keeps working without pymongo for every path that
    does not push frames.
    """
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise ImportError(
            "pymongo is required to push interface frames "
            "(battery: interfaces: mongo: in input.yaml). pip install pymongo"
        ) from exc

    kwargs = {"host": cfg["host"], "port": int(cfg.get("port", 27017))}
    if cfg.get("user"):
        kwargs.update(username=cfg["user"], password=cfg.get("password"),
                      authSource=cfg.get("auth_source", "admin"))
    if cfg.get("tls_ca") or cfg.get("tls_cert"):
        kwargs.update(tls=True, tlsCAFile=cfg.get("tls_ca"),
                      tlsCertificateKeyFile=cfg.get("tls_cert"))
    return MongoClient(**kwargs)


def push_frames(cfg, frames):
    """Insert frame documents; returns the number written.

    Raises rather than returning 0 on a connection or write failure -- a
    silently empty push looks exactly like a relaxation that produced no
    frames, and the two need different fixes.
    """
    if not frames:
        return 0
    client = _client(cfg)
    try:
        coll = client[cfg["db_name"]][cfg.get("collection", "interface_frames")]
        # index the fields a consumer actually queries on; idempotent
        coll.create_index([("status", 1), ("generation", 1)])
        coll.create_index([("label", 1)])
        res = coll.insert_many(frames, ordered=False)
        return len(res.inserted_ids)
    finally:
        client.close()


def frame_documents(base, relax_result, generation, model, head):
    """Turn one interface_relax result into frame documents.

    ``base`` carries the provenance shared by every frame of this junction
    (electrode, electrolyte, working_ion, millers, termination, ...).
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    docs = []
    for fr in relax_result.get("frames", []):
        doc = dict(base)
        doc.update({
            "generation": int(generation),
            "status": "pending",
            "step": fr.get("step"),
            "fmax": fr.get("fmax"),
            "mlip_energy": fr.get("energy"),
            "structure": fr.get("structure"),
            "n_atoms": len(fr.get("structure", {}).get("sites", []) or []),
            "model": model,
            "model_head": head,
            "dft": None,
            "ctime": now,
        })
        docs.append(doc)
    return docs


def pending_count(cfg, generation=None):
    """How many frames still await DFT. Handy for a status report."""
    client = _client(cfg)
    try:
        coll = client[cfg["db_name"]][cfg.get("collection", "interface_frames")]
        q = {"status": "pending"}
        if generation is not None:
            q["generation"] = int(generation)
        return coll.count_documents(q)
    finally:
        client.close()
