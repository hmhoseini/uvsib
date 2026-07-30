from pymatgen.core.structure import Composition
from aiida.orm import QueryBuilder, WorkChainNode
from uvsib.db.tables import DBFrontend, DBChemsys, DBComposition, DBSurfaceMLAdsorbate, DBNanoParticles
from uvsib.db.utils import update_row, add_row, delete_row, get_chemical_systems, query_by_columns
from uvsib.workchains.submit import submit_mainworkchain


def check_valid(reaction, reaction_path):
    # BATTERY is the bulk (deintercalation) pathway: the "path" is the working
    # ion, and the run skips surface builder + adsorbates (see docs/batteries.md)
    from uvsib.workchains.cer import CER_PATHWAYS
    from uvsib.workchains.co2rr import CO2RR_PATHWAYS
    from uvsib.workchains.her import HER_PATHWAYS
    from uvsib.workchains.noxrr import NOXRR_PATHWAYS
    from uvsib.workchains.nrr import NRR_PATHWAYS
    from uvsib.workchains.orr import ORR_PATHWAYS
    # Path lists come from the single source of truth (the *_PATHWAYS dict of
    # each workchain) so this gate cannot drift out of sync again — a stale
    # hand-copied list here is what blocked ORR/HER/NRR/CER and the newer
    # CO2RR chains. OER has no pathway dict; 'default' is its only route.
    implemented_reactions = {'OER': ['default'],
                             'HER': sorted(HER_PATHWAYS),
                             'ORR': sorted(ORR_PATHWAYS),
                             'CER': sorted(CER_PATHWAYS),
                             'NRR': sorted(NRR_PATHWAYS),
                             'CO2RR': sorted(CO2RR_PATHWAYS),
                             'NOXRR': sorted(NOXRR_PATHWAYS)}
    # Normalize case so 'battery'/'Li'/'li' all work, and so one canonical
    # spelling reaches the DB rows, step_status keys, and workchain labels.
    reaction = reaction.strip().upper()
    if reaction not in implemented_reactions:
        raise NotImplementedError(f"Reaction {reaction} not implemented.")
    if reaction == 'BATTERY':
        reaction_path = reaction_path.strip().capitalize()   # ion symbol: li -> Li
    else:
        reaction_path = reaction_path.strip().lower()
    if reaction_path not in implemented_reactions[reaction]:
        raise NotImplementedError(f"Path {reaction_path} not implemented for {reaction}.")
    return reaction, reaction_path

def add_from_frontend(dict_from_frontend_list):
    """Process frontend submissions and update the database accordingly."""
    # Count reactions per composition so we only gate (wait) when there are
    # follower reactions that would benefit from reusing the shared steps.
    reactions_per_comp = {}
    for e in dict_from_frontend_list:
        comp = Composition(e["chemical_formula"]).reduced_formula
        reactions_per_comp[comp] = reactions_per_comp.get(comp, 0) + 1

    # pioneered = set()   # compositions whose shared steps a pioneer reaction owns
    for entry in dict_from_frontend_list:
        chemical_formula = Composition(entry["chemical_formula"]).reduced_formula
        user = entry["user"]
        reaction = entry["reaction"]
        reaction_path = entry["reaction_path"]

        retry = entry["retry"] if "retry" in entry else False

        if "nano_particles" in entry:
            nano = entry['nano_particles']
        else:
            nano = False

        if "similarities" in entry:
            similars = entry['similarities']
        else:
            similars = {}

        # The SQS request payload (parent structure, sublattices, composition
        # grid, surfaces, defects) rides through verbatim; {} means a normal
        # (non-SQS) submission. submit_mainworkchain wraps it in an aiida Dict.
        sqs = entry.get("sqs", {})

        check_valid(reaction, reaction_path)

        existing_frontend_rows = query_by_columns(DBFrontend,{"composition": chemical_formula})
        user_already_exists = any(row.username == user for row in existing_frontend_rows)

        if not user_already_exists:
            add_row(DBFrontend, {
                "username": user,
                "composition": chemical_formula,
                "reaction": reaction,
                "reaction_path": reaction_path,
                "nano_particles": nano}
            )

        # check if a composition is already processed
        existing_composition = query_by_columns(DBComposition, {"composition": chemical_formula})
        if not existing_composition:
            add_row(DBComposition, {"composition": chemical_formula})

        # check if elements are processes for nano particles
        elements = '-'.join(sorted(list([str(el) for el in Composition(chemical_formula).elements])))
        existing_particles = query_by_columns(DBNanoParticles, {"elements": elements})
        if not existing_particles:
            add_row(DBNanoParticles,{"elements": elements})

#        # remove crashed chemical systems
#        chemical_systems, _ = get_chemical_systems(chemical_formula)
#        for chemsys in chemical_systems:
#            r = query_by_columns(DBChemsys, {"chemsys": chemsys})
#            if r:
#                row = r[0]
#                if not row.gen_structures:
#                    label = f"CatalystChain {reaction}:{reaction_path} on {chemical_formula}"
#                    killed = QueryBuilder().append(
#                        WorkChainNode,
#                        filters={"label": label,
#                                 "attributes.process_state": {"in": ["killed"]}},
#                    ).all()
#                    if killed:
#                        pass

        # only new chemical systems
        _, new_chemsys = get_chemical_systems(chemical_formula)
        for chemsys in new_chemsys:
            add_row(DBChemsys, {"chemsys": chemsys})

        # (a) already finished -> a result row exists
        row = query_by_columns(DBSurfaceMLAdsorbate, {"composition": chemical_formula,
                                                      "reaction": reaction,
                                                      "reaction_path": reaction_path})
        if row:
            continue

        # (b) already in flight -> an active MainWorkChain carries this label;
        # (c) failed + no retry  -> a terminated MainWorkChain carries this label.
        # Label must match launch_calculations.get_inputs_and_processclass_from_extras.
        if nano:
            label = f"NanoParticleChain: {chemical_formula}"
        else:
            label = f"CatalystChain {reaction}:{reaction_path} on {chemical_formula}"
        try:
            active = QueryBuilder().append(
                WorkChainNode,
                filters={"label": label,
                         "attributes.process_state": {"in": ["created", "running", "waiting"]}},
            ).count()
        except Exception:
            active = 0
        if active:
            continue
        if not retry:
            ran_before = QueryBuilder().append(
                WorkChainNode,
                filters={"label": label,
                         "attributes.process_state": {"in": ["finished", "excepted", "killed"]}},
            ).count()
            if ran_before:        # ran before without a result row -> failed
                continue

        submit_mainworkchain(chemical_formula=chemical_formula, chemical_systems=new_chemsys,
                             reaction=reaction, reaction_path=reaction_path,
                             nano=nano, similarities=similars, sqs=sqs)
        update_dbfrontend()

def update_dbfrontend():
    """Updateing DBFrontend status"""
    for status in ["Created", "Running", "Failed"]:
        db_fe_rows = query_by_columns(DBFrontend, {"status": status})
        for fe_row in db_fe_rows:
            db_c_row = query_by_columns(DBComposition,{"composition": fe_row.composition})[0]
            update_row(DBFrontend, fe_row.uuid,{"status": db_c_row.status, "step_status": db_c_row.step_status})
        #
        db_np_rows = query_by_columns(DBNanoParticles, {"status": status})
        for np_row in db_np_rows:
            db_row = query_by_columns(DBNanoParticles,{"elements": np_row.elements})[0]
            update_row(DBNanoParticles, np_row.uuid,{"status": db_row.status, "step_status": db_row.step_status})
