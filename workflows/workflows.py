from pymatgen.core.structure import Composition
from uvsib.db.tables import DBFrontend, DBChemsys, DBComposition
from uvsib.db.utils import update_row, add_row, get_chemical_systems, query_by_columns
from uvsib.workchains.submit import submit_mainworkchain

def add_from_frontend(dict_from_frontend_list):
    """Process frontend submissions and update the database accordingly."""
    for entry in dict_from_frontend_list:
        chemical_formula = Composition(entry["chemical_formula"]).reduced_formula
        user = entry["user"]
        model = entry["model"]
        reaction = entry["reaction"]
        retry = entry["retry"]

        existing_frontend_rows = query_by_columns(
                DBFrontend,
                {"composition": chemical_formula}
        )
        user_already_exists = any(row.username == user for row in existing_frontend_rows)

        if not user_already_exists:
            add_row(DBFrontend, {
                "username": user,
                "composition": chemical_formula,
                "reaction": reaction,
                "model": model}
            )

        # check if composition is already processed
        existing_composition = query_by_columns(
                DBComposition,
                {"composition": chemical_formula}
        )
        if not existing_composition:
            add_row(DBComposition,
                    {"composition": chemical_formula}
            )

        if user_already_exists:
            if not retry or existing_composition[0].status == "Running":
                continue
        # only new chemical systems
        new_chemsys = get_chemical_systems(chemical_formula, new=True)
        for chemsys in new_chemsys:
            add_row(DBChemsys, {"chemsys": chemsys})
        # submit the main workflow
        submit_mainworkchain(
                chemical_formula,
                new_chemsys,
                model,
                reaction,
        )
        # update DBFrontend
        update_dbfrontend()

def update_dbfrontend():
    """Updateing DBFrontend status"""
    for status in ["Created", "Running", "Failed"]:
        db_fe_rows = query_by_columns(DBFrontend, {"status": status})
        for fe_row in db_fe_rows:
            db_c_row = query_by_columns(DBComposition, {"composition": fe_row.composition})[0]
            update_row(DBFrontend, fe_row.uuid,
                        {"status": db_c_row.status,
                        "step_status": db_c_row.step_status
                        }
            )
