import sys
import uuid
from db.session import get_session
from db.query import query_by_columns
from db.tables import DBFrontend, DBChemsys, DBComposition
from db.utils import update_row, add_row, get_chemical_systems
from workchains.submit import submit_mainworkchain

def add_from_frontend(dict_from_frontend_list):
    """Add input from the frontend to the database"""
    with get_session() as session:
        for entry in dict_from_frontend_list:
            user = entry["user"]
            comp = entry["chemical_formula"]
            process = entry["process"]
            metadata = entry["metadata"]

            # composition already exists in DBComposition
            db_comp = query_by_columns(DBComposition, {"composition": comp})

            if db_comp:
                if len(db_comp) > 1:
                    print(f"ERROR: Multiple entries found for composition: {comp}")
                    sys.exit()

                print(f"Compound {comp} is already in DBComposition.")
                comp_row = db_comp[0]

                # check if frontend record for this user already exists
                db_front = query_by_columns(DBFrontend, {"username": user, "composition": comp})
                if not db_front:
                    new_row = add_row(DBFrontend,
                                      {"username": user,
                                       "composition": comp,
                                       "attributes": metadata,
                                       "status": comp_row.status,
                                       "step_status": comp_row.step_status
                                      }
                    )
                continue

            # if not found in DBComposition
            db_front = query_by_columns(DBFrontend, {"username": user, "composition": comp})
            if not db_front:
                # add row to DBFrontend
                new_row = add_row(DBFrontend,
                                  {"username": user,
                                   "composition": comp,                           
                                   "attributes": metadata
                                  }
                )
                # add row to DBComposition
                new_row = add_row(DBComposition,
                                  {"composition": comp,
                                   "status": "Running",
                                   "step_status":{"phase_diagram_ML": "Running"}
                                  }
                )

                new_chemsys = get_chemical_systems(comp, new=True)
                for chemsys in new_chemsys:
                    add_row(DBChemsys,
                            {"chemsys": chemsys}
                    )
                submited_pks = submit_mainworkchain(comp, new_chemsys, str(uuid.uuid4()))
                update_frontend()
            else:
                print(f"Composition {comp} has already been submitted by user {user}")

def update_frontend():
    """Updateing DBFrontend status"""
    for status in ["Created", "Running"]:
        db_fe_rows = query_by_columns(DBFrontend, {"status": status})
        for fe_row in db_fe_rows:
            db_c_row = query_by_columns(DBComposition, {"composition": fe_row.composition})[0]
            update_row(DBFrontend, fe_row.uuid,
                        {"status": db_c_row.status,
                        "step_status": db_c_row.step_status
                        }
            )
