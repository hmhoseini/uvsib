from itertools import combinations
from sqlalchemy import inspect, delete
from pymatgen.core import Composition, Structure
from db.session import get_session
from db.tables import DBChemsys, DBStructure, DBStructureVersion
from db.query import query_by_columns

def update_row(table_class, uuid_value, columns_values):
    """
    Update a row in a table by uuid (overwrite existing data).
    Args:
        table_class: table class (declarative)
        uuid_value: UUID value (either UUID object or str)
        columns_values: dict of column_name and value to update
    """
    # Remove uuid from update fields to avoid updating primary key
    update_values = dict(columns_values)

    stmt = (
        table_class.__table__
        .update()
        .where(table_class.uuid == uuid_value)
        .values(**update_values)
    )
    with get_session() as session:
        session.execute(stmt)
        session.commit()

def get_chemical_systems(chemical_formula, new=True):
    """
    Given a chemical formula, return iether all or new chemical systems.
    """
    comp = Composition(chemical_formula)
    elements = sorted(el.symbol for el in comp.elements)

    subsystems = []
    repeated_chemsys = []
    new_chemsys = []

    for n in range(1, len(elements) + 1):
        for combo in combinations(elements, n):
            subsystem = "-".join(combo)
            subsystems.append(subsystem)

    for subsystem in subsystems:
        result = query_by_columns(DBChemsys,
                                  {"chemsys": subsystem}
        )
        if result:
            repeated_chemsys.append(subsystem)
        else:
            new_chemsys.append(subsystem)
    if new:
        return new_chemsys
    return subsystems

def add_structures(chemical_system,
                   ML_model,
                   selected_structures,
                   selected_energies,
                   selected_epas
    ):
    with get_session() as session:
        db_structures = []
        db_versions = []

        for indx, structure in enumerate(selected_structures):
            composition = str(Structure.from_dict(structure).composition.reduced_formula)
            added_struct = DBStructure(
                    composition=composition,
                    chemsys=chemical_system,
            )
            db_structures.append(added_struct)

        session.add_all(db_structures)
        session.commit()

        for indx, added_struct in enumerate(db_structures):
            structure_version = DBStructureVersion(
                    structure_uuid=added_struct.uuid,
                    method=ML_model,
                    structure=selected_structures[indx],
                    energy=selected_energies[indx],
                    epa=selected_epas[indx]
            )
            db_versions.append(structure_version)

        session.add_all(db_versions)
        session.commit()


def add_row(table_class, rows_data):
    """
    Add one or more rows to a table.

    Parameters:
    - table_class
    - rows_data: list of dictionaries (each dict is one row), or a single dict

    Returns:
    - List of newly added row objects
    """
    if isinstance(rows_data, dict):
        rows_data = [rows_data]  # Allow single dict input

    new_rows = [table_class(**row_data) for row_data in rows_data]

    with get_session() as session:
        session.add_all(new_rows)
        session.commit()
    return new_rows

def delete_row(table_class, row):
    """
    Parameters:
    - table_class: SQLAlchemy model class (e.g., DBFrontend)
    - row to be deleted
    """
    with get_session() as session:
        stmt = delete(table_class).where(table_class.uuid == row.uuid)
        session.execute(stmt)
        session.commit()

def get_table_data(session, table_class):
    """
    Fetch all rows from the given table class and return as a list of dictionaries
    Args:
        Session: SQLAlchemy session factory
        table_class: SQLAlchemy ORM model class
    """
    try:
        rows = session.query(table_class).all()
        columns = [column.name for column in inspect(table_class).columns]
        return [{col: getattr(row, col) for col in columns} for row in rows]
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        session.close()

def delete_all_rows(session, table_class):
    """
    Deletes all rows from the given SQLAlchemy table class
    Args:
        Session: SQLAlchemy session factory
        table_class: SQLAlchemy ORM model class
    """
    try:
        deleted_count = session.query(table_class).delete()
        session.commit()
        print(f"Deleted {deleted_count} rows from '{table_class.__tablename__}'")
    except Exception as e:
        session.rollback()
        print(f"Error deleting rows: {e}")
    finally:
        session.close()

def print_all_rows(session, table_class):
    """
    Reflect the table by name and print all rows.
    """
    rows = []
    rows = get_table_data(session, table_class)
    if rows:
        print(f"\nRows from table '{table_class.__table__}':")
        for row in rows:
            print(row)
