from itertools import combinations
from sqlalchemy import inspect, delete, select, text
from sqlalchemy.orm import aliased
from pymatgen.core import Composition, Structure
from uvsib.db.session import get_session
from uvsib.db.tables import DBChemsys, DBStructure, DBStructureVersion, DBSurface

def add_surface_adsorbate(structure_uuid, adsorbate_name, structure, adsorption_energy, attributes = None):
    """Store a new DBSurfaceAdsorbate row corresponding to a given DBStructure UUID."""
    with get_session() as session:
        pass

    return True

def get_structure_uuid_surface_id(composition):
    """
    Return (structure_uuid, surface_id) tuples for all DBStructure rows
    with the given composition that have corresponding DBSurface entries.
    """
    structure = aliased(DBStructure)
    surface = aliased(DBSurface)

    query = (
        select(
            structure.uuid.label("structure_uuid"),
            surface.id.label("surface_id"),
        )
        .join(surface, surface.structure_uuid == structure.uuid)
        .where(structure.composition == composition)
    )

    with get_session() as session:
        results = session.execute(query).all()

    return [(row.structure_uuid, row.surface_id) for row in results]

def add_slab(existing_uuid, slab_dict):
    with get_session() as session:
        structure_row = (
            session.query(DBStructure)
            .filter_by(uuid=existing_uuid)
            .first()
        )
        slab = DBSurface(
            structure_uuid=existing_uuid,
            slab=slab_dict,
        )

        session.add(slab)
        session.commit()
        return True

####################################

def add_structures(
        source,
        method,
        structure_energy_pairs
    ):
    """Add new structures and associated energies to the database"""
    with get_session() as session:
        db_structures = []
        db_versions = []

        for struct_dict, energy in structure_energy_pairs:
            struct = Structure.from_dict(struct_dict)
            composition = struct.composition.reduced_formula
            chemical_system = Composition(composition).chemical_system

            db_structure = DBStructure(
                composition=composition,
                chemsys=chemical_system,
            )
            session.add(db_structure)
            session.flush()

            db_version = DBStructureVersion(
                structure_uuid=db_structure.uuid,
                method=method,
                source=source,
                structure=struct_dict,
                energy=energy,
            )

            db_versions.append(db_version)
            db_structures.append(db_structure)

        session.add_all(db_versions)
        session.commit()

def add_version_to_existing_structure(
        existing_uuid,
        method,
        add_attributes,
        on_conflict = "error"):
    """
    Add a new version to an existing structure in the database.

    Parameters
    ----------
    existing_uuid : str
        UUID of the existing structure.
    method : str
        Method used to generate this version (e.g., "DFT", "MACE").
    add_attributes : dict
        Dictionary containing any additional attributes for DBStructureVersion.
    on_conflict : {"error", "ignore", "override"}, optional
        How to handle conflicts if a version with the same structure_uuid and
        method already exists.
        - "error": raise an exception (default)
        - "ignore": do nothing and return the existing version
        - "override": update the existing version with new attributes
    """
    with get_session() as session:
        existing_version = (
            session.query(DBStructureVersion)
            .filter_by(structure_uuid=existing_uuid, method=method)
            .first()
        )

        if existing_version:
            if on_conflict == "error":
                return False
            if on_conflict == "ignore":
                return True
            if on_conflict == "override":
                for key, value in add_attributes.items():
                    setattr(existing_version, key, value)
                session.commit()
                return True

        # If no conflict, add a new version
        db_version = DBStructureVersion(
            structure_uuid=existing_uuid,
            method=method,
            **add_attributes
        )
        session.add(db_version)
        session.commit()
        return True

def delete_structure(structure_filters: dict, **version_filters):
    """
    Delete DBStructureVersion entries (and possibly their parent DBStructure).
    
    Parameters
    ----------
    structure_filters : dict
        Filters for DBStructure (e.g., {'uuid': '...'}).
    version_filters : dict
        Filters for DBStructureVersion (e.g., version=1, status='draft').
    
    Returns
    -------
    int
        Number of DBStructureVersion rows deleted.
    """
    deleted_count = 0

    with get_session() as session:
        query = session.query(DBStructureVersion).join(DBStructure)

        # Apply filters on DBStructure
        for attr, value in structure_filters.items():
            query = query.filter(getattr(DBStructure, attr) == value)

        # Apply filters on DBStructureVersion
        for attr, value in version_filters.items():
            query = query.filter(getattr(DBStructureVersion, attr) == value)

        versions_to_delete = query.all()

        if versions_to_delete:
            # Track affected structures
            affected_structure_uuids = {v.structure_uuid for v in versions_to_delete}

            # Delete versions safely
            for v in versions_to_delete:
                session.delete(v)
                deleted_count += 1

            session.flush()

            # Delete parent structures if they have no remaining versions
            for uuid in affected_structure_uuids:
                remaining = (
                    session.query(DBStructureVersion)
                    .filter(DBStructureVersion.structure_uuid == uuid)
                    .count()
                )
                if remaining == 0:
                    session.query(DBStructure).filter(DBStructure.uuid == uuid).delete()

        session.commit()

def query_structure(structure_filters, **version_filters):
    """
    Query DBStructureVersion joined with DBStructure
    
    structure_filters: Dict of column names and values for DBStructure (e.g., {'uuid': '...'})
    version_filters: Keyword arguments for DBStructureVersion filters
    """
    with get_session() as session:
        query = session.query(DBStructureVersion).join(DBStructure)

        # Apply filters on DBStructure
        for attr, value in structure_filters.items():
            query = query.filter(getattr(DBStructure, attr) == value)

        # Apply filters on DBStructureVersion
        for attr, value in version_filters.items():
            query = query.filter(getattr(DBStructureVersion, attr) == value)

        return query.all()

def query_structureversions_by_attributes(**filters):
    """Query DBStructureVersion with flexible filters (method, energy, etc)"""
    with get_session() as session:
        query = session.query(DBStructureVersion)
        for attr, value in filters.items():
            query = query.filter(getattr(DBStructureVersion, attr) == value)
        return query.all()

####################################

def query_by_columns(table_class, filters):
    """
    Query a given table for rows matching all column-value pairs in `filters`.
    Args:
        table_name (str): Name of the table
        filters (dict): Dictionary of column names and their expected values
    Returns:
        list of rows
    """
    table_name = table_class.__tablename__
    where_clauses = [f"{col} = :{col}" for col in filters]
    query_str = f"""
        SELECT *
        FROM {table_name}
        WHERE {" AND ".join(where_clauses)}
    """
    query = text(query_str)
    with get_session() as session:
        result = session.execute(query, filters).fetchall()
    return result

####################################

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

####################################

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
