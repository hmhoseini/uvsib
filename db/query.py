from sqlalchemy import text
from db.session import get_session

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

def query_by_column(table_class, column_name, value):
    """
    Query a given table for rows where `column_name` == value.
    Args:
        table (str): table class
        column_name (str): column name to filter on
        value: value to match
    Returns:
        list of rows
    """
    table_name = table_class.__tablename__
    query = text(f"""
        SELECT *
        FROM {table_name}
        WHERE {column_name} = :val"""
    )
    with get_session() as session:
        result = session.execute(query, {"val": value}).fetchall()
    return result

def get_rows_with_attribute_value(table_class, attribute):
    """
    Query rows from a table
    """
    table_name = table_class.__tablename__
    query = text(f"""
        SELECT *
        FROM {table_name}
        WHERE attributes->> :attributes = :attribute_value"""
    )
    with get_session() as session:
        result = session.execute(query,
                                 {"attributes": attribute[0],
                                  "attribute_value": attribute[-1]
                                 }
        ).mappings().all()
    return result
