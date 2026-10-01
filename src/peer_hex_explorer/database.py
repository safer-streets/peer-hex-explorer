import duckdb

# Every dataset lives here; there is deliberately no local-data switch (see data.py)
SOURCE = "az://phase2"


def _load_extensions(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    INSTALL spatial;LOAD spatial;
    """)


def _enable_azure(con: duckdb.DuckDBPyConnection, connstr: str) -> None:
    con.execute("""
    INSTALL AZURE;
    LOAD AZURE;
    -- force the curl transport, otherwise the default transport fails to locate the SSL CA bundle
    -- GLOBAL scope: cursors created from this connection inherit global settings but NOT session-level SETs,
    -- and each Streamlit script run queries through its own cursor
    SET GLOBAL azure_transport_option_type = 'curl';
    """)
    con.execute("SET GLOBAL azure_storage_connection_string = ?", (connstr,))


def duckdb_connector(azure_connstr: str) -> duckdb.DuckDBPyConnection:
    """
    Connect to an in-memory DuckDB database that can read the Azure parquet store.

    Args:
        azure_connstr: Azure storage connection string.

    Returns:
        A DuckDB connection with the spatial and azure extensions loaded.

    Raises:
        Exception: If extension loading fails.
    """
    con = duckdb.connect(":memory:", read_only=False)
    try:
        _load_extensions(con)
        _enable_azure(con, azure_connstr)
        return con
    except Exception:
        con.close()
        raise
