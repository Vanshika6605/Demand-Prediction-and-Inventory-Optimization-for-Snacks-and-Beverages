from database_setup import get_connection, verify_login as db_verify_login

def verify_login(username, password) -> dict | None:
    """
    Checks user credentials against the SQLite database.
    Returns user record (dict) if valid, else None.
    """
    conn = get_connection()
    try:
        user = db_verify_login(conn, username, password)
        return user
    except Exception:
        return None
    finally:
        conn.close()
