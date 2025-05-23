from sqlalchemy.orm import Session
from db import SessionLocal
from langchain.tools import tool
from sqlalchemy import text

db_session: Session = SessionLocal()

def get_user_info_by_name(username: str) -> str:
    """
    Fetch all user info from the database where name matches the given username.
    """
    query = text("SELECT * FROM cred_transactions WHERE name = :username")

    try:
        db = SessionLocal()
        result = db.execute(query, {"username": username})
        rows = result.mappings().all()  # get dict-like rows
        if not rows:
            return "No user found with that name."
        formatted = "\n".join(str(dict(row)) for row in rows)  # convert each row to dict and then str
        return formatted
    except Exception as e:
        return f"Error: {e}"
    finally:
        db.close()
