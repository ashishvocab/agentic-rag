from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

class DAO:
    def __init__(self, db: Session):
        self.db = db

    def execute_query(self, query: str, params: dict = None):
        try:
            result = self.db.execute(query, params or {})
            return result.fetchall()
        except SQLAlchemyError as e:
            print(f"[ERROR] Failed to execute query: {e}")
            return None