import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

engine = create_engine(DB_URL)

def execute_query(sql: str) -> pd.DataFrame:
    """
    Execute the given SQL string and return the result as a Pandas DataFrame.
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text(sql))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
    except Exception as e:
        print(f"[ERROR] SQL execution failed: {e}")
        return pd.DataFrame()
