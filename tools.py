from langchain.tools import tool
from prompt_to_sql import generate_sql
from sql_executor import execute_query

@tool
def sql_generator(prompt: str) -> str:
    """Generate SQL query from natural language prompt."""
    schema_hint = """
    Tables:
    - public.person(id, full_name, joining_date, entity_id)
    - public.entity(id, name, region)
    """
    return generate_sql(prompt, schema_hint)

@tool
def sql_executor(sql: str) -> str:
    """Execute SQL query and return results."""
    df = execute_query(sql)
    return df.to_string(index=False) if not df.empty else "No data found."
