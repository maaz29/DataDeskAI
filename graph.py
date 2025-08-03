from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from prompt_to_sql import generate_sql, summarize_result_node, suggest_and_generate_chart_node
from sql_executor import execute_query

class InputState(TypedDict):
    user_prompt: str

class OutputState(TypedDict):
    final_result: str
    chart_base64: str

class AgentState(TypedDict):
    user_prompt: str
    generated_sql: str
    query_result: str
    summary_output: str
    chart_base64: str


def sql_generator_node(state: InputState) -> dict:
    prompt = state["user_prompt"]
    schema_hint = """
    Tables:
    - public.person(id, full_name, joining_date, status_id)
    - public.status(id, name)
    """
    sql = generate_sql(prompt, schema_hint)
    return {"generated_sql": sql}

def sql_executor_node(state: AgentState) -> dict:
    sql = state["generated_sql"]
    print("sql: ", sql)
    result = execute_query(sql)
    return {"query_result": result.to_string(index=False) if not result.empty else "No data found."}

def final_output_node(state: AgentState) -> dict:
    return {"final_result": state["summary_output"]}

# Build the state graph
graph_builder = StateGraph(AgentState, input=InputState, output=OutputState)

# Add nodes
graph_builder.add_node("generate_sql", sql_generator_node)
graph_builder.add_node("run_sql", sql_executor_node)
graph_builder.add_node("final_output", final_output_node)
graph_builder.add_node("summarize_result", summarize_result_node)
graph_builder.add_node("generate_chart", suggest_and_generate_chart_node)

# Define edges
graph_builder.add_edge(START, "generate_sql")
graph_builder.add_edge("generate_sql", "run_sql")
graph_builder.add_edge("run_sql", "summarize_result")
graph_builder.add_edge("summarize_result", "generate_chart")
graph_builder.add_edge("generate_chart", "final_output")
graph_builder.add_edge("final_output", END)

# Compile
agent_graph = graph_builder.compile()
