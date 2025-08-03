import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def generate_sql(prompt: str, schema_hint: str = ""):
    system_prompt = f"You are a helpful assistant that converts natural language to SQL. Your SQL is run against a database. This database is that of the company Arbisoft's Workstream product which is an ERP they use. The database schema is:\n{schema_hint}. You should only return the SQL query as code, no other text or backticks or headings or quotation marks. just the SQl query only. so don't start your response with ```sql. Also, use from the schema what is relevant to the query only. That means if someone asks for headcount, you dont need to filter based on status."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0
    )

    return response.choices[0].message.content

def summarize_result_node(state: Dict) -> dict:
    prompt = state["user_prompt"]
    sql = state["generated_sql"]
    result = state["query_result"]

    system_prompt = """
    You are a data analyst assistant. Your job is to present SQL query results in a clear, human-readable way.
    
    Rules:
    - If the result has only one or two rows, respond with a short natural sentence.
    - If the result is tabular with multiple rows, respond with a clean text table (monospace style). An example of a table could be this:
    ┌────────────────────┬─────────────────────┐
    │ status_name        │ number_of_people    │
    ├────────────────────┼─────────────────────┤
    │ Active             │ 467                 │
    │ Probation          │ 53                  │
    │ Intern             │ 9                   │
    │ Terminated         │ 0                   │
    └────────────────────┴─────────────────────┘
    - Use heading names in normal english, not as code variables. This means using 'Number of People' instead of 'number_of_people'.
    - Don't be verbose. Be to-the-point.
    - Do not include the SQL query.
    - Don't rephrase the user's original prompt.
    - Always use consistent alignment and spacing if you format as a table.
    - If the result is empty, say "No results found."
    
    Respond in plain text.
    """

    user_message = f"""User Prompt: "{prompt}"\n\nSQL Query Result:\n{result}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )

    summary = response.choices[0].message.content.strip()
    return {"summary_output": summary}

def suggest_and_generate_chart_node(state: Dict) -> dict:
    result_str = state["query_result"]
    prompt = state["user_prompt"]

    # Short-circuit if "No data found."
    if result_str.strip().lower() == "no data found.":
        return {"chart_base64": ""}

    # Try to parse back into DataFrame (you may store raw df instead for safety)
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(result_str), sep=r"\s{2,}|\t", engine='python')
    except Exception as e:
        print("Failed to parse result into DataFrame:", e)
        return {"chart_base64": ""}

    # Ask GPT if a chart should be shown and how
    col_info = ", ".join([f"{col} ({str(dtype)})" for col, dtype in df.dtypes.items()])

    viz_prompt = f"""
    You are a data visualization assistant.
    
    Given this user prompt: "{prompt}"
    And this DataFrame schema: {col_info}
    
    If this data can be meaningfully visualized, respond in JSON format, for example like this:
    {{
      "visualize": true,
      "chart_type": "bar",
      "x": "Status Name",
      "y": "Number of People",
      "title": "People by Status"
    }}
    
    Otherwise, respond:
    {{ "visualize": false }}
    
    You have to choose whatever chart or graph type works best in regards to the given prompt. The given JSONs are examples. You have to keep the keys same but of course, the values may be different.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[{"role": "user", "content": viz_prompt}]
    )

    import json
    try:
        config = json.loads(response.choices[0].message.content)
    except Exception as e:
        print("Invalid visualization config:", e)
        return {"chart_base64": ""}

    if not config.get("visualize"):
        return {"chart_base64": ""}

    # Generate chart
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = config["x"]
        y = config["y"]
        chart_type = config["chart_type"]

        if chart_type == "bar":
            ax.bar(df[x], df[y])
        elif chart_type == "line":
            ax.plot(df[x], df[y])
        elif chart_type == "pie":
            ax.pie(df[y], labels=df[x], autopct="%1.1f%%")
        else:
            return {"chart_base64": ""}

        ax.set_title(config["title"])
        if chart_type != "pie":
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        plt.xticks(rotation=45)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        return {"chart_base64": f"data:image/png;base64,{img_b64}"}

    except Exception as e:
        print("Failed to generate chart:", e)
        return {"chart_base64": ""}
