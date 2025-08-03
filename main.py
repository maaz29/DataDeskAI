from fastapi import FastAPI
from pydantic import BaseModel
from graph import agent_graph
from fastapi.responses import HTMLResponse

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/query")
async def query_agent(request: PromptRequest):
    state = agent_graph.invoke({"user_prompt": request.prompt})
    return {
        "response": state["final_result"],
        "chart_base64": state.get("chart_base64", "")
    }

@app.get("/", response_class=HTMLResponse)
async def landing_page():
    with open("frontend/landing.html", "r") as file:
        return file.read()
