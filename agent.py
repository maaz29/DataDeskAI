from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from tools import sql_generator, sql_executor

llm = ChatOpenAI(temperature=0, model="gpt-4")  # or gpt-3.5-turbo

# Bundle tools
tools = [sql_generator, sql_executor]

# Initialize agent with tools
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)
