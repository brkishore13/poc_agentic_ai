import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 1. LOAD KEYS (Will look for .env file first, or use system environment)
load_dotenv() 

# 2. DEFINE A TOOL
@tool
def get_current_weather(location: str):
    """Returns the weather for a given city. Use this for weather queries."""
    # This is a mock response; in real use, you'd call a weather API here.
    return f"The weather in {location} is 22°C and sunny."

# 3. INITIALIZE THE BRAIN (OpenAI)
# If you didn't use a .env file, you can also pass the key directly:
# llm = ChatOpenAI(model="gpt-4o", api_key="sk-...")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 4. BUILD THE AGENT
tools = [get_current_weather]
agent = create_react_agent(llm, tools)

# 5. EXECUTE AND PRINT
def run_agent(query: str):
    print(f"User: {query}")
    # The new 2026 'invoke' pattern uses a message list
    result = agent.invoke({"messages": [("user", query)]})
    
    # Extract the last message content (the agent's final answer)
    final_answer = result["messages"][-1].content
    print(f"Agent: {final_answer}")

if __name__ == "__main__":
    run_agent("What is the weather like in Hyderabad?")