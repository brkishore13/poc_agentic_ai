import os
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent

# 1. Load your API key from the .env file
load_dotenv()

# 2. (Optional) Suppress the Pydantic V1 / Python 3.14 warning
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

# 3. Setup Wikipedia Tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
tools = [wiki_tool]

# 4. Initialize the LLM (It will now find the key in your environment)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 5. Create the Agent using the 2026 standard (LangGraph)
agent_executor = create_react_agent(llm, tools)

# 6. Run the query
query = "What is the 'Attention is All You Need' paper about?"
print("--- Agent is searching Wikipedia ---")

result = agent_executor.invoke({"messages": [("human", query)]})

# Output the agent's final answer
print("\n--- Final Response ---")
print(result["messages"][-1].content)