import os
import certifi
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# --- STEP 0: Setup ---
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

# --- STEP 1: Define Tools & Model ---
tools = [TavilySearchResults(max_results=3)]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# We "bind" tools to the model so it knows it has the OPTION to call them
llm_with_tools = llm.bind_tools(tools)

# --- STEP 2: Define State ---
# The "State" is the memory of the graph. 
# 'add_messages' means: when a node returns a message, append it to the list (don't overwrite).
class State(TypedDict):
    messages: Annotated[list, add_messages]

# --- STEP 3: Define Nodes ---
# A Node is just a function that takes the State and returns an update.

def reasoner(state: State):
    # The model looks at the chat history and decides what to do
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# --- STEP 4: Build the Graph ---
builder = StateGraph(State)

# Add Nodes
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools)) # Prebuilt node that executes tools

# Add Edges (The Logic Flow)
builder.add_edge(START, "reasoner")

# Conditional Edge:
# After "reasoner" runs, check the output.
# If the model asked for a tool -> Go to "tools" node.
# If the model gave an answer -> Go to END.
builder.add_conditional_edges(
    "reasoner",
    tools_condition,
)

# If we ran a tool, always go back to the reasoner to interpret the result
builder.add_edge("tools", "reasoner")

# Compile the graph
react_graph = builder.compile()

# --- STEP 5: Run It ---
print("LangGraph Agent is ready! Asking a question...")
question = "Who is the current CEO of Google?"

# Stream the updates from the graph
events = react_graph.stream(
    {"messages": [("user", question)]},
    stream_mode="values"
)

for event in events:
    if "messages" in event:
        last_msg = event["messages"][-1]
        print(f"[{last_msg.type.upper()}]: {last_msg.content}")
