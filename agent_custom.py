import os
import certifi

# Fix for SSL Certificate Verify Failed error
os.environ["SSL_CERT_FILE"] = certifi.where()

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

load_dotenv()

# --- STEP 1: Define Custom Tools ---
# The @tool decorator turns a Python function into a LangChain Tool.
# The docstring is CRITICAL: the LLM reads it to know WHEN to use the tool.

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

@tool
def calculate_circle_area(radius: float) -> float:
    """Calculate the area of a circle given its radius."""
    return 3.14159 * (radius ** 2)

# Initialize the search tool
search = TavilySearchResults()

# List of tools the agent can use (Web Search + Custom Math/String logic)
tools = [search, get_word_length, calculate_circle_area]

# --- STEP 2: The Model ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# --- STEP 3: The Prompt ---
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. You can use tools to answer questions."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# --- STEP 4: Create the Agent ---
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- STEP 5: Run It ---
# This question requires searching (for the radius) and then calculating.
# The Agent should:
# 1. Search Tavily for Mars radius.
# 2. Extract the number.
# 3. Call 'calculate_circle_area' with that number.
question = "Search for the radius of the planet Mars in km, then calculate the area of a circle with that radius."
print(f"Question: {question}")

response = agent_executor.invoke({"input": question})

print(f"\nAnswer: {response['output']}")

