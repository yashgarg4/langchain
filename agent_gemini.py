import os
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- STEP 0: Setup ---
load_dotenv()

# STEP 1: Define Tool
# Tools are functions the Agent can use. 
# Tavily is a search engine optimized for LLMs and Agents.
search = TavilySearchResults()

tools = [search]

# --- STEP 2: The Model ---
# We use Gemini, which supports "Tool Calling" natively.
# This means it is trained to output structured data to call functions.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# --- STEP 3: The Prompt ---
# The prompt must include a placeholder for "agent_scratchpad".
# This is where the agent writes down its thoughts and tool outputs.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. You can search the web to answer questions."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# --- STEP 4: Create the Agent ---
# This constructs the logic: LLM + Tools + Prompt
agent = create_tool_calling_agent(llm, tools, prompt)

# AgentExecutor is the runtime that actually calls the agent and executes the tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- STEP 5: Run It ---
question = "What is langchain and how to use tavily search?"
print(f"Question: {question}")

# The agent will decide to use the 'search' tool because it doesn't know the answer.
response = agent_executor.invoke({"input": question})

print(f"\nAnswer: {response['output']}")

