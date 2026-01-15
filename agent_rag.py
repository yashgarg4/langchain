import os
import certifi

# Fix for SSL Certificate Verify Failed error
os.environ["SSL_CERT_FILE"] = certifi.where()

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- STEP 0: Setup ---
load_dotenv()

# --- STEP 1: The Data (PDF) ---
# We need to index the PDF so the agent can search it.
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(current_dir, "sample.pdf")

loader = PyPDFLoader(pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings,
    collection_name="agent_rag_collection"
)
retriever = vectorstore.as_retriever()

# --- STEP 2: Create Tools ---

# 1. The Retriever Tool (for the PDF)
# The description is CRITICAL. It tells the Agent WHAT is in the PDF.
retriever_tool = create_retriever_tool(
    retriever,
    "pdf_search",
    "Search for information about SIT, Prod Beta, and Prod environments. Use this tool for questions about the document."
)

# 2. The Web Search Tool (for everything else)
search_tool = TavilySearchResults()

tools = [search_tool, retriever_tool]

# --- STEP 3: The Model ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# --- STEP 4: The Agent ---
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. You have access to a PDF document and the web."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- STEP 5: Run It ---
print("Agent is ready! Asking about the PDF...")
question = "What is the current stock price of google?"
response = agent_executor.invoke({"input": question})
print(f"\nAnswer: {response['output']}")

# Cleanup
vectorstore.delete_collection()
