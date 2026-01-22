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
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

load_dotenv()

# --- STEP 1: The Data (PDF) ---
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
    collection_name="agent_memory_collection" # Unique collection name
)
retriever = vectorstore.as_retriever()

# --- STEP 2: Create Tools
retriever_tool = create_retriever_tool(
    retriever,
    "pdf_search",
    "Search for information about SIT, Prod Beta, and Prod environments. Use this tool for questions about the document."
)

search_tool = TavilySearchResults()

tools = [search_tool, retriever_tool]

# --- STEP 3: The Model ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# --- STEP 4: The Agent ---
# We add "chat_history" to the prompt so the agent can see previous messages.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. You have access to a PDF document and the web."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Helper: Sanitize Output ---
# Gemini sometimes returns a list of content parts. We need to convert this to a string
# before saving it to history, otherwise the next turn will crash.
def sanitize_gemini_output(result):
    if isinstance(result.get("output"), list):
        text_content = "".join([part.get("text", "") for part in result["output"] if isinstance(part, dict)])
        result["output"] = text_content
    return result

agent_with_sanitization = agent_executor | RunnableLambda(sanitize_gemini_output)

# --- STEP 5: Add Memory ---
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_chat_history = RunnableWithMessageHistory(
    agent_with_sanitization,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output",
)

# --- STEP 6: Chat Loop ---
print("\nAgent with Memory is ready! Type 'exit' to stop.")
print("Try asking: 'What is SIT?' then 'Who is responsible for it?'")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break
    
    response = agent_with_chat_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "my_session"}}
    )
    
    print(f"AI: {response['output']}")

# Cleanup
vectorstore.delete_collection()

