import os
import certifi
import streamlit as st

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
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

load_dotenv()

st.set_page_config(page_title="Gemini RAG Agent", page_icon="🤖")
st.title("🤖 Gemini RAG Agent with Memory")

# --- STEP 1: Cache Data Loading ---
# We use @st.cache_resource so we don't reload the PDF on every interaction
@st.cache_resource
def get_retriever():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "sample.pdf")
    
    if not os.path.exists(pdf_path):
        st.error("sample.pdf not found! Please add it to the project directory.")
        return None

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Use a unique collection name for the UI to avoid conflicts
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        collection_name="streamlit_collection"
    )
    return vectorstore.as_retriever()

retriever = get_retriever()

#STEP 2: Cache Agent Setup
@st.cache_resource
def get_agent_executor(_retriever):
    # 1. Create Tools
    retriever_tool = create_retriever_tool(
        _retriever,
        "pdf_search",
        "Search for information about SIT, Prod Beta, and Prod environments. Use this tool for questions about the document."
    )
    search_tool = TavilySearchResults()
    tools = [search_tool, retriever_tool]

    # 2. Create Model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    # 3. Create Agent
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. You have access to a PDF document and the web."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

if retriever:
    agent_executor = get_agent_executor(retriever)

    # --- STEP 3: Manage History with Streamlit ---
    # StreamlitChatMessageHistory stores messages in st.session_state.langchain_messages
    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    if len(msgs.messages) == 0:
        msgs.add_ai_message("Hello! I can answer questions about your PDF or search the web. What's on your mind?")

    # Helper to sanitize Gemini output
    def sanitize_gemini_output(result):
        if isinstance(result.get("output"), list):
            text_content = "".join([part.get("text", "") for part in result["output"] if isinstance(part, dict)])
            result["output"] = text_content
        return result

    agent_with_sanitization = agent_executor | RunnableLambda(sanitize_gemini_output)

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_with_sanitization,
        lambda session_id: msgs, # Always return the Streamlit history
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output",
    )

    # --- STEP 4: Render UI ---
    # Display previous messages
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # Handle new user input
    if prompt_text := st.chat_input():
        st.chat_message("human").write(prompt_text)

        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                response = agent_with_chat_history.invoke(
                    {"input": prompt_text},
                    config={"configurable": {"session_id": "any"}}
                )
                st.write(response["output"])



