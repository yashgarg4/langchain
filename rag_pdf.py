import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- STEP 0: Setup ---
load_dotenv()

# --- STEP 1: The Data (PDF Load) ---
# Ensure you have a file named 'sample.pdf' in the same directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(current_dir, "sample.pdf")

if not os.path.exists(pdf_path):
    print(f"Error: File not found at {pdf_path}")
    print("Please add a 'sample.pdf' file to run this step.")
    exit()

print("Loading PDF...")
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# --- STEP 2: Chunking and Indexing ---
print("Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print("Creating embeddings and vector store...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# We use a temporary collection name to avoid mixing with previous data
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings,
    collection_name="pdf_collection"
)
retriever = vectorstore.as_retriever()

# --- STEP 3: The Prompt ---
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# --- STEP 4: The Model ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# --- STEP 5: The Chain ---
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- STEP 6: Execution ---
question = "Summarize the main topic of this document."
print(f"\nQuestion: {question}")

response = rag_chain.invoke(question)
print(f"Answer: {response}")

# Cleanup
vectorstore.delete_collection()