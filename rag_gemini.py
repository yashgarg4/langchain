import os
import getpass

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- STEP 0: Setup API Key ---
# If you haven't set this in your environment variables, this will prompt you.
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    print(f"DEBUG: Using API Key: {api_key[:5]}...{api_key[-5:]}")

# --- STEP 1: The Data ---
# This simulates loading a document.
raw_text = """
Gemini is a family of multimodal AI models developed by Google DeepMind. 
It was announced on December 6, 2023. Gemini is designed to be multimodal from the ground up, 
meaning it can generalize and seamlessly understand, operate across, and combine different types of information 
including text, code, audio, image, and video. The first version, Gemini 1.0, came in three sizes: 
Ultra, Pro, and Nano.
"""

# --- STEP 2: Chunking and Indexing ---
# Split the text into smaller chunks for the model to digest.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = text_splitter.create_documents([raw_text])

# Create Embeddings using Google's model
# We use "models/embedding-001" which is optimized for text retrieval.
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Store the chunks in a local vector database (Chroma)
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

# --- STEP 3: The Prompt ---
# Standard RAG prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# --- STEP 4: The Model ---
# We instantiate the Gemini model here.
# 'gemini-1.5-flash' is fast and cost-effective for this type of task.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# --- STEP 5: The Chain (LCEL) ---
# 1. Retrieve docs -> 2. Pass question -> 3. Format prompt -> 4. Gemini generates -> 5. Parse string
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- STEP 6: Execution ---
question = "What are the three sizes of the first version of Gemini?"
print(f"Question: {question}")

# Invoke the chain
response = rag_chain.invoke(question)
print(f"\nAnswer: {response}")

# Cleanup (Optional)
# vectorstore.delete_collection()
