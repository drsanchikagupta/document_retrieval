import os
import shutil
import logging
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA

#-----logging configuration-----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global container to keep loaded models and index in memory across API calls
state = {}
INDEX_PATH = "faiss_index"

def load_and_split_documents(file_path, file_extension, chunk_size=500, chunk_overlap=100):
    logger.info(f"Loading document from: {file_path} with extension: {file_extension}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []
    
    # Select loader dynamically based on file type extension
    if file_extension == ".docx":
        loader = Docx2txtLoader(file_path)
    elif file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        logger.error(f"Unsupported file extension: {file_extension}")
        return []

    documents = loader.load()
    logger.info(f"Document loaded successfully. Number of documents: {len(documents)}")
    # Split the document into chunks. We are using fixed sized chunking for now.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Document split into chunks. Number of chunks: {len(chunks)}")
    return chunks

def create_vector_store(chunks, embeddings):
    # Create a vector store from the chunks
    logger.info("Creating vector store from chunks...")
    # FAISS is an in memory vector store so it doesnot require additional storage
    vector_store = FAISS.from_documents(chunks, embeddings)
    logger.info(f"Vector store created successfully.")
    return vector_store

# FastAPI Lifespan loads the static models once on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing baseline RAG models...")
    
    # Define embeddings here so they are available for both branches
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    state["embeddings"] = embeddings

    # Initialize Ollama LLM
    llm = ChatOllama(model="llama3")

    # Check if a previous vector store can skip processing and load from disk
    if os.path.exists(INDEX_PATH):
        logger.info(f"Loading existing index from {INDEX_PATH}...")
        vector_store = FAISS.load_local(
            INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        state["vector_store"] = vector_store
        
        # Create a Retrieval Chain
        state["qa_chain"] = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # "stuff" means "stuff all found chunks into the prompt"
            retriever=vector_store.as_retriever(search_kwargs={"k": 3})
        )
    else:
        logger.info("No prior index found. System starting without loaded documents.")
        state["vector_store"] = None
        state["qa_chain"] = None

    state["llm"] = llm
    yield
    state.clear()

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    embeddings = state.get("embeddings")
    if not embeddings:
        raise HTTPException(status_code=503, detail="Embeddings engine unavailable.")

    # Extract target file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".docx", ".pdf", ".txt", ".md"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

    # Write binary uploaded data out to a managed temporary file block
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        # Load and split the document using the dynamic pipeline
        chunks = load_and_split_documents(temp_file_path, file_ext)
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to parse document text or no text found.")

        # If a store already exists, add chunks directly. Otherwise, build it fresh.
        if state.get("vector_store") is not None:
            logger.info("Adding chunks to existing local vector index...")
            state["vector_store"].add_documents(chunks)
        else:
            state["vector_store"] = create_vector_store(chunks, embeddings)

        # Save the vector store for future use
        state["vector_store"].save_local(INDEX_PATH)
        logger.info(f"Vector store created and saved successfully.")

        # Update or create the running Retrieval Chain instance with the newest data
        state["qa_chain"] = RetrievalQA.from_chain_type(
            llm=state["llm"],
            chain_type="stuff",
            retriever=state["vector_store"].as_retriever(search_kwargs={"k": 3})
        )

        return {"status": "success", "message": f"Successfully parsed and indexed {file.filename}"}

    finally:
        # Clean up temporary disk space immediately 
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/ask")
async def ask_question(request: QueryRequest):
    if not state.get("qa_chain"):
        raise HTTPException(status_code=400, detail="No documents have been indexed yet. Please upload a file first.")
    
    try:
        query = request.question
        logger.info(f"Retrieving relevant chunks for query: '{query}'")
        
        # Generate the answer using RAG
        response = state["qa_chain"].invoke(query)
        
        return {
            "question": query,
            "answer": response["result"]
        }
    except Exception as e:
        logger.error(f"Error handling query execution: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during answer generation.")
