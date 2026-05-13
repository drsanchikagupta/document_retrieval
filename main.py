import os
import time
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA

# Import Langfuse's native callback handler for LangChain
from langfuse.callback import CallbackHandler

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
# Target append-only evaluation log file for Ragas
RAGAS_EVAL_FILE = "ragas_eval_data.jsonl"

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
    # Ensure all remaining background tracking logs are pushed out on shutdown
    if "langfuse_handler" in state:
        logger.info("Flushing background monitoring traces...")
        state["langfuse_handler"].flush() [1.21]
    state.clear()

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    embeddings = state.get("embeddings")
    if not embeddings:
        raise HTTPException(status_code=503, detail="Embeddings engine unavailable.")

    file_ext = os.path.splitext(file.filename).lower()
    if file_ext not in [".docx", ".pdf", ".txt", ".md"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

    import tempfile
    import shutil
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        chunks = load_and_split_documents(temp_file_path, file_ext)
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to parse document text.")

        if state.get("vector_store") is not None:
            logger.info("Adding chunks to existing local vector index...")
            state["vector_store"].add_documents(chunks)
        else:
            state["vector_store"] = create_vector_store(chunks, embeddings)

        state["vector_store"].save_local(INDEX_PATH)
        logger.info(f"Vector store created and saved successfully.")

        state["qa_chain"] = RetrievalQA.from_chain_type(
            llm=state["llm"],
            chain_type="stuff",
            retriever=state["vector_store"].as_retriever(search_kwargs={"k": 3})
        )

        return {"status": "success", "message": f"Successfully indexed {file.filename}"}

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/ask")
async def ask_question(request: QueryRequest):
    if not state.get("qa_chain") or not state.get("vector_store"):
        raise HTTPException(status_code=400, detail="No documents indexed yet.")
    
    try:
        start_total = time.time()
        query = request.question
        
        # 1. Initialize Langfuse dynamic callbacks on-demand per request
        langfuse_handler = CallbackHandler()
        
        # 2. Measure Retrieval Latency & extract context chunks
        start_retrieval = time.time()
        logger.info(f"Retrieving relevant chunks for query: '{query}'")
        relevant_chunks = state["vector_store"].similarity_search(query, k=3)
        retrieval_latency = round(time.time() - start_retrieval, 3)
        
        # Format the retrieved texts for the Ragas pipeline requirement
        contexts = [chunk.page_content for chunk in relevant_chunks]
        
        # 3. Measure Generation Latency passing the Langfuse callback directly to the invocation loop
        start_generation = time.time()
        response = state["qa_chain"].invoke(
            query, 
            config={"callbacks": [langfuse_handler]} # Hooks everything into your Langfuse UI automatically
        )
        generation_latency = round(time.time() - start_generation, 3)
        
        total_latency = round(time.time() - start_total, 3)
        answer = response["result"]

        # 4. Construct dataset record strictly matching the evaluation format for Ragas
        eval_record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "question": query,          # Maps to Ragas 'question'
            "answer": answer,            # Maps to Ragas 'answer'
            "contexts": contexts,        # Maps to Ragas 'contexts'
            "metrics": {
                "retrieval_latency_seconds": retrieval_latency,
                "generation_latency_seconds": generation_latency,
                "total_latency_seconds": total_latency
            }
        }

        # 5. Append-only file writer ('a' flag ensures data is preserved and not overwritten)
        with open(RAGAS_EVAL_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(eval_record) + "\n")

        logger.info(f"Log appended to {RAGAS_EVAL_FILE}. Total latency: {total_latency}s")
        
        return {
            "question": query,
            "answer": answer,
            "metrics": eval_record["metrics"]
        }
    except Exception as e:
        logger.error(f"Error handling query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")

