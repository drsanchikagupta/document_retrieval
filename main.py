import os
import time
import json
import logging
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

# 1. Load environment variables FIRST before importing any LangChain/Langfuse tools
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA

# Import Langfuse integration (Targeting the modern LangChain namespace)
from langfuse.langchain import CallbackHandler

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
    state.clear()

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str

class FolderRequest(BaseModel):
    folder_path: str

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    embeddings = state.get("embeddings")
    if not embeddings:
        raise HTTPException(status_code=503, detail="Embeddings engine unavailable.")

    uploaded_summary = []
    import tempfile
    import shutil

    for file in files:
        file_ext = os.path.splitext(file.filename).lower()
        if file_ext not in [".docx", ".pdf", ".txt", ".md"]:
            logger.warning(f"Skipping unsupported file type: {file.filename}")
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        try:
            chunks = load_and_split_documents(temp_file_path, file_ext)
            if not chunks:
                continue

            if state.get("vector_store") is not None:
                logger.info(f"Adding chunks from {file.filename} to universal index...")
                state["vector_store"].add_documents(chunks)
            else:
                logger.info(f"Creating baseline universal index with {file.filename}...")
                state["vector_store"] = create_vector_store(chunks, embeddings)

            uploaded_summary.append(file.filename)

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    if state.get("vector_store") is not None:
        state["vector_store"].save_local(INDEX_PATH)
        state["qa_chain"] = RetrievalQA.from_chain_type(
            llm=state["llm"],
            chain_type="stuff",
            retriever=state["vector_store"].as_retriever(search_kwargs={"k": 3})
        )
        return {"status": "success", "message": f"Successfully indexed: {', '.join(uploaded_summary)}"}
    
    raise HTTPException(status_code=400, detail="No valid documents were successfully processed.")

@app.post("/upload-folder")
async def upload_entire_folder(request: FolderRequest):
    embeddings = state.get("embeddings")
    if not embeddings:
        raise HTTPException(status_code=503, detail="Embeddings engine unavailable.")

    if not os.path.exists(request.folder_path):
        raise HTTPException(status_code=404, detail=f"The system directory path does not exist: {request.folder_path}")

    indexed_files = []
    
    # Scan the target folder for matching formats
    for filename in os.listdir(request.folder_path):
        file_path = os.path.join(request.folder_path, filename)
        
        # Skip sub-directories, focus strictly on files
        if os.path.isdir(file_path):
            continue
            
        file_ext = os.path.splitext(filename).lower()
        if file_ext not in [".docx", ".pdf", ".txt", ".md"]:
            continue

        try:
            # Parse documents natively directly from their local directory paths
            chunks = load_and_split_documents(file_path, file_ext)
            if not chunks:
                continue

            if state.get("vector_store") is not None:
                logger.info(f"Adding folder file chunks from {filename} into universal index...")
                state["vector_store"].add_documents(chunks)
            else:
                logger.info(f"Creating baseline universal index with folder file {filename}...")
                state["vector_store"] = create_vector_store(chunks, embeddings)
                
            indexed_files.append(filename)
        except Exception as e:
            logger.error(f"Error indexing file {filename} from folder: {str(e)}")

    if not indexed_files:
        raise HTTPException(status_code=400, detail="No valid documents matching supported formats (.pdf, .docx, .txt, .md) found in folder.")

    # Commit all compiled folder additions to storage once loop resolves
    state["vector_store"].save_local(INDEX_PATH)
    state["qa_chain"] = RetrievalQA.from_chain_type(
        llm=state["llm"],
        chain_type="stuff",
        retriever=state["vector_store"].as_retriever(search_kwargs={"k": 3})
    )

    return {"status": "success", "message": f"Successfully batch indexed {len(indexed_files)} files from folder.", "files": indexed_files}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    if not state.get("qa_chain") or not state.get("vector_store"):
        raise HTTPException(status_code=400, detail="No documents indexed yet.")
    
    try:
        start_total = time.time()
        query = request.question
        
        # Initialize Langfuse dynamic callbacks on-demand per request
        langfuse_handler = CallbackHandler()
        
        # Measure Retrieval Latency & extract context chunks
        start_retrieval = time.time()
        logger.info(f"Retrieving relevant chunks for query: '{query}'")
        relevant_chunks = state["vector_store"].similarity_search(query, k=3)
        retrieval_latency = round(time.time() - start_retrieval, 3)
        
        # Format the retrieved texts for the Ragas pipeline requirement
        contexts = [chunk.page_content for chunk in relevant_chunks]
        
        # Measure Generation Latency passing the Langfuse callback directly to the invocation loop
        start_generation = time.time()
        response = state["qa_chain"].invoke(
            query, 
            config={"callbacks": [langfuse_handler]}
        )
        generation_latency = round(time.time() - start_generation, 3)
        
        total_latency = round(time.time() - start_total, 3)
        answer = response["result"]

        # Construct dataset record strictly matching the evaluation format for Ragas
        eval_record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "metrics": {
                "retrieval_latency_seconds": retrieval_latency,
                "generation_latency_seconds": generation_latency,
                "total_latency_seconds": total_latency
            }
        }

        # Append-only file writer ('a' flag ensures data is preserved and not overwritten)
        with open(RAGAS_EVAL_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(eval_record) + "\n")

        logger.info(f"Log appended to {RAGAS_EVAL_FILE}. Total latency: {total_latency}s")
        
        # Force the underlying client to push traces immediately
        langfuse_handler.client.flush()
        
        return {
            "question": query,
            "answer": answer,
            "metrics": eval_record["metrics"]
        }
    except Exception as e:
        logger.error(f"Error handling query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")
