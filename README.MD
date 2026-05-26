## Local RAG Pipeline with FastAPI, FAISS, & Ollama
This project implements a fully local, privacy-centric Retrieval-Augmented Generation (RAG) pipeline. By combining FastAPI, FAISS, HuggingFace Embeddings, and Ollama (llama3), it processes documents and answers semantic queries entirely on your machine with zero cloud dependencies or API subscription fees.
------------------------------
## 🛠️ Architecture Stack Comparison

| Component | 🟢 Local Stack (Recommended) | ☁️ Cloud "Free Tier" Stack |
|---|---|---|
| Vector DB | ChromaDB / FAISS (Local files) | Pinecone (Starter Tier) |
| LLM | Ollama (llama3 / mistral) | Groq (Free API) |
| Embeddings | HuggingFace (all-mpnet-base-v2) | HuggingFace (Inference API) |
| Pros | Unlimited usage, private, works offline. | No hardware requirements. |
| Cons | Consumes local RAM/GPU resources. | Stringent rate limits; paywalls. |

------------------------------
## 📐 Design & System Decisions## 1. Fully Local RAG Workflow
RAG grounds the LLM in your private knowledge base, eliminating hallucinations through three core stages:

* Retrieval: vector_store.similarity_search(query) converts your question into a vector and pulls the top matching text chunks from the local FAISS database.
* Augmentation: The RetrievalQA chain injects these retrieved chunks into a structured prompt context block.
* Generation: Local Ollama (llama3) synthesizes a factual response utilizing only the context provided.

## 2. High-Performance Vector Storage
This stack leverages FAISS, an in-memory, lightweight vector similarity library. It processes indexing tasks seamlessly without a standalone database server application.

💡 Design Note: Keep using FAISS for low-overhead local systems. Only migrate to OpenSearch or Elasticsearch if your architecture transitions into a distributed, multi-user production scale environment.

## 3. State Management & Lifecycle Hooks
To maximize speed, we use FastAPI's lifespan feature alongside a global state = {} container to load model weight configurations exactly once:

* On Startup (lifespan): Heavy models parse once and populate state["embeddings"], state["vector_store"], and state["qa_chain"] in your RAM.
* During API Calls (/ask): Fetches the pre-warmed models from RAM instantly, dropping latency to milliseconds.
* During Uploads (/upload or /upload-folder): Dynamically appends files via add_documents() to your local faiss_index file tree without a system reboot.

Note: The standard Python entrypoint if __name__ == "__main__": main() has been removed. Web servers like Uvicorn serve as the exclusive interface entry point for FastAPI.
## 4. Embedding Selection
Explicitly declare your embedding model target name inside HuggingFaceEmbeddings to avoid deprecation exceptions.

* sentence-transformers/all-MiniLM-L6-v2: Fastest option, low hardware footprint.
* sentence-transformers/all-mpnet-base-v2: Industry standard for high semantic accuracy (uses Apple Silicon mps or NVIDIA cuda hardware acceleration when available).

------------------------------
## 📊 Performance Metrics & Observability## Latency Profiles
The API utilizes Python's native time module to track three diagnostic dimensions inside your server terminal logs and output JSON objects:

   1. Retrieval Latency (retrieval_latency_seconds): Tracks vector query generation and FAISS similarity lookup speeds. High counts pinpoint CPU bottlenecks or bloated indices.
   2. Generation Latency (generation_latency_seconds): Measures local Ollama response synthesis time. This remains your main computation bottleneck.
   3. Total Latency (total_latency_seconds): Captures complete round-trip API processing times.

## Accuracy Testing: Ragas vs. Langfuse

* Ragas: A free open-source framework used as a standalone, offline script to grade Faithfulness (hallucination checks), Answer Relevancy, and Context Precision. Do not run it inline with endpoints as it uses a secondary judge LLM that doubles runtime latency.
* Langfuse: A free, self-hosted, MIT-licensed observability tool used to monitor execution traces, prompts, and performance intervals in real-time.

------------------------------
## 🚀 Getting Started## 1. Project Initialization & Dependencies
Navigate to the root directory /ask, configure your virtual environment, and install dependencies:

# Setup Git
git init -b main
# Configure Virtual Environment
python3 -m venv venv
source venv/bin/activate
# Install Core Requirements
pip install fastapi uvicorn pydantic python-multipart langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu pymupdf docx2txt langfuse ragas datasets python-dotenv

Create a .gitignore file to ensure security and cache isolation:

venv/
__pycache__/
faiss_index/
.env
ragas_eval_data.jsonl

## 2. Configure Environment Variables
Create a local .env configuration file to store your credentials:

LANGFUSE_PUBLIC_KEY="pk-lf-your-public-key"
LANGFUSE_SECRET_KEY="sk-lf-your-secret-key"
LANGFUSE_HOST="https://langfuse.com"

## 3. Launching the Web Server
Launch the server using Uvicorn:

uvicorn main:app --reload

------------------------------
## 📡 API Endpoint Testing
Open a secondary terminal window or execute queries using the built-in browser interface.
## Method A: Terminal (Curl Commands)
Step 1: Upload and Index Documents

curl -X POST "127.0.0.1:8000/upload" -F "file=@docs/System Design Interview prep.docx"

Result: Automatically extracts document chunks, commits the vector matrix down to faiss_index/, and warms up your QA pipeline.
Step 2: Execute Semantic Context Queries

curl -X POST "127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the issues with agent-based systems?"}'

## Method B: Graphical Interface (Browser)

   1. Navigate to: http://127.0.0.1:8000/docs to open the interactive Swagger UI Dashboard.
   2. Click the green POST /upload header block $\rightarrow$ Try it out $\rightarrow$ select your document $\rightarrow$ Execute.
   3. Once indexed, scroll to POST /ask $\rightarrow$ Try it out $\rightarrow$ type your question string $\rightarrow$ Execute to view outputs and telemetry metadata instantly.

------------------------------
## 📝 Future Implementation Roadmap (ToDo)

* Compile application assets into an isolated Docker Image.
* Introduce Streamlit to deliver a consumer-grade chatbot UI layout.
* Establish rigorous agentic loops, Guardrails, and security layers to stop prompt injection and jailbreaks.
* Enhance formatting logic to output Source Citations or relevant fallback errors if data is absent.
* Interface with live web scrapers to deliver hybrid internet search fallbacks.
* Support specialized memory layers splitting Short-Term Context vs. Long-Term Context.

------------------------------
Would you like help mapping out any of the advanced features from the ToDo roadmap, such as implementing a Streamlit User Interface or setting up Prompt Injection Guardrails?

