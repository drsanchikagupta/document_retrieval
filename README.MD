Component	Local Stack (Recommended)	Cloud "Free Tier" Stack
Vector DB	ChromaDB (Local files)	Pinecone (Starter Tier)
LLM	Ollama (Llama 3 / Mistral)	Groq (Free API for now)
Embeddings	Ollama / HuggingFace	HuggingFace (Inference API)
Pros	Unlimited usage, private, works offline.	No hardware requirements.
Cons	Uses your RAM/GPU.	Rate limits; may start charging later.


1. Create a venv - python3 -m venv venv
2. Activate it - source venv/bin/activate
3. Install the required dependencies. 
pip install langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu pymupdf
4. FAISS is a local- in memory, lightweight library used for fast similarity searches without needing a server or complex setup. You should only switch to OpenSearch if you need a production-grade, distributed database to handle massive scale or multi-user access.
5. Write code. Add logging alongside so that its easier to debug. 
6. Issues - install the libraries for doc2txt loader.
7. Install git in the root dir /ask - git init -b main
8. create a .gitignore file to ignore files that need not be committed.
9. Update your create_vector_store function to include the specific model you want to use. Common choices include sentence-transformers/all-MiniLM-L6-v2 (fast and light) or sentence-transformers/all-mpnet-base-v2 (higher accuracy)
10. When you ask a query, the vector store - 1. Embeds query, 2. Compares vectors, 3. Returns closest chunks.
11. RAG is defined by three core stages, all of which are present in this script:Retrieval: vector_store.similarity_search(query) takes the question, turns it into a vector, and finds the most relevant text chunks from your local FAISS database.Augmentation: The RetrievalQA chain takes those retrieved chunks and "stuffs" them into a prompt as additional context for the AI. Generation: Ollama (Llama 3) uses that specific context to generate a factual answer, rather than just guessing from its own internal training data.
12. By using FAISS (vector store), HuggingFace (local embeddings), and Ollama (local LLM), we have built a fully local RAG pipeline.
13. Added fastapi support to upload the documents using /upload endpoint and /ask a question. Also added FastAPI Lifespan to load the static models once on startup.
14. Added a state variable to log the state throughout the process. 
  - On Startup (lifespan): The server boots up, reads your heavy models once, and tucks them inside state["embeddings"], state["vector_store"], and state["qa_chain"].
  - During an API Call (/ask): When a user asks a question, the endpoint reaching out to state["qa_chain"] pulls the already-warmed-up model from RAM instantly. This allows your RAG application to respond in milliseconds.
  - During an Upload (/upload): When a new document is sent to the server, the route grabs the database directly out of state["vector_store"], modifies it with the new text vectors, saves it back into the state locker, and updates the chain instantly without restarting the server.
15. Removed if __name__ == "__main__": main(): FastAPI applications use web servers like uvicorn as their entry points rather than executing standard script loops.

ToDo:
- Add docker image
- Check metrics tools
- Rag and agents
- Agents security and guadrails - prompt injection, jailbreaking
- Streamlit for UI

How to run-
- Install fastapi -- requirements.txt
- uvicorn main:app --reload where main.py is the name of the file.
- In another terminal window - curl -X POST "127.0.0.1:8000/upload" -F "file=@docs/System Design Interview prep.docx" - a faiss_index would be saved after this command and RAG model is warmed in RAM to ask questions
- curl -X POST "127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"question": "What are the issues with agent-based systems?"}'
- Run directly on browser - type address http://127.0.0.1:8000/docs
What to do on that page:Click the green POST /upload bar to expand it.Click the "Try it out" button on the right.Click "Choose File", select your document, and click the blue "Execute" button at the bottom.Once it succeeds, scroll down to the POST /ask bar, click "Try it out", type your question into the text box, and click "Execute".


/*Information - What hugging face embeddings is using and what should be done to identify what embeddings to use
 Default values for HuggingFaceEmbeddings.model_name were deprecated in LangChain 0.2.16 and will be removed in 0.4.0. Explicitly pass a model_name to the HuggingFaceEmbeddings constructor instead.
  embeddings = HuggingFaceEmbeddings()
17:31:24 - INFO - Use pytorch device_name: mps
17:31:24 - INFO - Load pretrained SentenceTransformer: sentence-transformers/all-mpnet-base-v2
config_sentence_transformers.json:
sentence_bert_config.json
model.safetensors */
