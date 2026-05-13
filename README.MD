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
11. RAG is defined by three core stages, all of which are present in your script:Retrieval: Your vector_store.similarity_search(query) takes your question, turns it into a vector, and finds the most relevant text chunks from your local FAISS database.Augmentation: The RetrievalQA chain takes those retrieved chunks and "stuffs" them into a prompt as additional context for the AI.Generation: Ollama (Llama 3) uses that specific context to generate a factual answer, rather than just guessing from its own internal training data.
12. By using FAISS (vector store), HuggingFace (local embeddings), and Ollama (local LLM), you have built a fully local RAG pipeline.

ToDo:
- Add Fastapi - add capability to load the model once and use thereafter.
- Add docker image
- Check metrics tools
- Rag and agents
- Streamlit for UI



/*Information - What hugging face embeddings is using and what should be done to identify what embeddings to use
 Default values for HuggingFaceEmbeddings.model_name were deprecated in LangChain 0.2.16 and will be removed in 0.4.0. Explicitly pass a model_name to the HuggingFaceEmbeddings constructor instead.
  embeddings = HuggingFaceEmbeddings()
17:31:24 - INFO - Use pytorch device_name: mps
17:31:24 - INFO - Load pretrained SentenceTransformer: sentence-transformers/all-mpnet-base-v2
config_sentence_transformers.json:
sentence_bert_config.json
model.safetensors */
