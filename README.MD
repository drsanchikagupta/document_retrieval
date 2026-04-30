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
4. Write code. Add logging alongside so that its easier to debug. 
5. Issues - install the libraries for doc2txt loader.
6. Install git in the root dir /ask - git init -b main
7. create a .gitignore file to ignore files that need not be committed.
  



/*Information - What hugging face embeddings is using and what should be done to identify what embeddings to use
 Default values for HuggingFaceEmbeddings.model_name were deprecated in LangChain 0.2.16 and will be removed in 0.4.0. Explicitly pass a model_name to the HuggingFaceEmbeddings constructor instead.
  embeddings = HuggingFaceEmbeddings()
17:31:24 - INFO - Use pytorch device_name: mps
17:31:24 - INFO - Load pretrained SentenceTransformer: sentence-transformers/all-mpnet-base-v2
config_sentence_transformers.json:
sentence_bert_config.json
model.safetensors */
