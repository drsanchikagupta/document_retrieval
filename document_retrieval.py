from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
import os

#-----logging configuration-----
import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_and_split_documents(file_path, chunk_size=500, chunk_overlap=100):
    logger.info(f"Loading document from: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []
    # Load the document
    loader = Docx2txtLoader(file_path)
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



def main():
    # Path to the document file_path
    file_path = 'docs/System Design Interview prep.docx'
    index_path = "faiss_index"

    # Define embeddings here so they are available for both branches
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Check if we can skip processing and load from disk
    if os.path.exists(index_path):
        logger.info(f"Loading existing index from {index_path}...")
        vector_store = FAISS.load_local(
            index_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        # Load and split the document
        chunks = load_and_split_documents(file_path)
        if not chunks:
            logger.error("No chunks created. Exiting.")
            return

        # Create a vector store from the chunks
        vector_store = create_vector_store(chunks, embeddings)
        vector_store.save_local(index_path)
        logger.info(f"Vector store created and saved successfully.")

    # Now you can use vector_store for retrieval tasks
    # Initialize Ollama LLM
    llm = ChatOllama(model="llama3")

    # Create a Retrieval Chain
    # This automatically fetches chunks and formats them into a prompt for the LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" means "stuff all found chunks into the prompt"
        retriever=vector_store.as_retriever(search_kwargs={"k": 3})
    )

    # For example, to retrieve relevant chunks based on a query:
    query = "What are the issues with agent-based systems?"
    logger.info(f"Retrieving relevant chunks for query: '{query}'")
    
    # Generate the answer using RAG
    response = qa_chain.invoke(query)

    print("\n--- ANSWER ---")
    print(response["result"])

if __name__ == "__main__":
    main()