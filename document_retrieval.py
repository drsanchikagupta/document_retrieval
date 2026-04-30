from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
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

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Document split into chunks. Number of chunks: {len(chunks)}")

    return chunks

def create_vector_store(chunks):
    # Create embeddings for the chunks
    logger.info("Creating vector store from chunks...")
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    logger.info(f"Vector store created successfully.")

    return vector_store

def main():
    # Path to the document
    file_path = 'docs/System Design Interview prep.docx'

    # Load and split the document
    chunks = load_and_split_documents(file_path)
    if not chunks:
        logger.error("No chunks created. Exiting.")
        return

    # Create a vector store from the chunks
    vector_store = create_vector_store(chunks)

    # Now you can use vector_store for retrieval tasks
    # For example, to retrieve relevant chunks based on a query:
    query = "What are the issues with agent-based systems?"
    logger.info(f"Retrieving relevant chunks for query: '{query}'")
    relevant_chunks = vector_store.similarity_search(query)

    for i, chunk in enumerate(relevant_chunks):
        logger.info(f"Chunk {i+1}: {chunk.page_content[:200]}...")  # Print the first 200 characters of each relevant chunk
    print(relevant_chunks)

if __name__ == "__main__":    main()
