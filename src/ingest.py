"""Document ingestion pipeline for RAG system"""

import os
import logging
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config import (
    DATA_PATH,
    CHROMA_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    OPENAI_API_KEY
)

#set api key for openai
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

#setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_documents():
    """Load documents from data directory"""
    #load pdfs
    pdf_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    pdf_docs = pdf_loader.load()
    
    #load text files (utf-8)
    text_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    text_docs = text_loader.load()
    
    #combine into a list of documents
    all_docs = pdf_docs + text_docs
    logger.info(f"Loaded {len(all_docs)} documents")
    return all_docs

def split_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks

def create_vectorstore(chunks):
    """Create and persist vector store"""
    #initialize embeddings
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    #create chroma db
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME
    )
    
    logger.info(f"Created vector store at {CHROMA_PATH}")
    return vectorstore

def main():
    """Run the ingestion pipeline"""
    logger.info("Starting document ingestion...")
    
    try:
        #load docs
        docs = load_documents()
        if not docs:
            raise ValueError("No documents found in data directory")
        
        #split into chunks
        chunks = split_documents(docs)
        
        #create vector store
        vectorstore = create_vectorstore(chunks)
        
        logger.info("Ingestion complete")
        logger.info(f"Total pages: {len(docs)}")
        logger.info(f"Total chunks: {len(chunks)}")
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise

if __name__ == "__main__":
    main()