"""Config settings for the RAG app"""

import os 
from dotenv import load_dotenv

#load env variables
load_dotenv()

#llm api key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#models
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

#chromadb settings
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"

#data path
DATA_PATH = "./data"