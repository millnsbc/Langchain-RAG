"""Config settings for the RAG app"""

import os 
from pathlib import Path
from dotenv import load_dotenv

#get absolute path to project root
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"

#load env variables
loaded = load_dotenv(env_path)
if not loaded:
    print(f"Warning: .env file not found at {env_path}")

#llm api key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Check your .env file.")

#models
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

#chromadb settings
CHROMA_PATH = str(project_root / "chroma_db")
COLLECTION_NAME = "documents"

#data path
DATA_PATH = str(project_root / "data")

#retrieval settings
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60