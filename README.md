# Langchain RAG

A conversational RAG system for querying academic research papers on bouldering injuries. Built with LangChain, ChromaDB, and Chainlit.

## Setup

**Prerequisites:**
- Python 3.10+
- OpenAI API Key

**Installation:**
```bash
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

echo OPENAI_API_KEY=sk-your-key-here > .env

python src/ingest.py
chainlit run main.py -w
```

Access the UI at `http://localhost:8000`.

## Architecture

**LangChain (LCEL)**: Uses the expression language pattern (`prompt | llm`) for explicit chain composition and easier debugging.

**ChromaDB**: Local vector store with disk persistence. Production deployments could migrate to Pinecone or Weaviate for horizontal scaling.

**MMR Retrieval**: Maximal Marginal Relevance provides diverse results instead of semantically similar duplicates. Retrieves 4 chunks per query.

**Chainlit**: Provides chat UI with built-in history management and source citations.

## Project Structure

```
├── src/
│   ├── config.py       # Settings and paths
│   ├── ingest.py       # Document loading and embedding
│   └── rag.py          # Query logic with conversation history
├── main.py             # Chainlit UI
├── data/               # Source documents
└── chroma_db/          # Vector store (gitignored)
```

## Known Limitations

- Conversation history stored in-memory (not persistent across sessions)
- Basic error handling without retry logic or circuit breakers
- No metrics collection or user feedback mechanism

## Future Roadmap

**Production Deployment**: Current implementation runs as a single Python process. For production, containerize the application with Docker (separate containers for ingestion and query API). Deploy on Kubernetes for real-time applications with automatic scaling and health checks, or use Azure Container Apps for batch processing workloads with simpler deployment.

**Advanced Ingestion**: Support for complex file types (PDFs with tables, code files) and multiple chunking strategies. For code repositories, implement AST-based chunking to preserve function/class boundaries. Add incremental ingestion by hashing document content to skip unchanged files.

**Testing Strategy**: Build multi-layered testing approach:
- Unit tests for retriever behavior and error handling
- Golden dataset of 50+ Q&A pairs to track hit rate and faithfulness metrics
- Automated evaluation with RAGAS framework in CI/CD pipeline

**Observability**: Implement token usage tracking and cost monitoring per query. Add logging for LLM calls, embedding operations, and retrieval performance to identify optimization opportunities.

