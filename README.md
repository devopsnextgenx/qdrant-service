# Qdrant Indexing & Search Service (FastAPI + Uvicorn)

🔧 Small FastAPI service to index and search two content types into Qdrant:
- **captions** — per-image YAML files in `data/captions/<thread>/` (use `corrected_text`, `translated_text`, `ocr.full_text`)
- **stories** — pages in `data/stories/<thread>/ymls/page_*.yml` (each page contains posts; each post is indexed)

Features
- Index data from workspace YAML files into Qdrant (vector DB)
- Pluggable embedding backends (default: `sentence-transformers`) — set `EMBEDDING_BACKEND=ollama` to use Ollama
- REST API for indexing and vector search

Quickstart
1. Start Qdrant (example):

   docker run -p 6333:6333 qdrant/qdrant:v1.2.0

2. Sync dependencies with `uv`

   # install the `uv` tool and sync dependencies into an isolated environment
   python -m pip install --upgrade pip
   python -m pip install uv
   uv sync

3. Start service (inside the uv environment)

   uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

4. Build package with `uv`

   uv build

4. Index all data

   curl -X POST "http://localhost:8000/index?type=all"

4. Index all data

   curl -X POST "http://localhost:8000/index?type=all"

5. Search

   curl "http://localhost:8000/search?q=missing%20girl&type=captions&limit=5"

Configuration
- The project reads settings from `config/config.yml` by default. You can set `CONFIG_PATH` environment variable to point to a different YAML file.
- Key configuration options:
  - `embeddings.backend`: `ollama` (default) or `sentence_transformers`
  - `qdrant.url`: address of the Qdrant service
  - `indexing.batch_size`: number of documents to index per batch (default 64)
  - `search.top_k` and `search.score_threshold`: control default search behavior
  - `logging.file`: location of log file (default: `logs/qdrant-service.log`)

Notes
- Configure QDRANT_URL via environment variable (default http://localhost:6333) or via `config/config.yml` (preferred)
- The code is intentionally defensive about YAML structure — it will try to extract text fields automatically

Embeddings
- Default: sentence-transformers (`all-MiniLM-L6-v2`) or `ollama` if configured in `config/config.yml`
- To use Ollama (local embedding server) via config:
  - Set `embeddings.backend: ollama`
  - Set `embeddings.ollama.model` and/or `embeddings.ollama.endpoint` as needed (default endpoint: `http://localhost:11434/embed`). Use a model name present on your Ollama instance — check available models with `ollama ls` (examples: `all-minilm:latest`, `nomic-embed-text:latest`, `embeddinggemma:latest`).
    ```bash
    CONFIG_PATH=config/config.yml \
      uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```

If your Ollama endpoint returns a different JSON format, provide a sample response and I can adapt the adapter to parse it.

Files
- `app/` — FastAPI app, ingestion and qdrant wrappers
- `scripts/index_all.py` — CLI to index workspace data

