# Retrieval-Augmented Generation (RAG) System

A production-ready, microservices RAG stack for ingesting PDFs/DOCX/Web pages, chunking them hierarchically, vectorizing with **bge-small-en-v1.5**, storing in **Weaviate**, tracking parents in **MongoDB**, and answering questions via a **FastAPI** query engine + **OpenRouter LLM**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/Framework-FastAPI-teal)
![Weaviate](https://img.shields.io/badge/VectorDB-Weaviate-orange)
![MongoDB](https://img.shields.io/badge/Database-MongoDB-green)
![Docker](https://img.shields.io/badge/Container-Docker-blueviolet)
![LangChain](https://img.shields.io/badge/Orchestration-LangChain-yellow)

---

## Features

- **End-to-end pipeline**: ingest → hierarchical chunking (parent/child) → embeddings → Weaviate + MongoDB → RAG query
- **Strict local embeddings** using `BAAI/bge-small-en-v1.5` (pull once, then served locally)
- **Three microservices**:
  - **API Gateway** (file uploads, routing)
  - **ETL Service** (load, split, embed, upload)
  - **Query Engine** (retrieve+generate via OpenRouter)
- **Typed FastAPI endpoints**, robust logging, health checks, and test scripts
- **Docker Compose** for one-command spin up

---

## Requirements

- Python **3.10+**
- Docker + Docker Compose (recommended)
- Weaviate (Cloud or Self-hosted) & API key
- MongoDB instance (Atlas or local)
- OpenRouter API key (or compatible OpenAI API endpoint)
- Disk space for local HF model cache (`bge-small-en-v1.5`)

---

## Quickstart (Docker)

### 1) Clone & enter the repo
```bash
git clone https://github.com/Treespunking/End_To_End_RAG_System.git
cd End_To_End_RAG_System
````

### 2) Prepare model cache (one-time)

```bash
python download_model.py
# caches BAAI/bge-small-en-v1.5 to ./models
```

### 3) Create `.env`

```env
# External services
WEAVIATE_URL=https://your-weaviate-endpoint
WEAVIATE_API_KEY=your_weaviate_key
MONGODB_URI=mongodb+srv://<user>:<pass>@<cluster>/rag_system

# LLM
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=mistralai/mistral-7b-instruct:free

# App metadata (optional)
APP_URL=http://localhost
APP_NAME=RAG-Query-Engine
```

### 4) Run the stack

```bash
docker compose up -d --build
```

**Services**

* API Gateway: `http://localhost:8000`
* ETL Service: `http://localhost:8001`
* Query Engine: `http://localhost:8002`

> Health checks are built in; API Gateway depends on the other two being healthy.

---

## Local Dev (without Docker)

> Recommended only if you’re comfortable installing system deps (Poppler, Tesseract, etc.).

* Create three virtualenvs or reuse one (order matters due to deps):

  1. **ETL**: `pip install -r requirements-etl.txt`
  2. **Query**: `pip install -r requirements-query.txt`
  3. **Gateway**: `pip install -r requirements-gateway.txt`
* Ensure `typing_extensions>=4.7.0` is installed first if needed.
* Export environment variables (see `.env` above). Also set:

  * `HF_CACHE_DIR=./models` (or absolute path)
* Run services separately:

```bash
# Terminal 1
uvicorn etl_service:app --host 0.0.0.0 --port 8001

# Terminal 2
uvicorn query_engine:app --host 0.0.0.0 --port 8002

# Terminal 3
uvicorn api_gateway:app --host 0.0.0.0 --port 8000
```

---

## Usage

### Ingest content

**PDF**

```bash
curl -F "file=@path/to/file.pdf" http://localhost:8000/ingest/pdf
```

**DOCX**

```bash
curl -F "file=@path/to/file.docx" http://localhost:8000/ingest/docx
```

**Web URL**

```bash
curl -X POST -H "Content-Type: application/x-www-form-urlencoded" \
     -d "url=https://example.com/page" \
     http://localhost:8000/ingest/web
```

### Ask a question

```bash
curl -X POST -H "Content-Type: application/x-www-form-urlencoded" \
     -d "question=What is Retrieval-Augmented Generation?" \
     http://localhost:8000/query
```

### Health & Status

```bash
curl http://localhost:8000/health
curl http://localhost:8000/status
```

---

## Project Structure

```
.
├── api_gateway.py                # FastAPI gateway (uploads, routing)
├── etl_service.py                # ETL: load → split → embed → store
├── query_engine.py               # Retriever + LLM generation
├── docker-compose.yml            # Three-service orchestration
├── Dockerfile.api
├── Dockerfile.etl
├── Dockerfile.query
├── requirements-gateway.txt
├── requirements-etl.txt
├── requirements-query.txt
├── download_model.py             # Pull BAAI/bge-small-en-v1.5 to ./models
├── test_etl.py                   # Ingestion tester (PDF/DOCX/Web)
├── test_query.py                 # Query tester
└── uploads/                      # Mounted upload dir (created at runtime)
```

---

## Endpoints (API Gateway)

* `POST /ingest/pdf` — multipart upload (`file`) → starts ETL for PDF
* `POST /ingest/docx` — multipart upload (`file`) → starts ETL for DOCX
* `POST /ingest/web` — form field (`url`) → starts ETL for web page
* `POST /query` — form field (`question`) → answers via RAG
* `GET /health` — gateway health ping
* `GET /status` — reports health of ETL & Query Engine

> ETL exposes `POST /process` and `GET /health`; Query Engine exposes `POST /ask` and `GET /health`.

---

## How it Works (Pipeline)

1. **Load**: PDFs via `UnstructuredPDFLoader`, DOCX via `Docx2txtLoader`, and pages via `WebBaseLoader`.
2. **Chunk**: Parent chunks with `RecursiveCharacterTextSplitter` (token-aware), then child chunks via NLTK sentence splits + **KMeans clustering** to group semantically related sentences.
3. **Embed**: `HuggingFaceEmbeddings` using locally cached `bge-small-en-v1.5` (384-dim).
4. **Store**:

   * **Weaviate**: child chunks + vectors in collection `ChildChunk` with rich metadata.
   * **MongoDB**: parent chunks/materialized parent store for lineage.
5. **Query**: Retriever (similarity + score threshold) → RAG prompt → OpenRouter LLM → answer.

---

## Testing

### Ingestion tests

```bash
python test_etl.py
# or specific:
python test_etl.py pdf ./docs/sample.pdf
python test_etl.py docx ./docs/sample.docx
python test_etl.py web https://example.com/page
```

### Query tests

```bash
python test_query.py "How does RAG improve answer accuracy?"
```

---

## Configuration Notes

* **Models**: keep `HF_CACHE_DIR=/app/models` in Docker (volume-mount `./models:/app/models`).
* **OpenRouter**: set `OPENROUTER_API_KEY` and (optionally) `OPENROUTER_MODEL`.
* **Weaviate**: set `WEAVIATE_URL` and `WEAVIATE_API_KEY`. Ensure the cluster is reachable.
* **MongoDB**: set `MONGODB_URI`. The ETL creates/uses DB `rag_system`, collection `parents`.

---

## Troubleshooting

* **Embeddings model not found**: run `python download_model.py` before starting the stack.
* **Weaviate not ready**: verify credentials and network; the gateway waits for health but check `docker compose logs query-engine` / `etl-service`.
* **Tokenization errors**: ensure `tiktoken` installed and `typing_extensions>=4.7.0`.
* **Large files**: adjust parent/child chunk sizes or ETL timeouts in code if needed.

---

## License

```
