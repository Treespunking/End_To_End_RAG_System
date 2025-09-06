# etl_service.py
"""
ETL Service for RAG System
Receives ingestion requests from API Gateway, processes PDF, DOCX, or web URLs,
performs hierarchical chunking, and uploads to Weaviate + MongoDB.
"""

from dotenv import load_dotenv

load_dotenv()

import os
import uuid
import logging
import asyncio
from typing import List, Dict, Any
import numpy as np

# -------------------------------
# Set Environment Variables Early
# -------------------------------
os.environ["USER_AGENT"] = "my-rag-app"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

# -------------------------------
# Configure Logging
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------
# FastAPI Setup
# -------------------------------
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="ETL Service",
    description="Handles document loading, chunking, and vector storage.",
    version="1.0.0"
)

# -------------------------------
# Request Model
# -------------------------------
class ProcessRequest(BaseModel):
    path: str
    type: str  # "pdf", "docx", "web"


# -------------------------------
# Step 1: Load Documents
# -------------------------------
from langchain_community.document_loaders import UnstructuredPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain_core.documents import Document

def load_documents(req: ProcessRequest) -> List[Document]:
    docs = []

    if req.type == "pdf":
        file_path = req.path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found: {file_path}")
        loader = UnstructuredPDFLoader(file_path, strategy="auto")
        pdf_docs = loader.load()
        docs.extend(pdf_docs)
        logger.info(f"✅ Loaded {len(pdf_docs)} documents from PDF.")

    elif req.type == "docx":
        file_path = req.path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"DOCX not found: {file_path}")
        loader = Docx2txtLoader(file_path)
        docx_docs = loader.load()
        docs.extend(docx_docs)
        logger.info(f"✅ Loaded {len(docx_docs)} documents from DOCX.")

    elif req.type == "web":
        url = req.path.strip()
        if not url.startswith("http"):
            raise ValueError(f"Invalid URL: {url}")
        loader = WebBaseLoader([url])
        web_docs = loader.load()
        docs.extend(web_docs)
        logger.info(f"✅ Loaded {len(web_docs)} documents from web.")

    else:
        raise ValueError(f"Unsupported document type: {req.type}")

    if not docs:
        raise ValueError("No content extracted from the document.")

    return docs


# -------------------------------
# Step 2: Token Counter
# -------------------------------
import tiktoken

def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# -------------------------------
# Step 3: Initialize Embeddings (Strict Local Loading)
# -------------------------------
from langchain_huggingface import HuggingFaceEmbeddings

logger.info("🔹 Preparing HuggingFace embeddings...")

cache_dir = os.getenv("HF_CACHE_DIR", "./models")
model_name = "bge-small-en-v1.5"

model_path = os.path.join(cache_dir, model_name)
if not os.path.exists(model_path):
    logger.critical(
        f"❌ Model {model_name} not found in {cache_dir}. "
        "Please ensure you've downloaded it locally first:\n\n"
        f"from sentence_transformers import SentenceTransformer\n"
        f"SentenceTransformer('{model_name}', cache_folder='{cache_dir}')"
    )
    raise FileNotFoundError(f"Model not found: {model_path}")

embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs={"device": "cpu"},  # Use "cuda" if GPU available
    encode_kwargs={"normalize_embeddings": True},
)

# -------------------------------
# Step 4: Parent Splitter
# -------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=count_tokens,
    separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
)


# -------------------------------
# Step 5: Child Chunks via NLTK + Clustering
# -------------------------------
from langchain_text_splitters import NLTKTextSplitter
from sklearn.cluster import KMeans

import nltk
nltk.download('punkt', quiet=True)

sentence_splitter = NLTKTextSplitter(chunk_size=100, chunk_overlap=20)

def create_hierarchical_chunks(docs: List[Document]):
    parents = parent_splitter.split_documents(docs)
    children: List[Document] = []
    parent_store = {}

    for parent in parents:
        parent_id = str(uuid.uuid4())
        parent_store[parent_id] = {
            "content": parent.page_content,
            "metadata": parent.metadata
        }

        sentences = sentence_splitter.split_documents([parent])
        if not sentences:
            continue

        if len(sentences) == 1:
            child = sentences[0]
            child.metadata.update({
                "parent_id": parent_id,
                "child_chunk_tokens": count_tokens(child.page_content),
                "child_source_type": "pdf" if "pdf" in parent.metadata.get("source", "").lower()
                else "docx" if "docx" in parent.metadata.get("source", "").lower() else "web"
            })
            if "page" in parent.metadata:
                child.metadata["source_page"] = parent.metadata["page"]
            children.append(child)
            continue

        try:
            vectors = np.array([embeddings.embed_query(s.page_content) for s in sentences])
        except Exception as e:
            logger.warning(f"⚠️ Embedding failed during clustering: {e}")
            continue

        n_clusters = max(2, min(len(sentences) // 3, 6))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(vectors)

        clusters = {}
        for s, label in zip(sentences, labels):
            clusters.setdefault(label, []).append(s)

        for label, group in clusters.items():
            content = " ".join([s.page_content.strip() for s in group])
            meta = {
                "parent_id": parent_id,
                "source": parent.metadata.get("source", "unknown"),
                "child_chunk_tokens": count_tokens(content),
                "cluster_id": int(label),
                "num_sentences": len(group),
                "child_source_type": "pdf" if "pdf" in parent.metadata.get("source", "").lower()
                else "docx" if "docx" in parent.metadata.get("source", "").lower() else "web"
            }
            if "page" in parent.metadata:
                meta["source_page"] = int(parent.metadata["page"])

            children.append(Document(page_content=content, metadata=meta))

    # Add global child indexing
    for i, child in enumerate(children):
        child.metadata["child_index"] = int(i)
        child.metadata["total_children"] = int(len(children))

    logger.info(f"✅ Created {len(parents)} parents and {len(children)} children.")
    return parents, children, parent_store


# -------------------------------
# Step 6: Async Upload to Weaviate
# -------------------------------
async def upload_to_weaviate(children: List[Document]):
    weaviate_client = None
    try:
        import weaviate
        from weaviate.classes.init import Auth
        from weaviate.classes.config import Configure, Property, DataType
        from weaviate.classes.data import DataObject

        WEAVIATE_URL = os.getenv("WEAVIATE_URL")
        WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

        if not WEAVIATE_URL or not WEAVIATE_API_KEY:
            raise ValueError("Weaviate URL or API key missing.")

        weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
            skip_init_checks=True
        )

        if not weaviate_client.is_ready():
            raise ConnectionError("Weaviate client not ready.")

        collection_name = "ChildChunk"
        if not weaviate_client.collections.exists(collection_name):
            weaviate_client.collections.create(
                name=collection_name,
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="parent_id", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="child_source_type", data_type=DataType.TEXT),
                    Property(name="source_page", data_type=DataType.INT),
                    Property(name="child_chunk_tokens", data_type=DataType.INT),
                    Property(name="child_index", data_type=DataType.INT),
                    Property(name="total_children", data_type=DataType.INT),
                    Property(name="cluster_id", data_type=DataType.INT),
                    Property(name="num_sentences", data_type=DataType.INT),
                ],
                vector_config=Configure.Vectors.self_provided()  # bge-small-en-v1.5 → 384 dim
            )
            logger.info(f"✅ Created Weaviate collection: '{collection_name}'")

        collection = weaviate_client.collections.get(collection_name)
        data_to_upload: List[DataObject] = []
        failed_count = 0

        logger.info("🧠 Generating embeddings for child chunks...")
        for i, child in enumerate(children):
            try:
                vec = embeddings.embed_query(child.page_content)
                props: Dict[str, Any] = {"text": child.page_content}
                for k, v in child.metadata.items():
                    if isinstance(v, np.integer):
                        props[k] = int(v)
                    elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                        props[k] = None
                    elif isinstance(v, float):
                        props[k] = float(v)
                    elif v is None:
                        props[k] = None
                    else:
                        props[k] = str(v) if not isinstance(v, (str, int, bool)) else v

                data_to_upload.append(DataObject(properties=props, vector=vec))
            except Exception as e:
                logger.warning(f"Failed to process chunk {i}: {e}")
                failed_count += 1

        total = len(data_to_upload)
        if total == 0:
            logger.warning("❌ No valid data to upload to Weaviate.")
            return

        batch_size = 32
        success_count = 0

        for i in range(0, total, batch_size):
            batch = data_to_upload[i:i + batch_size]
            try:
                result = collection.data.insert_many(batch)
                if result.has_errors:
                    for uuid, err in result.errors.items():
                        logger.error(f"❌ Insert error: {err}")
                else:
                    success_count += len(batch)
            except Exception as e:
                logger.error(f"❌ Batch upload failed: {e}")

        logger.info(f"✅ Uploaded {success_count}/{total} child chunks to Weaviate.")

    except Exception as e:
        logger.error(f"❌ Weaviate upload failed: {e}")
    finally:
        if weaviate_client:
            weaviate_client.close()
            logger.info("🟢 Weaviate client closed.")


# -------------------------------
# Step 7: Async Upload to MongoDB
# -------------------------------
async def upload_to_mongodb(parent_store: Dict[str, Any]):
    mongo_client = None
    try:
        from motor.motor_asyncio import AsyncIOMotorClient

        MONGODB_URI = os.getenv("MONGODB_URI")
        if not MONGODB_URI:
            raise ValueError("MONGODB_URI not set.")

        mongo_client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
        await mongo_client.admin.command('ping')
        logger.info("🟢 MongoDB: Async connection successful!")

        db = mongo_client["rag_system"]
        collection = db["parents"]

        await collection.delete_many({})
        if parent_store:
            records = [
                {"_id": pid, "content": data["content"], "metadata": data["metadata"]}
                for pid, data in parent_store.items()
            ]
            result = await collection.insert_many(records)
            await collection.create_index("_id")
            logger.info(f"✅ Stored {len(result.inserted_ids)} parent chunks in MongoDB.")
        else:
            logger.info("🟡 No parent chunks to store.")

    except Exception as e:
        logger.error(f"❌ MongoDB operation failed: {e}")
    finally:
        if mongo_client:
            mongo_client.close()
            logger.info("🟢 MongoDB client closed.")


# -------------------------------
# Main Processing Endpoint
# -------------------------------
@app.post("/process")
async def process_document(request: ProcessRequest):
    try:
        logger.info(f"📥 Received {request.type.upper()} processing request: {request.path}")

        # Load documents
        docs = load_documents(request)

        # Chunking
        parents, children, parent_store = create_hierarchical_chunks(docs)

        # Upload
        await asyncio.gather(
            upload_to_weaviate(children),
            upload_to_mongodb(parent_store),
            return_exceptions=True
        )

        return {
            "status": "success",
            "message": "Document processed and stored successfully.",
            "stats": {
                "parents": len(parents),
                "children": len(children)
            }
        }

    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# Health Check
# -------------------------------
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "ETL Service"}


# -------------------------------
# Run with Uvicorn (if script is main)
# -------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    logger.info(f"🚀 Starting ETL Service on port {port}")
    uvicorn.run("etl_service:app", host="0.0.0.0", port=port, reload=False)
