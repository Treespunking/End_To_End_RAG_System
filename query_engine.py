"""
Query Engine Microservice for Retrieval-Augmented Generation (RAG)
Exposes a FastAPI endpoint to handle user queries by retrieving relevant context
from Weaviate and generating responses using OpenRouter's LLM API.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -------------------------------
# Pydantic Models
# -------------------------------
class QueryRequest(BaseModel):
    query: str                # The user's question
    top_k: Optional[int] = 5  # Optional: number of results to retrieve

class QueryResponse(BaseModel):
    answer: str                     # The AI's generated answer
    sources: Optional[List[str]] = None  # Optional: list of source documents

# -------------------------------
# LangChain & Weaviate Imports
# -------------------------------
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_weaviate import WeaviateVectorStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import weaviate
from weaviate.classes.init import Auth

# -------------------------------
# Query Engine Class
# -------------------------------
class QueryEngine:
    def __init__(self):
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all components: LLM, embeddings, vector store, retriever, and chain."""
        try:
            # --- LLM ---
            logger.info("🔹 Initializing LLM...")
            self.llm = ChatOpenAI(
                model=os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free"),
                base_url="https://openrouter.ai/api/v1".strip(),
                api_key=os.getenv("OPENROUTER_API_KEY"),
                extra_headers={
                    "HTTP-Referer": os.getenv("APP_URL", "http://localhost"),
                    "X-Title": os.getenv("APP_NAME", "RAG-Query-Engine")
                },
                timeout=30
            )
            logger.info("✅ LLM initialized successfully")

            # --- Embeddings (strict local loading) ---
            logger.info("🔹 Preparing HuggingFace embeddings...")
            cache_dir = os.getenv("HF_CACHE_DIR", "./models")
            model_name = "bge-small-en-v1.5"

            # ✅ Verify model exists locally — if missing, raise critical error
            model_path = os.path.join(cache_dir, model_name)
            if not os.path.exists(model_path):
                logger.critical(
                    f"❌ Model {model_name} not found in {cache_dir}. "
                    "Please ensure you've downloaded it locally first:\n\n"
                    f"from sentence_transformers import SentenceTransformer\n"
                    f"SentenceTransformer('{model_name}', cache_folder='{cache_dir}')"
                )
                raise FileNotFoundError(f"Model not found: {model_path}")

            # ✅ Use the locally cached model only
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("✅ Embeddings loaded successfully from local cache")

            # --- Weaviate ---
            logger.info("🔹 Connecting to Weaviate...")
            weaviate_url = os.getenv("WEAVIATE_URL")
            weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
            if not weaviate_url or not weaviate_api_key:
                raise ValueError("❌ Missing WEAVIATE_URL or WEAVIATE_API_KEY")

            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,
                auth_credentials=Auth.api_key(weaviate_api_key),
                skip_init_checks=True
            )

            if not client.is_ready():
                raise ConnectionError("❌ Weaviate client not ready")
            logger.info("✅ Weaviate connected successfully")

            # --- Vector Store ---
            logger.info("🔹 Initializing vector store...")
            self.vectorstore = WeaviateVectorStore(
                client=client,
                index_name="ChildChunk",
                text_key="text",
                embedding=self.embeddings
            )

            # --- Retriever ---
            logger.info("🔹 Initializing retriever...")
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.4}
            )
            logger.info("✅ Retriever initialized successfully")

            # --- RAG Chain ---
            logger.info("🔹 Creating RAG chain...")
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are a helpful AI assistant that answers questions based on the provided context. "
                    "Use only the information in the context. If the context is insufficient, say so. "
                    "Cite sources when possible."
                )),
                ("human", "Context:\n{context}\n\nQuestion: {question}")
            ])

            self.output_parser = StrOutputParser()
            self.chain = (
                {"context": self.retriever | self._format_context, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | self.output_parser
            )
            logger.info("✅ Query Engine components initialized successfully")

        except Exception:
            logger.exception("❌ Failed to initialize Query Engine")
            raise

    def _format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents into a context string."""
        formatted = []
        for doc in docs:
            meta_str = "\n".join([f"{k}: {v}" for k, v in doc.metadata.items() if k != "text"])
            formatted.append(
                f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                f"Content: {doc.page_content}\n"
                f"{meta_str}\n---"
            )
        return "\n".join(formatted)

    def query(self, question: str) -> str:
        """Run the RAG pipeline synchronously."""
        try:
            logger.info(f"🔍 Querying: {question}")
            result = self.chain.invoke(question)
            logger.info("✅ Query completed")
            return result
        except Exception:
            logger.exception("❌ Error during query")
            return "Sorry, I encountered an error while processing your request."

    async def async_query(self, question: str) -> str:
        """Asynchronous wrapper for query."""
        return await asyncio.get_event_loop().run_in_executor(None, self.query, question)

# -------------------------------
# Initialize Global Engine
# -------------------------------
query_engine = None

# -------------------------------
# FastAPI Lifespan
# -------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global query_engine
    logger.info("🚀 Query Engine starting up...")

    try:
        query_engine = QueryEngine()
        logger.info("🟢 Query Engine loaded successfully")
    except Exception as e:
        logger.critical(f"💥 Failed to start Query Engine: {e}")
        query_engine = None

    yield

    logger.info("🛑 Query Engine shutting down...")

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(
    title="Query Engine",
    description="RAG-based question answering using Weaviate and OpenRouter.",
    version="1.0.0",
    lifespan=lifespan
)

# -------------------------------
# API Endpoints
# -------------------------------
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Endpoint to answer user questions using RAG."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if query_engine is None:
        raise HTTPException(status_code=503, detail="Query engine not initialized")

    answer = await query_engine.async_query(request.query)
    return QueryResponse(answer=answer)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if query_engine else "unavailable",
        "service": "Query Engine"
    }

# -------------------------------
# Run with Uvicorn
# -------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8002))
    logger.info(f"🚀 Starting Query Engine on port {port}")
    import uvicorn
    uvicorn.run("query_engine:app", host="0.0.0.0", port=port, reload=False)
