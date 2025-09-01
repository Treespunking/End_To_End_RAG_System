# api_gateway.py
"""
API Gateway for RAG System
This service acts as the entry point for all external requests,
handling ingestion and querying operations through inter-service communication.
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from pydantic import BaseModel
import aiofiles
import uuid

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 API Gateway starting up...")
    yield
    logger.info("🛑 API Gateway shutting down...")

app = FastAPI(
    title="RAG System API Gateway",
    description="API Gateway for Document Ingestion and Query Processing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration for microservices
ETL_SERVICE_URL = os.getenv("ETL_SERVICE_URL", "http://etl-service:8001")
QUERY_ENGINE_URL = os.getenv("QUERY_ENGINE_URL", "http://query-engine:8002")

# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    timestamp: datetime

class IngestionResponse(BaseModel):
    status: str
    message: str
    file_path: Optional[str] = None

# Helper functions
async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file to disk and return path"""
    file_extension = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = UPLOAD_DIR / unique_filename
    
    async with aiofiles.open(file_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)
    
    logger.info(f"📁 Saved uploaded file: {file_path}")
    return str(file_path)

async def call_etl_service(file_path: str, file_type: str) -> Dict[str, Any]:
    """Call ETL service to process the uploaded file"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ETL_SERVICE_URL}/process",
                json={
                    "path": file_path,
                    "type": file_type
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                logger.info("✅ ETL service processing initiated successfully")
                return {
                    "status": "success",
                    "message": "Ingestion started successfully",
                    "details": response.json()
                }
            else:
                logger.error(f"❌ ETL service returned error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"ETL service error: {response.text}"
                )
                
    except httpx.RequestError as e:
        logger.error(f"❌ Network error calling ETL service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ETL service unavailable"
        )

async def call_query_engine(question: str) -> Dict[str, Any]:
    """Call query engine service to process the question"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{QUERY_ENGINE_URL}/ask",
                json={"question": question},
                timeout=120.0
            )
            
            if response.status_code == 200:
                logger.info("✅ Query engine processed successfully")
                return response.json()
            else:
                logger.error(f"❌ Query engine returned error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Query engine error: {response.text}"
                )
                
    except httpx.RequestError as e:
        logger.error(f"❌ Network error calling query engine: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query engine unavailable"
        )

# API Endpoints
@app.post("/ingest/pdf", response_model=IngestionResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Ingest a PDF file for processing.
    
    Args:
        file: PDF file to be ingested
        
    Returns:
        IngestionResponse: Status of the ingestion process
    """
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
        )
    
    try:
        # Save the uploaded file
        file_path = await save_uploaded_file(file)
        
        # Call ETL service asynchronously
        etl_result = await call_etl_service(file_path, "pdf")
        
        return IngestionResponse(
            status="success",
            message="PDF ingestion started successfully",
            file_path=file_path
        )
        
    except Exception as e:
        logger.error(f"❌ Error during PDF ingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )

@app.post("/ingest/docx", response_model=IngestionResponse)
async def ingest_docx(file: UploadFile = File(...)):
    """
    Ingest a DOCX file for processing.
    
    Args:
        file: DOCX file to be ingested
        
    Returns:
        IngestionResponse: Status of the ingestion process
    """
    if file.content_type != "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only DOCX files are allowed"
        )
    
    try:
        # Save the uploaded file
        file_path = await save_uploaded_file(file)
        
        # Call ETL service asynchronously
        etl_result = await call_etl_service(file_path, "docx")
        
        return IngestionResponse(
            status="success",
            message="DOCX ingestion started successfully",
            file_path=file_path
        )
        
    except Exception as e:
        logger.error(f"❌ Error during DOCX ingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )

@app.post("/ingest/web", response_model=IngestionResponse)
async def ingest_web(url: str = Form(...)):
    """
    Ingest content from a web URL for processing.
    
    Args:
        url: Web URL to be ingested
        
    Returns:
        IngestionResponse: Status of the ingestion process
    """
    try:
        # Call ETL service asynchronously with URL
        etl_result = await call_etl_service(url, "web")
        
        return IngestionResponse(
            status="success",
            message="Web content ingestion started successfully",
            file_path=url
        )
        
    except Exception as e:
        logger.error(f"❌ Error during web ingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_rag(question: str = Form(...)):
    """
    Process a user query using the RAG system.
    
    Args:
        question: Question to be answered
        
    Returns:
        QueryResponse: Answer to the question
    """
    if not question or not question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )
    
    try:
        # Call query engine service
        result = await call_query_engine(question)
        
        return QueryResponse(
            answer=result.get("answer", "No answer provided"),
            timestamp=datetime.now()
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Error during query processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """
    Health check endpoint to verify service status.
    
    Returns:
        Dict[str, str]: Health status information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "API Gateway"
    }

@app.get("/status", response_model=Dict[str, Any])
async def service_status():
    """
    Detailed service status including connectivity to microservices.
    
    Returns:
        Dict[str, Any]: Service status information
    """
    status_info = {
        "timestamp": datetime.now().isoformat(),
        "service": "API Gateway",
        "microservices": {}
    }
    
    # Check ETL service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ETL_SERVICE_URL}/health", timeout=5.0)
            status_info["microservices"]["etl_service"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_code": response.status_code
            }
    except Exception as e:
        status_info["microservices"]["etl_service"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Check Query Engine
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{QUERY_ENGINE_URL}/health", timeout=5.0)
            status_info["microservices"]["query_engine"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_code": response.status_code
            }
    except Exception as e:
        status_info["microservices"]["query_engine"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    return status_info

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error occurred",
            "error": str(exc)
        }
    )

# Run the application
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_GATEWAY_HOST", "0.0.0.0")
    port = int(os.getenv("API_GATEWAY_PORT", 8000))
    reload = os.getenv("API_GATEWAY_RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting API Gateway on {host}:{port}")
    uvicorn.run(
        "api_gateway:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )