"""
Contract Assistance FastAPI Application
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.contracts import router as contracts_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("Contract Assistance API starting up...")
    yield
    logger.info("Contract Assistance API shutting down...")


app = FastAPI(
    title="Contract Assistance API",
    description="Backend API for Contract Assistance application - handles document processing and metadata extraction",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(contracts_router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "application": "Contract Assistance API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Global health check"""
    return {"status": "healthy"}

