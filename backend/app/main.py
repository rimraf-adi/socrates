"""FastAPI application for Socrates research assistant."""

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

from app.routers import search, research, models, history


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("🚀 Socrates API starting...")
    yield
    print("👋 Socrates API shutting down...")


app = FastAPI(
    title="Socrates API",
    description="AI-powered research assistant with LangGraph agents",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(research.router, prefix="/api", tags=["research"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(history.router, prefix="/api", tags=["history"])


@app.get("/")
async def root():
    return {"message": "Socrates API v2.0 - Python + LangGraph"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
