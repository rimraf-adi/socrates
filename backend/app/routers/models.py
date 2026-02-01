"""Model management router for LMStudio model configuration."""

import json
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

router = APIRouter()

# Path to store models configuration
MODELS_FILE = os.path.join(os.path.dirname(__file__), "..", "models.json")

# Default models - simple model IDs as used in LMStudio
DEFAULT_MODELS = [
    "mistralai/ministral-3-3b",
]


class ModelConfig(BaseModel):
    """Model configuration response."""
    models: List[str]
    selected: str | None = None


class AddModelRequest(BaseModel):
    """Request to add a new model."""
    model_id: str


def load_models() -> List[str]:
    """Load models from JSON file."""
    if os.path.exists(MODELS_FILE):
        try:
            with open(MODELS_FILE, "r") as f:
                data = json.load(f)
                return data.get("models", DEFAULT_MODELS)
        except (json.JSONDecodeError, IOError):
            pass
    return DEFAULT_MODELS.copy()


def save_models(models: List[str]) -> None:
    """Save models to JSON file."""
    with open(MODELS_FILE, "w") as f:
        json.dump({"models": models}, f, indent=2)


@router.get("/models", response_model=ModelConfig)
async def get_models():
    """Get list of configured models."""
    models = load_models()
    return ModelConfig(models=models, selected=models[0] if models else None)


@router.post("/models", response_model=ModelConfig)
async def add_model(request: AddModelRequest):
    """Add a new model to the list."""
    models = load_models()
    
    if request.model_id in models:
        raise HTTPException(status_code=400, detail="Model already exists")
    
    models.append(request.model_id)
    save_models(models)
    
    return ModelConfig(models=models, selected=request.model_id)


@router.delete("/models/{model_id:path}", response_model=ModelConfig)
async def delete_model(model_id: str):
    """Remove a model from the list."""
    models = load_models()
    
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    models.remove(model_id)
    save_models(models)
    
    return ModelConfig(models=models, selected=models[0] if models else None)
