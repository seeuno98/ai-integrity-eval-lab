"""FastAPI inference endpoint for toxicity classification."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_config
from src.predict import ToxicityPredictor
from src.utils import get_logger

logger = get_logger(__name__)

# Global predictor instance — initialised in the lifespan handler.
_predictor: ToxicityPredictor | None = None
_start_time: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once on startup; release on shutdown."""
    global _predictor, _start_time
    config_path = app.state.config_path if hasattr(app.state, "config_path") else "configs/base.yaml"
    cfg = load_config(config_path)
    logger.info("Loading model for API ...")
    _predictor = ToxicityPredictor(
        model_dir=cfg.artifacts.model_dir,
        max_length=cfg.model.max_length,
    )
    _start_time = time.time()
    logger.info("API ready.")
    yield
    logger.info("Shutting down API.")


app = FastAPI(
    title="AI Integrity Eval Lab — Toxicity Classifier",
    description=(
        "Binary toxicity classifier fine-tuned on lmsys/toxic-chat. "
        "Returns predicted label and confidence score."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request / response schemas ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        example="This is perfectly fine content.",
    )


class PredictResponse(BaseModel):
    label: str = Field(..., example="non-toxic")
    label_id: int = Field(..., example=0)
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.9876)
    toxic_prob: float = Field(..., ge=0.0, le=1.0, example=0.0124)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> dict[str, Any]:
    """Return API health status and uptime."""
    return {
        "status": "ok",
        "model_loaded": _predictor is not None,
        "uptime_seconds": round(time.time() - _start_time, 2),
    }


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request: PredictRequest) -> dict[str, Any]:
    """Classify a text string as toxic or non-toxic.

    Returns the predicted label, label integer id, confidence score for
    the predicted label, and the raw probability for the toxic class.

    Raises:
        HTTPException(503): If the model has not been loaded yet.
        HTTPException(422): If the input text is empty (handled by Pydantic).
        HTTPException(500): On unexpected inference errors.
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        result = _predictor.predict(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Inference error.") from exc

    return result


# ── Dev entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
