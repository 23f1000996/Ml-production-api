"""
FastAPI application serving the trained Iris classification model.

Run locally:
    uvicorn main:app --reload

Interactive API docs:
    http://127.0.0.1:8000/docs

Example prediction:
    POST /predict
    {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
"""

from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

app = FastAPI(
    title="Iris ML Production API",
    description="A simple FastAPI service serving a scikit-learn Iris classifier.",
    version="1.0.0",
)

IRIS_CLASSES = ["setosa", "versicolor", "virginica"]

# Load the model once when the application starts.
try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description=(
            "Iris features in this order: "
            "[sepal_length, sepal_width, petal_length, petal_width]"
        ),
    )


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str


@app.get("/")
def root():
    return {
        "message": "Iris ML API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Make sure model.pkl exists.",
        )

    features = np.array(request.features, dtype=float).reshape(1, -1)
    prediction = int(model.predict(features)[0])

    return PredictionResponse(
        prediction=prediction,
        class_name=IRIS_CLASSES[prediction],
    )
