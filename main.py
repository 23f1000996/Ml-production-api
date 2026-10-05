from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
INDEX_PATH = BASE_DIR / "static" / "index.html"

app = FastAPI(title="Iris AI — Flower Classifier", version="1.0.0")
IRIS_CLASSES = ["setosa", "versicolor", "virginica"]

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=4, max_length=4)

@app.get("/")
def home():
    return FileResponse(INDEX_PATH)

@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "error", "model_loaded": model is not None}

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    features = np.array(request.features, dtype=float).reshape(1, -1)
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0].tolist() if hasattr(model, "predict_proba") else None
    return {"prediction": prediction, "class_name": IRIS_CLASSES[prediction], "probabilities": probabilities}
