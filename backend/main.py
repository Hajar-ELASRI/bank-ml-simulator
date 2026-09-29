"""FastAPI backend serving predictions and the static simulator interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FRONTEND_DIR = ROOT / "frontend"
ALGORITHMS = ("svm", "random_forest", "mlp")


def read_json(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(
            f"Fichier requis absent : {path}. Exécutez d'abord 'python train_models.py'."
        )
    return json.loads(path.read_text(encoding="utf-8"))


METRICS = read_json(MODELS_DIR / "metrics.json")
SCHEMA = read_json(MODELS_DIR / "schema.json")
MODELS = {
    algorithm: joblib.load(MODELS_DIR / f"{algorithm}.pkl")
    for algorithm in ALGORITHMS
}


class PredictionRequest(BaseModel):
    """All 16 predictors from bank.csv plus the selected model."""

    model_config = ConfigDict(extra="forbid")

    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    algorithm: Literal["svm", "random_forest", "mlp"]


app = FastAPI(
    title="Bank ML Simulator API",
    description="Prédiction de la souscription à un dépôt bancaire avec SVM, Random Forest ou MLP.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "models": list(MODELS)}


@app.get("/api/schema")
def get_schema() -> dict:
    """Expose field types and dataset values used to generate the form."""
    return SCHEMA


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    payload = request.model_dump()
    algorithm = payload.pop("algorithm")

    # Refuse categorical values outside this dataset instead of silently accepting typos.
    for feature, allowed_values in SCHEMA["categorical_features"].items():
        if payload[feature] not in allowed_values:
            raise HTTPException(
                status_code=422,
                detail=f"Valeur invalide pour '{feature}': {payload[feature]!r}",
            )

    ordered_data = {feature: payload[feature] for feature in SCHEMA["feature_order"]}
    sample = pd.DataFrame([ordered_data])
    prediction = str(MODELS[algorithm].predict(sample)[0])

    return {
        "prediction": prediction,
        "algorithm": algorithm,
        "accuracy": METRICS[algorithm]["accuracy"],
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
