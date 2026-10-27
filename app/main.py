import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "final_readmission_model.joblib"


class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature dictionary matching the training schema.")


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_name: str
    status: str = "ok"


app = FastAPI(title="Readmission Risk API", version="1.0.0")


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(payload: PredictionRequest):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact not found. Train the model first.")

    model = joblib.load(MODEL_PATH)

    feature_record = pd.DataFrame([payload.features])
    feature_record = feature_record.replace({"None": None, "nan": None})

    prediction = int(model.predict(feature_record)[0])
    probability = float(model.predict_proba(feature_record)[0, 1])

    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        model_name="random_forest",
    )
