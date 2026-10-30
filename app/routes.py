from importlib.resources import files

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import ValidationError

from app.observability import log_event
from app.prediction import MODEL_PATH, predict_features
from app.schemas import Features, PredictionRequest, PredictionResponse

router = APIRouter()
FORM_HTML = files("app").joinpath("templates/prediction_form.html").read_text(encoding="utf-8")


@router.get("/", response_class=HTMLResponse)
async def form_page():
    return FORM_HTML


@router.post("/predict_form", response_model=PredictionResponse)
async def predict_from_form(request: Request):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact not found. Train the model first.")

    form_data = await request.form()
    try:
        features = Features.model_validate({
            key: None if value == "" else value
            for key, value in form_data.items()
        })
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=error.errors()) from error
    prediction, probability = predict_features(features)

    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        model_name="readmission_risk_model",
    )


@router.get("/health")
def health_check() -> dict[str, str]:
    log_event("health_check", status="ok")
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
def predict_risk(payload: PredictionRequest):
    patient_id = payload.features.patient_nbr
    encounter_id = payload.features.encounter_id
    log_event(
        "prediction_request_received",
        patient_nbr=patient_id,
        encounter_id=encounter_id,
        endpoint="/predict",
    )

    if not MODEL_PATH.exists():
        log_event(
            "prediction_failed",
            patient_nbr=patient_id,
            encounter_id=encounter_id,
            reason="model_not_found",
            status_code=503,
        )
        raise HTTPException(status_code=503, detail="Model artifact not found. Train the model first.")

    prediction, probability = predict_features(payload.features)

    log_event(
        "prediction_completed",
        patient_nbr=patient_id,
        encounter_id=encounter_id,
        prediction=prediction,
        probability=probability,
        model_name="readmission_risk_model",
    )

    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        model_name="readmission_risk_model",
    )
