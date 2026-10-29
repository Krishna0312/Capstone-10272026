import ast
from html.parser import HTMLParser
from pathlib import Path
from typing import get_args

import numpy as np
from fastapi.testclient import TestClient

from app.main import Features, PredictionRequest, app


client = TestClient(app)


def _feature_values():
    values = {}
    for field_name, field in Features.model_fields.items():
        wire_name = field.alias or field_name
        if field_name == "weight":
            value = "[150-175)"
        elif int in get_args(field.annotation):
            value = 7
        else:
            value = f"test-{field_name}"
        values[wire_name] = value
    return values


class _FormControlNames(HTMLParser):
    def __init__(self):
        super().__init__()
        self.names = set()
        self.numeric_minimums = {}

    def handle_starttag(self, tag, attrs):
        if tag in {"input", "select", "textarea"}:
            attributes = dict(attrs)
            name = attributes.get("name")
            if name:
                self.names.add(name)
                if tag == "input" and attributes.get("type") == "number":
                    self.numeric_minimums[name] = attributes.get("min")


def test_predict_endpoint_returns_probability_and_label():
    payload = {
        "features": {
            "encounter_id": 123,
            "patient_nbr": 987,
            "race": "Caucasian",
            "gender": "Female",
            "age": "[50-60)",
            "weight": "[150-175)",
            "admission_type_id": 1,
            "discharge_disposition_id": 1,
            "admission_source_id": 7,
            "time_in_hospital": 4,
            "payer_code": None,
            "medical_specialty": "Cardiology",
            "num_lab_procedures": 12,
            "num_procedures": 1,
            "num_medications": 20,
            "number_outpatient": 0,
            "number_emergency": 0,
            "number_inpatient": 0,
            "diag_1": "250.83",
            "diag_2": "276.2",
            "diag_3": "401.9",
            "number_diagnoses": 5,
            "max_glu_serum": "None",
            "A1Cresult": "None",
            "metformin": "No",
            "repaglinide": "No",
            "nateglinide": "No",
            "chlorpropamide": "No",
            "glimepiride": "No",
            "acetohexamide": "No",
            "glipizide": "No",
            "glyburide": "No",
            "tolbutamide": "No",
            "pioglitazone": "No",
            "rosiglitazone": "No",
            "acarbose": "No",
            "miglitol": "No",
            "troglitazone": "No",
            "tolazamide": "No",
            "examide": "No",
            "citoglipton": "No",
            "insulin": "No",
            "glyburide-metformin": "No",
            "glipizide-metformin": "No",
            "glimepiride-pioglitazone": "No",
            "metformin-rosiglitazone": "No",
            "metformin-pioglitazone": "No",
            "change": "No",
            "diabetesMed": "Yes",
        }
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "probability" in body
    assert body["prediction"] in {0, 1}
    assert 0.0 <= float(body["probability"]) <= 1.0


def test_predict_endpoint_passes_every_feature_to_model(monkeypatch):
    from app import main

    class CapturingModel:
        frame = None

        def predict(self, frame):
            self.frame = frame.copy()
            return [1]

        def predict_proba(self, frame):
            self.frame = frame.copy()
            return np.array([[0.25, 0.75]])

    model = CapturingModel()
    monkeypatch.setattr(main, "_load_model", lambda: model)
    feature_values = _feature_values()

    response = client.post("/predict", json={"features": feature_values})

    assert response.status_code == 200
    assert model.frame is not None
    assert set(model.frame.columns) == set(feature_values)
    actual_values = model.frame.iloc[0].to_dict()
    for field_name, expected_value in feature_values.items():
        assert actual_values[field_name] == expected_value


def test_locust_payload_matches_and_validates_all_prediction_fields():
    locust_path = Path(__file__).resolve().parents[1] / "locustfile.py"
    locust_tree = ast.parse(locust_path.read_text(encoding="utf-8"))
    payload_assignment = next(
        node
        for node in ast.walk(locust_tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "PREDICTION_PAYLOAD"
            for target in node.targets
        )
    )
    payload = ast.literal_eval(payload_assignment.value)

    feature_values = payload["features"]
    expected_names = {
        field.alias or field_name
        for field_name, field in Features.model_fields.items()
    }

    assert set(feature_values) == expected_names
    PredictionRequest.model_validate(payload)


def test_form_has_a_control_for_every_prediction_feature():
    response = client.get("/")
    assert response.status_code == 200
    parser = _FormControlNames()
    parser.feed(response.text)

    expected_names = {
        field.alias or field_name
        for field_name, field in Features.model_fields.items()
    }
    assert parser.names == expected_names
    assert parser.numeric_minimums
    assert all(minimum == "0" for minimum in parser.numeric_minimums.values())


def test_prediction_api_rejects_negative_numeric_features():
    negative_fields = [
        "encounter_id",
        "patient_nbr",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]

    for field_name in negative_fields:
        feature_values = _feature_values()
        feature_values[field_name] = -1

        response = client.post("/predict", json={"features": feature_values})

        assert response.status_code == 422, field_name


def test_health_endpoint_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics_include_request_audit_events():
    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert 'readmission_audit_events_total{event="started"}' in response.text
    assert 'readmission_audit_events_total{event="completed"}' in response.text
    assert "http_request_duration_highr_seconds_bucket" in response.text


def test_model_loader_caches_loaded_model(monkeypatch):
    from app import main

    main._load_model.cache_clear()
    loads = []
    sentinel = object()

    def fake_load(path):
        loads.append(path)
        return sentinel

    monkeypatch.setattr(main.joblib, "load", fake_load)
    try:
        assert main._load_model() is sentinel
        assert main._load_model() is sentinel
        assert len(loads) == 1
    finally:
        main._load_model.cache_clear()


def test_prediction_form_accepts_model_fields():
    response = client.post(
        "/predict_form",
        data={
            "race": "Caucasian",
            "gender": "Female",
            "age": "[50-60)",
            "weight": "[150-175)",
            "payer_code": "MC",
            "medical_specialty": "Cardiology",
            "tolazamide": "Steady",
            "glyburide-metformin": "Steady",
        },
    )

    assert response.status_code == 200
    assert "Prediction Result" in response.text
