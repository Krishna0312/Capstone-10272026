from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_predict_endpoint_returns_probability_and_label():
    payload = {
        "features": {
            "encounter_id": 123,
            "patient_nbr": 987,
            "race": "Caucasian",
            "gender": "Female",
            "age": "[50-60)",
            "weight": None,
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


def test_health_endpoint_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
