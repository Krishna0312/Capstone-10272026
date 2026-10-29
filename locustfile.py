from locust import HttpUser, between, task


PREDICTION_PAYLOAD = {
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
        "payer_code": "MC",
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


class ReadmissionUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict_readmission(self):
        with self.client.post(
            "/predict",
            json=PREDICTION_PAYLOAD,
            name="/predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Prediction returned HTTP {response.status_code}")
                return

            try:
                body = response.json()
                prediction = body["prediction"]
                probability = float(body["probability"])
            except (KeyError, TypeError, ValueError) as error:
                response.failure(f"Invalid prediction response: {error}")
                return

            if prediction not in (0, 1) or not 0 <= probability <= 1:
                response.failure("Prediction response contains invalid values")
