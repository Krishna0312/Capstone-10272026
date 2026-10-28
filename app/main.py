import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "final_readmission_model.joblib"


class Features(BaseModel):
    encounter_id: Optional[int] = Field(None, description="Unique identifier for the encounter")
    patient_nbr: Optional[int] = Field(None, description="Unique identifier for the patient")
    race: Optional[str] = Field(None, description="Patient's race")
    gender: Optional[str] = Field(None, description="Patient's gender")
    age: Optional[str] = Field(None, description="Patient's age group")
    weight: Optional[float] = Field(None, description="Patient's weight")
    admission_type_id: Optional[int] = Field(None, description="Type of admission")
    discharge_disposition_id: Optional[int] = Field(None, description="Discharge status")
    admission_source_id: Optional[int] = Field(None, description="Source of admission")
    time_in_hospital: Optional[int] = Field(None, description="Number of days spent in the hospital")
    payer_code: Optional[str] = Field(None, description="Insurance payer code")
    medical_specialty: Optional[str] = Field(None, description="Specialty of the admitting physician")
    num_lab_procedures: Optional[int] = Field(None, description="Number of lab procedures performed")
    num_procedures: Optional[int] = Field(None, description="Number of procedures performed")
    num_medications: Optional[int] = Field(None, description="Number of medications prescribed")
    number_outpatient: Optional[int] = Field(None, description="Number of outpatient visits")
    number_emergency: Optional[int] = Field(None, description="Number of emergency visits")
    number_inpatient: Optional[int] = Field(None, description="Number of inpatient visits")
    diag_1: Optional[str] = Field(None, description="Primary diagnosis (ICD code)")
    diag_2: Optional[str] = Field(None, description="Secondary diagnosis (ICD code)")
    diag_3: Optional[str] = Field(None, description="Tertiary diagnosis (ICD code)")
    number_diagnoses: Optional[int] = Field(None, description="Total number of diagnoses")
    max_glu_serum: Optional[str] = Field(None, description="Maximum glucose serum test result")
    A1Cresult: Optional[str] = Field(None, description="Hemoglobin A1c test result")
    metformin: Optional[str] = Field(None, description="Metformin medication status")
    repaglinide: Optional[str] = Field(None, description="Repaglinide medication status")
    nateglinide: Optional[str] = Field(None, description="Nateglinide medication status")
    chlorpropamide: Optional[str] = Field(None, description="Chlorpropamide medication status")
    glimepiride: Optional[str] = Field(None, description="Glimepiride medication status")
    acetohexamide: Optional[str] = Field(None, description="Acetohexamide medication status")
    glipizide: Optional[str] = Field(None, description="Glipizide medication status")
    glyburide: Optional[str] = Field(None, description="Glyburide medication status")
    tolbutamide: Optional[str] = Field(None, description="Tolbutamide medication status")
    pioglitazone: Optional[str] = Field(None, description="Pioglitazone medication status")
    rosiglitazone: Optional[str] = Field(None, description="Rosiglitazone medication status")
    acarbose: Optional[str] = Field(None, description="Acarbose medication status")
    miglitol: Optional[str] = Field(None, description="Miglitol medication status")
    troglitazone: Optional[str] = Field(None, description="Troglitazone medication status")
    tolazamide: Optional[str] = Field(None, description="Tolazamide medication status")
    examide: Optional[str] = Field(None, description="Examide medication status")
    citoglipton: Optional[str] = Field(None, description="Citoglipton medication status")
    insulin: Optional[str] = Field(None, description="Insulin medication status")
    glyburide_metformin: Optional[str] = Field(None, description="Glyburide-metformin medication status")
    glipizide_metformin: Optional[str] = Field(None, description="Glipizide-metformin medication status")
    glimepiride_pioglitazone: Optional[str] = Field(None, description="Glimepiride-pioglitazone medication status")
    metformin_rosiglitazone: Optional[str] = Field(None, description="Metformin-rosiglitazone medication status")
    metformin_pioglitazone: Optional[str] = Field(None, description="Metformin-pioglitazone medication status")
    change: Optional[str] = Field(None, description="Change in diabetes medication")
    diabetesMed: Optional[str] = Field(None, description="Whether the patient is on diabetes medication")

class PredictionRequest(BaseModel):
    features: Features

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_name: str
    status: str = "ok"


app = FastAPI(title="Readmission Risk API", version="1.0.0")

@app.get("/", response_class=HTMLResponse)
async def form_page():
    """Render an HTML form for user input with improved styling."""
    return """
    <html>
        <head>
            <title>Readmission Risk Prediction</title>
            
        <style>
            * {
                box-sizing: border-box;
            }

            body {
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #eef2f7, #dce6f1);
                margin: 0;
                padding: 35px 15px;
                color: #263445;
            }

            .container {
                background: #ffffff;
                padding: 35px;
                border-radius: 16px;
                box-shadow: 0 10px 35px rgba(31, 55, 80, 0.12);
                width: 100%;
                max-width: 850px;
                margin: 0 auto;
            }

            h1 {
                text-align: center;
                color: #173b65;
                font-size: 28px;
                margin: 0 0 12px;
                line-height: 1.4;
            }

            form {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 18px 22px;
                margin-top: 30px;
            }

            label {
                margin-bottom: 8px;
                font-weight: 600;
                font-size: 14px;
                color: #34465a;
            }

            input[type="text"],
            input[type="number"],
            select {
                width: 100%;
                padding: 12px 14px;
                border: 1px solid #d5deea;
                border-radius: 8px;
                font-size: 14px;
                font-family: inherit;
                color: #263445;
                background-color: #f9fbfd;
                transition: border-color 0.2s, box-shadow 0.2s;
            }

            input:hover,
            select:hover {
                border-color: #9aafc7;
            }

            input:focus,
            select:focus {
                outline: none;
                border-color: #2878c8;
                background-color: #ffffff;
                box-shadow: 0 0 0 3px rgba(40, 120, 200, 0.12);
            }

            input::placeholder {
                color: #9aa7b6;
            }

            button {
                grid-column: 1 / -1;
                margin-top: 12px;
                padding: 15px 20px;
                background: linear-gradient(135deg, #2878c8, #18579a);
                color: #ffffff;
                border: none;
                border-radius: 9px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                box-shadow: 0 4px 10px rgba(40, 120, 200, 0.2);
                transition: transform 0.2s, box-shadow 0.2s;
            }

            button:hover {
                transform: translateY(-1px);
                box-shadow: 0 6px 15px rgba(40, 120, 200, 0.3);
            }

            button:active {
                transform: translateY(0);
            }

            .footer {
                text-align: center;
                margin-top: 25px;
                font-size: 12px;
                color: #8794a5;
            }

            @media (max-width: 600px) {
                body {
                    padding: 15px 10px;
                }

                .container {
                    padding: 22px 17px;
                    border-radius: 12px;
                }

                h1 {
                    font-size: 23px;
                }

                form {
                    grid-template-columns: 1fr;
                    gap: 15px;
                    margin-top: 22px;
                }

                button {
                    grid-column: 1;
                }
            }
        </style>

        </head>
        <body>
            <div class="container">
                <h1>Readmission Risk Prediction</h1>
                <form action="/predict_form" method="post">
                    <label for="encounter_id">Encounter ID:</label>
                    <input type="number" id="encounter_id" name="encounter_id">

                    <label for="patient_nbr">Patient Number:</label>
                    <input type="number" id="patient_nbr" name="patient_nbr">

                    <label for="race">Race:</label>
                    <select id="race" name="race" required>
                        <option value="" disabled selected>Select Race</option>
                        <option value="Caucasian">Caucasian</option>
                        <option value="AfricanAmerican">AfricanAmerican</option>
                        <option value="Asian">Asian</option>
                        <option value="Hispanic">Hispanic</option>
                        <option value="Other">Other</option>
                    </select>

                    <label for="gender">Gender:</label>
                    <select id="gender" name="gender" required>
                        <option value="" disabled selected>Select Gender</option>
                        <option value="Female">Female</option>
                        <option value="Male">Male</option>
                        <option value="Unknown/Invalid">Unknown/Invalid</option>
                    </select>

                    <label for="age">Age:</label>
                    <select id="age" name="age" required>
                        <option value="" disabled selected>Select Age Range</option>
                        <option value="[0-10)">[0-10)</option>
                        <option value="[10-20)">[10-20)</option>
                        <option value="[20-30)">[20-30)</option>
                        <option value="[30-40)">[30-40)</option>
                        <option value="[40-50)">[40-50)</option>
                        <option value="[50-60)">[50-60)</option>
                        <option value="[60-70)">[60-70)</option>
                        <option value="[70-80)">[70-80)</option>
                        <option value="[80-90)">[80-90)</option>
                        <option value="[90-100)">[90-100)</option>
                    </select>

                    <label for="weight">Weight:</label>
                    <select id="weight" name="weight" required>
                        <option value="" disabled selected>Select Weight Range</option>
                        <option value="[0-25)">[0-25)</option>
                        <option value="[25-50)">[25-50)</option>
                        <option value="[50-75)">[50-75)</option>
                        <option value="[75-100)">[75-100)</option>
                        <option value="[100-125)">[100-125)</option>
                        <option value="[125-150)">[125-150)</option>
                        <option value="[150-175)">[150-175)</option>
                        <option value="[175-200)">[175-200)</option>
                        <option value=">200">>200</option>
                        <option value="">Unknown</option>
                    </select>

                    <label for="admission_type_id">Admission Type ID:</label>
                    <input type="number" id="admission_type_id" name="admission_type_id">

                    <label for="discharge_disposition_id">Discharge Disposition ID:</label>
                    <input type="number" id="discharge_disposition_id" name="discharge_disposition_id">

                    <label for="admission_source_id">Admission Source ID:</label>
                    <input type="number" id="admission_source_id" name="admission_source_id">

                    <label for="time_in_hospital">Time in Hospital:</label>
                    <input type="number" id="time_in_hospital" name="time_in_hospital">
                    
                    <label for="payer_code">Payer Code:</label>
                    <input type="text" id="payer_code" name="payer_code">
                    
                    <label for="medical_specialty">Medical Speciality:</label>
                    <input type="text" id="medical_specialty" name="medical_specialty">

                    <label for="num_lab_procedures">Number of Lab Procedures:</label>
                    <input type="number" id="num_lab_procedures" name="num_lab_procedures">

                    <label for="num_procedures">Number of Procedures:</label>
                    <input type="number" id="num_procedures" name="num_procedures">

                    <label for="num_medications">Number of Medications:</label>
                    <input type="number" id="num_medications" name="num_medications">

                    <label for="number_outpatient">Number of Outpatient Visits:</label>
                    <input type="number" id="number_outpatient" name="number_outpatient">

                    <label for="number_emergency">Number of Emergency Visits:</label>
                    <input type="number" id="number_emergency" name="number_emergency">

                    <label for="number_inpatient">Number of Inpatient Visits:</label>
                    <input type="number" id="number_inpatient" name="number_inpatient">

                    <label for="diag_1">Primary Diagnosis (ICD Code):</label>
                    <input type="text" id="diag_1" name="diag_1">

                    <label for="diag_2">Secondary Diagnosis (ICD Code):</label>
                    <input type="text" id="diag_2" name="diag_2">

                    <label for="diag_3">Tertiary Diagnosis (ICD Code):</label>
                    <input type="text" id="diag_3" name="diag_3">

                    <label for="number_diagnoses">Total Number of Diagnoses:</label>
                    <input type="number" id="number_diagnoses" name="number_diagnoses">

                    <label for="max_glu_serum">Maximum Glucose Serum:</label>
                    <input type="text" id="max_glu_serum" name="max_glu_serum">

                    <label for="A1Cresult">Hemoglobin A1c Result:</label>
                    <input type="text" id="A1Cresult" name="A1Cresult">

                    <label for="metformin">Metformin:</label>
                    <select id="metformin" name="metformin">
                    <option value="No">No</option>
                    <option value="Steady">Steady</option>
                    <option value="Up">Up</option>
                    <option value="Down">Down</option>
                    </select>

                    <label for="repaglinide">Repaglinide:</label>
                    <select id="repaglinide" name="repaglinide">
                    <option value="No">No</option>
                    <option value="Steady">Steady</option>
                    <option value="Up">Up</option>
                    <option value="Down">Down</option>
                    </select>

                    <label for="nateglinide">Nateglinide:</label>
                    <select id="nateglinide" name="nateglinide">
                    <option value="No">No</option>
                    <option value="Steady">Steady</option>
                    <option value="Up">Up</option>
                    <option value="Down">Down</option>
                    </select>

                    <label for="chlorpropamide">Chlorpropamide:</label>
                    <select id="chlorpropamide" name="chlorpropamide">
                    <option value="No">No</option>
                    <option value="Steady">Steady</option>
                    <option value="Up">Up</option>
                    <option value="Down">Down</option>
                    </select>

                    <label for="glimepiride">Glimepiride:</label>
                    <select id="glimepiride" name="glimepiride">
                    <option value="No">No</option>
                    <option value="Steady">Steady</option>
                    <option value="Up">Up</option>
                    <option value="Down">Down</option>
                    </select>

                    <label for="acetohexamide">Acetohexamide:</label>
                    <select id="acetohexamide" name="acetohexamide">
                    <option value="No">No</option>
                    <option value="Steady">Steady</option>
                    </select>

                    <label for="glipizide">Glipizide:</label>
                    <select id="glipizide" name="glipizide">
                    <option value="No">No</option>
                    <option value="Steady">Steady</option>
                    <option value="Up">Up</option>
                    <option value="Down">Down</option>
                    </select>

                    <label for="glyburide">Glyburide:</label>
                    <select id="glyburide" name="glyburide">
                    <option value="No">No</option>
                    <option value="Steady">Steady</option>
                    <option value="Up">Up</option>
                    <option value="Down">Down</option>
                    </select>

                    <label for="tolbutamide">Tolbutamide:</label>
                    <select id="tolbutamide" name="tolbutamide">
                    <option value="No">No</option>
                    <option value="Steady">Steady</option>
                    </select>

                    <label for="pioglitazone">Pioglitazone:</label>
                    <select id="pioglitazone" name="pioglitazone">
                    <option value="No">No</option>
                    <option value="Steady">Steady</option>
                    <option value="Up">Up</option>
                    <option value="Down">Down</option>
                    </select>

                    <label for="rosiglitazone">Rosiglitazone:</label>
                    <select id="rosiglitazone" name="rosiglitazone">
                    <option value="No">No</option>
                    <option value="Steady">Steady</option>
                    <option value="Up">Up</option>
                    <option value="Down">Down</option>
                    </select>

                    <label for="acarbose">Acarbose:</label>
                    <select id="acarbose" name="acarbose">
                    <option value="No">No</option>
                    <option value="Steady">Steady</option>
                    <option value="Up">Up</option>
                    <option value="Down">Down</option>
                    </select>

                    <label for="miglitol">Miglitol:</label>
                    <select id="miglitol" name="miglitol">
                    <option value="No">No</option>
                    <option value="Steady">Steady</option>
                    </select>

                    <label for="troglitazone">Troglitazone:</label>
                    <select id="troglitazone" name="troglitazone">
                    <option value="No">No</option>
                    <option value="Steady">Steady</option>
                    </select>

                    <label for="tolazamide">Tolazamide:</label>
                    <select id="tolazamide" name="tolazamide">
                    <option value="No" selected>No</option>
                    <option value="Steady">Steady</option>
                    <option value="Up">Up</option>
                    <option value="Down">Down</option>
                    </select>

                    <label for="examide">Examide:</label>
                    <select id="examide" name="examide">
                    <option value="No" selected>No</option>
                    </select>

                    <label for="citoglipton">Citoglipton:</label>
                    <select id="citoglipton" name="citoglipton">
                    <option value="No" selected>No</option>
                    </select>

                    <label for="insulin">Insulin:</label>
                    <select id="insulin" name="insulin">
                    <option value="No" selected>No</option>
                    <option value="Steady">Steady</option>
                    <option value="Up">Up</option>
                    <option value="Down">Down</option>
                    <option value="Up/Down">Up/Down</option>
                    </select>

                    <label for="glyburide-metformin">Glyburide-Metformin:</label>
                    <select id="glyburide-metformin" name="glyburide-metformin">
                    <option value="No" selected>No</option>
                    <option value="Steady">Steady</option>
                    <option value="Up">Up</option>
                    <option value="Down">Down</option>
                    </select>

                    <label for="glipizide-metformin">Glipizide-Metformin:</label>
                    <select id="glipizide-metformin" name="glipizide-metformin">
                    <option value="No" selected>No</option>
                    <option value="Steady">Steady</option>
                    </select>

                    <label for="glimepiride-pioglitazone">Glimepiride-Pioglitazone:</label>
                    <select id="glimepiride-pioglitazone" name="glimepiride-pioglitazone">
                    <option value="No" selected>No</option>
                    <option value="Steady">Steady</option>
                    </select>

                    <label for="metformin-rosiglitazone">Metformin-Rosiglitazone:</label>
                    <select id="metformin-rosiglitazone" name="metformin-rosiglitazone">
                    <option value="No" selected>No</option>
                    <option value="Steady">Steady</option>
                    </select>

                    <label for="metformin-pioglitazone">Metformin-Pioglitazone:</label>
                    <select id="metformin-pioglitazone" name="metformin-pioglitazone">
                    <option value="No" selected>No</option>
                    <option value="Steady">Steady</option>
                    </select>

                    <label for="change">Change in Diabetes Medication:</label>
                    <select id="change" name="change">
                    <option value="No" selected>No</option>
                    <option value="Ch">Ch</option>
                    </select>

                    <label for="diabetesMed">On Diabetes Medication:</label>
                    <select id="diabetesMed" name="diabetesMed">
                    <option value="No" selected>No</option>
                    <option value="Yes">Yes</option>
                    </select>

                    <button type="submit">Submit</button>
                </form>
                <div class="footer">
                    <p>Powered by FastAPI</p>
                </div>
            </div>
        </body>
    </html>
    """


@app.post("/predict_form", response_class=HTMLResponse)
async def predict_from_form(
    encounter_id: Optional[int] = Form(None),
    patient_nbr: Optional[int] = Form(None),
    race: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    age: Optional[str] = Form(None),
    weight: Optional[float] = Form(None),
    admission_type_id: Optional[int] = Form(None),
    discharge_disposition_id: Optional[int] = Form(None),
    admission_source_id: Optional[int] = Form(None),
    time_in_hospital: Optional[int] = Form(None),
    num_lab_procedures: Optional[int] = Form(None),
    num_procedures: Optional[int] = Form(None),
    num_medications: Optional[int] = Form(None),
    number_outpatient: Optional[int] = Form(None),
    number_emergency: Optional[int] = Form(None),
    number_inpatient: Optional[int] = Form(None),
    diag_1: Optional[str] = Form(None),
    diag_2: Optional[str] = Form(None),
    diag_3: Optional[str] = Form(None),
    number_diagnoses: Optional[int] = Form(None),
    max_glu_serum: Optional[str] = Form(None),
    A1Cresult: Optional[str] = Form(None),
    metformin: Optional[str] = Form(None),
    repaglinide: Optional[str] = Form(None),
    nateglinide: Optional[str] = Form(None),
    chlorpropamide: Optional[str] = Form(None),
    glimepiride: Optional[str] = Form(None),
    acetohexamide: Optional[str] = Form(None),
    glipizide: Optional[str] = Form(None),
    glyburide: Optional[str] = Form(None),
    tolbutamide: Optional[str] = Form(None),
    pioglitazone: Optional[str] = Form(None),
    rosiglitazone: Optional[str] = Form(None),
    acarbose: Optional[str] = Form(None),
    miglitol: Optional[str] = Form(None),
    troglitazone: Optional[str] = Form(None),
    change: Optional[str] = Form(None),
    diabetesMed: Optional[str] = Form(None),
):
    """Handle form submission and make predictions."""
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact not found. Train the model first.")

    model = joblib.load(MODEL_PATH)
    
    # Define the required columns (based on the model's training schema)
    required_columns = [
        "encounter_id", "patient_nbr", "race", "gender", "age", "weight",
        "admission_type_id", "discharge_disposition_id", "admission_source_id",
        "time_in_hospital", "payer_code", "medical_specialty", "num_lab_procedures",
        "num_procedures", "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses",
        "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide",
        "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
        "troglitazone", "tolazamide", "examide", "citoglipton", "insulin",
        "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
        "metformin-rosiglitazone", "metformin-pioglitazone", "change", "diabetesMed"
    ]
    # Create a DataFrame from the form data
    feature_record = pd.DataFrame([{
        "encounter_id": encounter_id,
        "patient_nbr": patient_nbr,
        "race": race,
        "gender": gender,
        "age": age,
        "weight": weight,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "diag_1": diag_1,
        "diag_2": diag_2,
        "diag_3": diag_3,
        "number_diagnoses": number_diagnoses,
        "max_glu_serum": max_glu_serum,
        "A1Cresult": A1Cresult,
        "metformin": metformin,
        "repaglinide": repaglinide,
        "nateglinide": nateglinide,
        "chlorpropamide": chlorpropamide,
        "glimepiride": glimepiride,
        "acetohexamide": acetohexamide,
        "glipizide": glipizide,
        "glyburide": glyburide,
        "tolbutamide": tolbutamide,
        "pioglitazone": pioglitazone,
        "rosiglitazone": rosiglitazone,
        "acarbose": acarbose,
        "miglitol": miglitol,
        "troglitazone": troglitazone,
        "change": change,
        "diabetesMed": diabetesMed,
    }])

    
    # Ensure all required columns are present, set missing ones to None
    for col in required_columns:
        if col not in feature_record.columns:
            feature_record[col] = None
    
    # Replace string "None" or "nan" with Python None        
    feature_record = feature_record.replace({"None": None, "nan": None})
    
    # Ensure all column names are strings
    feature_record.columns = feature_record.columns.astype(str)


    # Make predictions
    prediction = int(model.predict(feature_record)[0])
    probability = float(model.predict_proba(feature_record)[0, 1])

    # Return the result as an HTML response
    return f"""
    <html>
        <head>
            <title>Prediction Result</title>
        </head>
        <body>
            <h1>Prediction Result</h1>
            <p><strong>Prediction:</strong> {prediction}</p>
            <p><strong>Probability:</strong> {probability:.2f}</p>
        </body>
    </html>
    """


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(payload: PredictionRequest):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact not found. Train the model first.")

    model = joblib.load(MODEL_PATH)
    
    # Define the required columns (based on the model's training schema)
    required_columns = [
        "encounter_id", "patient_nbr", "race", "gender", "age", "weight",
        "admission_type_id", "discharge_disposition_id", "admission_source_id",
        "time_in_hospital", "payer_code", "medical_specialty", "num_lab_procedures",
        "num_procedures", "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses",
        "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide",
        "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
        "troglitazone", "tolazamide", "examide", "citoglipton", "insulin",
        "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
        "metformin-rosiglitazone", "metformin-pioglitazone", "change", "diabetesMed"
    ]

    feature_record = pd.DataFrame([payload.features])
    
    # Ensure all required columns are present, set missing ones to None
    for col in required_columns:
        if col not in feature_record.columns:
            feature_record[col] = None
    
    # Replace string "None" or "nan" with Python None        
    feature_record = feature_record.replace({"None": None, "nan": None})
    
    # Ensure all column names are strings
    feature_record.columns = feature_record.columns.astype(str)


    # Make predictions
    prediction = int(model.predict(feature_record)[0])
    probability = float(model.predict_proba(feature_record)[0, 1])

    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        model_name="readmission_risk_model",
    )
