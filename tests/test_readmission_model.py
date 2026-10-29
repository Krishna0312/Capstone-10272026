import pandas as pd

import readmission_model
from readmission_model import make_readmission_target, train_and_select_models


def test_make_readmission_target_binary_mapping():
    df = pd.DataFrame({"readmitted": ["NO", ">30", "<30", "NO", ">30"]})
    result = make_readmission_target(df)
    assert result["readmission_30d"].tolist() == [0, 0, 1, 0, 0]


def test_train_and_select_models_returns_metrics():
    rng = pd.DataFrame(
        {
            "age": ["[0-10)", "[10-20)", "[20-30)", "[30-40)"] * 75,
            "gender": ["Female", "Male", "Female", "Male"] * 75,
            "race": ["Caucasian", "AfricanAmerican", "Caucasian", "Hispanic"] * 75,
            "time_in_hospital": [1, 2, 3, 4] * 75,
            "num_medications": [5, 8, 10, 12] * 75,
            "readmitted": ["NO", "NO", "<30", ">30"] * 75,
        }
    )

    summary = train_and_select_models(rng, target_col="readmitted", random_state=42)

    assert "best_model_name" in summary
    assert "best_model" in summary
    assert "test_metrics" in summary
    assert "train_metrics" in summary
    assert "validation_auc" in summary["test_metrics"]
    assert summary["best_model_name"] in summary["models"]


def test_run_training_pipeline_saves_unique_artifacts(tmp_path, monkeypatch):
    summary = {
        "best_model_name": "logistic_regression",
        "best_model": object(),
        "validation_auc": 0.8,
        "test_metrics": {"accuracy": 0.8},
    }
    saved_paths = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(readmission_model, "load_hospital_readmission_data", lambda data_dir: pd.DataFrame())
    monkeypatch.setattr(readmission_model, "describe_dataset", lambda df: {})
    monkeypatch.setattr(readmission_model, "plot_eda", lambda df: {})
    monkeypatch.setattr(readmission_model, "train_and_select_models", lambda df, target_col: summary)
    monkeypatch.setattr(
        readmission_model,
        "save_model_artifact",
        lambda model, output_path: saved_paths.append(output_path) or output_path,
    )
    monkeypatch.setattr(
        readmission_model,
        "save_evaluation_report",
        lambda model_summary, output_path: saved_paths.append(output_path) or output_path,
    )

    first_run = readmission_model.run_training_pipeline()
    second_run = readmission_model.run_training_pipeline()

    assert first_run["model_artifact"] != second_run["model_artifact"]
    assert first_run["report_path"] != second_run["report_path"]
    assert all(path.startswith("artifacts/readmission_model_") for path in saved_paths)
    assert first_run["model_artifact"] != "artifacts/final_readmission_model.joblib"
