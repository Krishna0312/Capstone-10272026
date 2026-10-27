import pandas as pd

from readmission_model import make_readmission_target, train_and_select_models


def test_make_readmission_target_binary_mapping():
    df = pd.DataFrame({"readmitted": ["NO", ">30", "<30", "NO", ">30"]})
    result = make_readmission_target(df)
    assert result["readmission_30d"].tolist() == [0, 1, 1, 0, 1]


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
