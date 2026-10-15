import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.inference.predict import predict_records, risk_band, run_batch_inference, to_frame
from src.models.evaluate_models import compute_metrics, evaluate_model, feature_importance
from src.models.model_pipelines import get_model_specs
from src.models.persistence import load_metadata, load_model
from src.models.train import run_training
from src.models.train_models import CV_SCORING, cv_summary, train_models
from src.tracking.select_champion import select_champion


def test_model_specs_contain_three_families():
    specs = get_model_specs()
    assert set(specs) == {"logistic_regression", "random_forest", "xgboost"}
    for pipeline, grid in specs.values():
        assert list(pipeline.named_steps) == ["preprocess", "model"]
        assert all(key.startswith("model__") for key in grid)


def test_quick_specs_have_single_candidate():
    for _, grid in get_model_specs(quick=True).values():
        assert all(len(values) == 1 for values in grid.values())


def test_train_and_cv_summary(clean_df):
    X, y = clean_df[FEATURE_COLUMNS], clean_df[TARGET_COLUMN]
    specs = {k: v for k, v in get_model_specs(quick=True).items() if k == "logistic_regression"}
    searches = train_models(specs, X, y, cv_folds=3, n_jobs=1)
    summary = cv_summary(searches["logistic_regression"])
    assert set(summary) == {f"cv_{m}_{s}" for m in CV_SCORING for s in ("mean", "std")}
    assert summary["cv_roc_auc_mean"] > 0.7  # synthetic data has a clear signal


def test_compute_metrics_perfect_prediction():
    metrics = compute_metrics([0, 1, 1, 0], [0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    assert metrics == {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "roc_auc": 1.0}


def test_evaluate_model_and_importance(trained_pipeline, clean_df):
    metrics, y_pred, y_proba = evaluate_model(trained_pipeline, clean_df[FEATURE_COLUMNS], clean_df[TARGET_COLUMN])
    assert all(0 <= v <= 1 for v in metrics.values())
    assert len(y_pred) == len(y_proba) == len(clean_df)
    importance = feature_importance(trained_pipeline)
    assert importance is not None and "hr_pct_of_max" in importance.index


def test_select_champion_uses_cv_auc_then_recall():
    results = {
        "a": {"cv_roc_auc_mean": 0.90, "test_recall": 0.80},
        "b": {"cv_roc_auc_mean": 0.9001, "test_recall": 0.95},
        "c": {"cv_roc_auc_mean": 0.85, "test_recall": 1.00},
    }
    assert select_champion(results)["model_name"] == "b"
    with pytest.raises(ValueError):
        select_champion({})


@pytest.mark.parametrize("p,band", [(0.1, "low"), (0.5, "moderate"), (0.9, "high")])
def test_risk_band(p, band):
    assert risk_band(p) == band


def test_predict_records_output_contract(trained_pipeline, sample_patient):
    output = predict_records(trained_pipeline, [sample_patient, {**sample_patient, "ca": None, "thal": None}])
    assert len(output) == 2
    for item in output:
        assert item["prediction"] in (0, 1)
        assert 0.5 <= item["confidence"] <= 1
        assert item["confidence"] == round(max(item["probability_heart_disease"],
                                               1 - item["probability_heart_disease"]), 4)


def test_to_frame_rejects_missing_features(sample_patient):
    sample_patient.pop("chol")
    with pytest.raises(ValueError, match="chol"):
        to_frame([sample_patient])


def test_batch_inference(tmp_path, model_files, clean_df):
    input_path = tmp_path / "in.csv"
    clean_df.to_csv(input_path, index=False)
    result = run_batch_inference(input_path, tmp_path / "out.csv", model_files[0])
    assert len(result) == len(clean_df)
    assert {"prediction", "confidence", "risk_level"} <= set(result.columns)
    assert (tmp_path / "out.csv").exists()


def test_run_training_end_to_end_without_mlflow(tmp_path, clean_df):
    outcome = run_training(
        quick=True, track=False, data=clean_df, model_path=tmp_path / "m.joblib",
        metadata_path=tmp_path / "meta.json", mlflow_model_dir=tmp_path / "mlflow_model",
        figures_dir=tmp_path / "fig", reports_dir=tmp_path, n_jobs=1,
    )
    assert outcome["champion"] in {"logistic_regression", "random_forest", "xgboost"}
    model = load_model(tmp_path / "m.joblib")
    metadata = load_metadata(tmp_path / "meta.json")
    assert metadata["model_family"] == outcome["champion"]
    assert (tmp_path / "mlflow_model" / "MLmodel").exists()
    assert (tmp_path / "model_comparison.csv").exists()
    proba = model.predict_proba(clean_df[FEATURE_COLUMNS].head(5))
    assert proba.shape == (5, 2)
    assert isinstance(pd.read_csv(tmp_path / "model_comparison.csv"), pd.DataFrame)
