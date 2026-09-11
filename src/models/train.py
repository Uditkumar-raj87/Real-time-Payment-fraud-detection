"""Train a cost-sensitive LightGBM fraud model and log it to MLflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from src.data.ingest import clean_paysim
from src.features.engineering import TARGET, build_features, feature_columns, time_split
from src.models.rules import rules_score


def evaluate_scores(labels: pd.Series, scores: pd.Series, review_capacity: int = 1000) -> dict:
    """Report fraud-operations metrics without using raw accuracy."""
    ranked = pd.DataFrame({"label": labels.to_numpy(), "score": scores.to_numpy()}).sort_values(
        "score", ascending=False
    )
    reviewed = ranked.head(min(review_capacity, len(ranked)))
    threshold = float(reviewed.score.iloc[-1]) if not reviewed.empty else 1.0
    predicted = scores >= threshold
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "recall_at_review_capacity": float(reviewed.label.sum() / max(labels.sum(), 1)),
        "precision_at_k": float(reviewed.label.mean()) if not reviewed.empty else 0.0,
        "expected_fraud_loss_prevented": float(reviewed.label.sum()),
        "false_decline_rate": float(((predicted) & (labels == 0)).sum() / max((labels == 0).sum(), 1)),
        "review_threshold": threshold,
    }


def train(input_path: str | Path, artifact_path: str | Path = "artifacts/model.joblib") -> dict:
    raw = pd.read_parquet(input_path) if str(input_path).endswith("parquet") else pd.read_csv(input_path)
    frame = clean_paysim(raw)
    train_frame, validation_frame, test_frame = time_split(frame)
    train_x = build_features(train_frame, train_frame)
    validation_x = build_features(validation_frame, train_frame)
    test_x = build_features(test_frame, pd.concat([train_frame, validation_frame]))
    columns = feature_columns(train_x)
    categorical = [column for column in columns if train_x[column].dtype.name in {"string", "object"}]
    for data in [train_x, validation_x, test_x]:
        for column in categorical:
            data[column] = data[column].astype("category")
    model = LGBMClassifier(
        n_estimators=250, learning_rate=0.05, num_leaves=31, class_weight={0: 1, 1: 25},
        objective="binary", random_state=42, verbosity=-1,
    )
    with mlflow.start_run(run_name="paysim-lightgbm"):
        model.fit(train_x[columns], train_frame[TARGET])
        validation_score = model.predict_proba(validation_x[columns])[:, 1]
        metrics = evaluate_scores(validation_frame[TARGET], validation_score)
        mlflow.log_params({"model": "lightgbm", "class_weight_fraud": 25})
        mlflow.log_metrics({f"validation_{key}": value for key, value in metrics.items()})
    artifact = Path(artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": columns, "categorical": categorical}, artifact)
    test_score = model.predict_proba(test_x[columns])[:, 1]
    return {"validation_pr_auc": metrics["pr_auc"], "test": evaluate_scores(test_frame[TARGET], test_score)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--artifact", default="artifacts/model.joblib")
    args = parser.parse_args()
    print(train(args.input, args.artifact))