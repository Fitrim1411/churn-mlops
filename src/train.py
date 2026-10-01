import json
from pathlib import Path

import joblib
import mlflow
from sklearn.metrics import (classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.data import clean, load_raw, split_xy
from src.features import build_preprocessor
from src.model import build_model
from src.tracking import data_is_dirty, git_is_dirty, log_run, setup_mlflow
from src.validation import validate


def train_and_evaluate(cfg: dict):
    """Load and validate data, train the model, and evaluate it on the test set."""
    # Data
    df = validate(clean(load_raw(cfg["data"]["raw_path"])), **cfg["validation"])
    X, y = split_xy(df, cfg["data"]["target"], cfg["data"]["id_column"])
    num_cols = cfg["features"]["numeric"]
    cat_cols = [c for c in X.columns if c not in num_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=y,
    )

    # Model
    model = build_model(cfg["model"], build_preprocessor(num_cols, cat_cols))
    model.fit(X_train, y_train)

    # Evaluate
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "precision_churn": round(precision_score(y_test, pred), 4),
        "recall_churn": round(recall_score(y_test, pred), 4),
        "f1_churn": round(f1_score(y_test, pred), 4),
    }
    print(f"Model: {cfg['model']['type']}")
    print(classification_report(y_test, pred, target_names=["Stay", "Churn"]))
    print("Metrics:", metrics)
    return model, metrics, X_train


def save_local(cfg: dict, model, metrics: dict) -> None:
    """Save the latest model and metrics to local files."""
    model_path = Path(cfg["output"]["model_path"])
    metrics_path = Path(cfg["output"]["metrics_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Model saved to {model_path}, metrics saved to {metrics_path}")


def main():
    cfg = load_config()
    setup_mlflow(cfg)

    with mlflow.start_run() as run:
        # Check for uncommitted changes BEFORE training writes any files
        mlflow.set_tag("git_dirty", str(git_is_dirty()))
        mlflow.set_tag("data_dirty", str(data_is_dirty(cfg["data"]["raw_path"])))

        model, metrics, X_train = train_and_evaluate(cfg)
        save_local(cfg, model, metrics)
        log_run(cfg, model, metrics, X_train)
        print(f"Run logged to MLflow, run_id: {run.info.run_id}")


if __name__ == "__main__":
    main()