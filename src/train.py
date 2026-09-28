import json
from pathlib import Path

import joblib
from sklearn.metrics import (classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.data import clean, load_raw, split_xy
from src.features import build_preprocessor
from src.model import build_model
from src.validation import validate


def main():
    cfg = load_config()

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

    # Evaluasi
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
    print("Metrik:", metrics)

    # Simpan model dan metrik
    model_path = Path(cfg["output"]["model_path"])
    metrics_path = Path(cfg["output"]["metrics_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Model tersimpan di {model_path}, metrik di {metrics_path}")


if __name__ == "__main__":
    main()