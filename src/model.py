from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def build_model(model_cfg: dict, preprocessor) -> Pipeline:
    """Bangun pipeline lengkap: preprocessing + model sesuai config."""
    model_type = model_cfg["type"]
    if model_type not in MODELS:
        raise ValueError(f"Model '{model_type}' tidak dikenal. Pilihan: {list(MODELS)}")
    clf = MODELS[model_type](**model_cfg.get("params", {}))
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])