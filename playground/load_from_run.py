import sys

import mlflow

from src.config import load_config
from src.predict import SAMPLE_CUSTOMER

cfg = load_config()
mlflow.set_tracking_uri(cfg["tracking"]["uri"])

run_id = sys.argv[1]
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

probability = model.predict_proba(SAMPLE_CUSTOMER)[0, 1]
print(f"Model from run {run_id[:8]}: churn probability {probability:.1%}")