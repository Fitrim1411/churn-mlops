import subprocess
from pathlib import Path

import mlflow
import yaml
from mlflow.models import infer_signature


def setup_mlflow(cfg: dict) -> None:
    """Point MLflow to the tracking store and experiment from the config."""
    mlflow.set_tracking_uri(cfg["tracking"]["uri"])
    mlflow.set_experiment(cfg["tracking"]["experiment"])


def git_is_dirty() -> bool:
    """Return True if the source code in src/ has uncommitted changes."""
    status = subprocess.check_output(["git", "status", "--porcelain", "--", "src"], text=True)
    return bool(status.strip())


def log_run(cfg: dict, model, metrics: dict, X_sample) -> None:
    """Log everything needed to understand and reproduce this run
    to the currently active MLflow run."""
    # Parameters: settings chosen before training
    mlflow.log_param("model_type", cfg["model"]["type"])
    for name, value in cfg["model"].get("params", {}).items():
        mlflow.log_param(f"model.{name}", value)
    mlflow.log_param("test_size", cfg["split"]["test_size"])
    mlflow.log_param("random_state", cfg["split"]["random_state"])
    mlflow.log_param("data_version", data_version(cfg["data"]["raw_path"]))

    # Metrics: results after training
    mlflow.log_metrics(metrics)

    # Artifacts: exact config and library versions used
    mlflow.log_artifact("configs/config.yaml")
    mlflow.log_artifact("requirements.txt")

    # The trained model itself, with its input/output contract (signature)
    signature = infer_signature(X_sample, model.predict_proba(X_sample)[:, 1])
    mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=X_sample.head(3),
        skops_trusted_types=["numpy.dtype"],
    )

def data_version(data_path: str) -> str:
    """Return the DVC hash (md5) of a data file, or 'untracked' if DVC
    is not tracking it."""
    dvc_file = Path(f"{data_path}.dvc")
    if not dvc_file.exists():
        return "untracked"
    meta = yaml.safe_load(dvc_file.read_text())
    return meta["outs"][0]["md5"]