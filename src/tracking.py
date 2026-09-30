import mlflow


def log_run(cfg: dict, metrics: dict) -> str:
    """Catat satu eksperimen ke MLflow: pengaturan, hasil, dan file config."""
    mlflow.set_tracking_uri(cfg["tracking"]["uri"])
    mlflow.set_experiment(cfg["tracking"]["experiment"])

    with mlflow.start_run() as run:
        # Pengaturan (parameters)
        mlflow.log_param("model_type", cfg["model"]["type"])
        for nama, nilai in cfg["model"].get("params", {}).items():
            mlflow.log_param(f"model.{nama}", nilai)
        mlflow.log_param("test_size", cfg["split"]["test_size"])
        mlflow.log_param("random_state", cfg["split"]["random_state"])

        # Hasil (metrics)
        mlflow.log_metrics(metrics)

        # Salinan file config apa adanya (artifact)
        mlflow.log_artifact("configs/config.yaml")

    return run.info.run_id