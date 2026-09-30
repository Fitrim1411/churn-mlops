import mlflow


def log_run(cfg: dict, metrics: dict) -> str:
    """Log one experiment to MLflow: parameters, metrics, and the config file."""
    mlflow.set_tracking_uri(cfg["tracking"]["uri"])
    mlflow.set_experiment(cfg["tracking"]["experiment"])

    with mlflow.start_run() as run:
        # Parameters: settings chosen before training
        mlflow.log_param("model_type", cfg["model"]["type"])
        for name, value in cfg["model"].get("params", {}).items():
            mlflow.log_param(f"model.{name}", value)
        mlflow.log_param("test_size", cfg["split"]["test_size"])
        mlflow.log_param("random_state", cfg["split"]["random_state"])

        # Metrics: results after training
        mlflow.log_metrics(metrics)

        # Artifact: an exact copy of the config file
        mlflow.log_artifact("configs/config.yaml")

    return run.info.run_id
