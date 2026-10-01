import sys

import mlflow

from src.config import load_config


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m src.register_model <run_id>")
        raise SystemExit(1)

    run_id = sys.argv[1]
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["tracking"]["uri"])

    name = cfg["registry"]["model_name"]
    version = mlflow.register_model(f"runs:/{run_id}/model", name)
    print(f"Registered '{name}' version {version.version} from run {run_id[:8]}")


if __name__ == "__main__":
    main()