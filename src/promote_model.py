import sys

import mlflow
from mlflow import MlflowClient

from src.config import load_config


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m src.promote_model <version>")
        raise SystemExit(1)

    version = sys.argv[1]
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["tracking"]["uri"])

    name = cfg["registry"]["model_name"]
    alias = cfg["registry"]["alias"]
    MlflowClient().set_registered_model_alias(name, alias, version)
    print(f"'{name}' version {version} is now '{alias}'")


if __name__ == "__main__":
    main()