# Churn MLOps

An end-to-end MLOps learning project: predicting customer churn for a telco company,
built step by step from a single script into a production-style ML pipeline.

## Setup

```bash
python3 -m venv churn_mlops.venv
source churn_mlops.venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python -m src.download_data   # download the raw dataset
dvc repro                     # run the pipeline: prepare -> train
python -m src.predict         # score a sample customer with the champion model
```

Inspect the pipeline and compare experiments:

```bash
dvc dag            # show the pipeline graph
dvc params diff    # parameter changes vs the last commit
dvc metrics diff   # metric changes vs the last commit
```

View experiments in the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```
