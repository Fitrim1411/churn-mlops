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
python -m src.validate_data   # check the data against the schema
python -m src.train           # train, evaluate, and log to MLflow
python -m src.predict         # score a sample customer
```

View experiments in the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```
