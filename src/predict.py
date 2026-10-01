import mlflow
import pandas as pd
from mlflow import MlflowClient

from src.config import load_config

SAMPLE_CUSTOMER = pd.DataFrame([{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
    "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5, "TotalCharges": 190.0,
}])


def main():
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["tracking"]["uri"])

    name = cfg["registry"]["model_name"]
    alias = cfg["registry"]["alias"]
    model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    version = MlflowClient().get_model_version_by_alias(name, alias).version

    probability = model.predict_proba(SAMPLE_CUSTOMER)[0, 1]
    print(f"Using '{name}' version {version} ({alias})")
    print(f"Churn probability for this customer: {probability:.1%}")


if __name__ == "__main__":
    main()