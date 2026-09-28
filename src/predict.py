import joblib
import pandas as pd

from src.config import load_config

cfg = load_config()
model = joblib.load(cfg["output"]["model_path"])

pelanggan_baru = pd.DataFrame([{
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

peluang = model.predict_proba(pelanggan_baru)[0, 1]
print(f"Peluang pelanggan ini churn: {peluang:.1%}")