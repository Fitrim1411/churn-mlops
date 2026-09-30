import joblib

from src.config import load_config
from src.data import clean, load_raw, split_xy
from src.validation import validate

ROW = 488  # one of the new customers with a blank TotalCharges

print("=== 1. Load config ===")
cfg = load_config()
print("Data path :", cfg["data"]["raw_path"])
print("Model     :", cfg["model"]["type"])

print("\n=== 2. Load raw data ===")
df = load_raw(cfg["data"]["raw_path"])
print("Rows:", len(df))
print(f"Row {ROW}, raw TotalCharges:", repr(df.loc[ROW, "TotalCharges"]))

print("\n=== 3. Clean ===")
df = clean(df)
print(f"Row {ROW}, TotalCharges after clean:", df.loc[ROW, "TotalCharges"])

print("\n=== 4. Validate ===")
df = validate(df, **cfg["validation"])
print("Rows passing validation:", len(df))

print("\n=== 5. Split features and label ===")
X, y = split_xy(df, cfg["data"]["target"], cfg["data"]["id_column"])
print("Features:", X.shape[1], "columns | Label: 1 column (Churn)")

print("\n=== 6. Score with the trained model ===")
model = joblib.load(cfg["output"]["model_path"])
probability = model.predict_proba(X.loc[[ROW]])[0, 1]
print(f"Churn probability for row {ROW}: {probability:.1%}")
print("Actual label:", df.loc[ROW, "Churn"])
