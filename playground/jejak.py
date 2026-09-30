import joblib

from src.config import load_config
from src.data import clean, load_raw, split_xy
from src.validation import validate

print("=== 1. Baca config ===")
cfg = load_config()
print("Lokasi data :", cfg["data"]["raw_path"])
print("Model       :", cfg["model"]["type"])

print("\n=== 2. Baca data mentah ===")
df = load_raw(cfg["data"]["raw_path"])
print("Jumlah baris:", len(df))
print("Pelanggan baris 488, TotalCharges mentah:", repr(df.loc[488, "TotalCharges"]))

print("\n=== 3. Bersihkan ===")
df = clean(df)
print("Pelanggan baris 488, TotalCharges setelah clean:", df.loc[488, "TotalCharges"])

print("\n=== 4. Periksa (validasi) ===")
df = validate(df, **cfg["validation"])
print("Lolos validasi:", len(df), "baris")

print("\n=== 5. Pisahkan fitur dan label ===")
X, y = split_xy(df, cfg["data"]["target"], cfg["data"]["id_column"])
print("Fitur:", X.shape[1], "kolom | Label: 1 kolom (Churn)")

print("\n=== 6. Pakai model yang sudah dilatih ===")
model = joblib.load(cfg["output"]["model_path"])
peluang = model.predict_proba(X.loc[[488]])[0, 1]
print(f"Peluang pelanggan baris 488 churn: {peluang:.1%}")
print("Jawaban sebenarnya:", df.loc[488, "Churn"])