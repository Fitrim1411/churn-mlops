from src.config import load_config
from src.data import clean, load_raw
from src.validation import validate

cfg = load_config()
df = clean(load_raw(cfg["data"]["raw_path"]))

# Sengaja merusak data dengan berbagai cara
df.loc[0, "MonthlyCharges"] = -50.0               # tagihan negatif
df.loc[1, "Contract"] = "Three year"               # kategori tak dikenal
df.loc[2, "Churn"] = "Maybe"                       # label aneh
df.loc[3, "customerID"] = df.loc[4, "customerID"]  # ID dobel
df.loc[5, "PhoneService"] = "No"                   # tanpa telepon...
df.loc[5, "MultipleLines"] = "Yes"                 # ...tapi punya banyak saluran
#df = df.drop(columns=["PaymentMethod"])            # kolom hilang

validate(df, **cfg["validation"])