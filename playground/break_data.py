from src.data import clean, load_raw
from src.validation import validate

df = clean(load_raw("data/raw/churn.csv"))

# Sengaja merusak data dengan berbagai cara
df.loc[0, "MonthlyCharges"] = -50.0               # tagihan negatif
df.loc[1, "Contract"] = "Three year"               # kategori tak dikenal
df.loc[2, "Churn"] = "Maybe"                       # label aneh
df.loc[3, "customerID"] = df.loc[4, "customerID"]  # ID dobel
df.loc[0, "MultipleLines"] = "Yes"                 # tanpa telepon tapi punya banyak saluran
df.loc[0, "InternetService"] = "No"                # tanpa internet tapi punya layanan internet
df = df.drop(columns=["PaymentMethod"])            # kolom hilang


validate(df)