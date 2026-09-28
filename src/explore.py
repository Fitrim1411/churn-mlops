import pandas as pd

df = pd.read_csv("data/raw/churn.csv")

print("Ukuran data (baris, kolom):", df.shape)
print("\nTipe data tiap kolom:")
print(df.dtypes)
print("\nDistribusi label Churn:")
print(df["Churn"].value_counts(normalize=True))
print("\nContoh nilai TotalCharges yang aneh:")
kosong = df[pd.to_numeric(df["TotalCharges"], errors="coerce").isna()]
print(kosong[["customerID", "tenure", "TotalCharges"]])