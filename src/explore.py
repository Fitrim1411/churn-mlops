import pandas as pd

df = pd.read_csv("data/raw/churn.csv")

print("Data shape (rows, columns):", df.shape)
print("\nColumn data types:")
print(df.dtypes)
print("\nChurn label distribution:")
print(df["Churn"].value_counts(normalize=True))
print("\nRows with non-numeric TotalCharges:")
invalid = df[pd.to_numeric(df["TotalCharges"], errors="coerce").isna()]
print(invalid[["customerID", "tenure", "TotalCharges"]])
