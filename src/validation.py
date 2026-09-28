from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

YES_NO = ["Yes", "No"]
ADDON = ["Yes", "No", "No internet service"]
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]


# ---------- Aturan antar-kolom (dijawab per baris) ----------

def tenure_nol_berarti_tagihan_nol(df):
    """Per baris: kalau tenure 0, TotalCharges harus 0."""
    return (df["tenure"] != 0) | (df["TotalCharges"] == 0)


def telepon_konsisten(df):
    """Per baris: tanpa telepon <=> MultipleLines 'No phone service'."""
    return (df["PhoneService"] == "No") == (df["MultipleLines"] == "No phone service")


def addon_konsisten(df):
    """Per baris: tanpa internet <=> semua addon 'No internet service'."""
    tanpa_internet = df["InternetService"] == "No"
    ok = pd.Series(True, index=df.index)
    for col in ADDON_COLS:
        ok = ok & (tanpa_internet == (df[col] == "No internet service"))
    return ok


# ---------- Skema ----------

churn_schema = DataFrameSchema(
    columns={
        "customerID": Column(nullable=False, unique=True),
        "Churn": Column(checks=Check.isin(YES_NO)),
        "gender": Column(checks=Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "Partner": Column(checks=Check.isin(YES_NO)),
        "Dependents": Column(checks=Check.isin(YES_NO)),
        "PhoneService": Column(checks=Check.isin(YES_NO)),
        "MultipleLines": Column(checks=Check.isin(["Yes", "No", "No phone service"])),
        "InternetService": Column(checks=Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": Column(checks=Check.isin(ADDON)),
        "OnlineBackup": Column(checks=Check.isin(ADDON)),
        "DeviceProtection": Column(checks=Check.isin(ADDON)),
        "TechSupport": Column(checks=Check.isin(ADDON)),
        "StreamingTV": Column(checks=Check.isin(ADDON)),
        "StreamingMovies": Column(checks=Check.isin(ADDON)),
        "Contract": Column(checks=Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": Column(checks=Check.isin(YES_NO)),
        "PaymentMethod": Column(checks=Check.isin([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])),
        "tenure": Column(int, Check.in_range(0, 120)),
        "MonthlyCharges": Column(float, Check.gt(0)),
        "TotalCharges": Column(float, Check.ge(0), nullable=False),
    },
    checks=[
        Check(lambda df: len(df) >= 1000,
              error="Jumlah baris terlalu sedikit (minimal 1000)"),
        Check(tenure_nol_berarti_tagihan_nol,
              error="Pelanggan dengan tenure 0 harus punya TotalCharges 0"),
        Check(telepon_konsisten,
              error="PhoneService 'No' harus berpasangan dengan MultipleLines 'No phone service'"),
        Check(addon_konsisten,
              error="InternetService 'No' harus berpasangan dengan semua addon 'No internet service'"),
    ],
    strict=True,
)


# ---------- Validasi dengan karantina ----------

def validate(df: pd.DataFrame, max_bad_ratio: float, quarantine_path: str) -> pd.DataFrame:
    """Validasi data. Error struktural -> berhenti. Error per baris -> karantina,
    selama jumlahnya tidak melebihi max_bad_ratio."""
    try:
        return churn_schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        cases = err.failure_cases

        # 1. Error struktural: tidak menunjuk baris tertentu -> wajib berhenti
        struktural = cases[cases["index"].isna()]
        if len(struktural) > 0:
            print("VALIDASI GAGAL (struktural), pipeline dihentikan:\n")
            print(struktural[["column", "check", "failure_case"]].to_string(index=False))
            raise SystemExit(1)

        # 2. Error per baris: hitung berapa baris yang bermasalah
        bad_idx = cases["index"].dropna().unique()
        bad_ratio = len(bad_idx) / len(df)
        print(f"PERINGATAN: {len(bad_idx)} baris bermasalah ({bad_ratio:.2%} dari data)")
        ringkasan = cases.dropna(subset=["index"]).groupby("check")["index"].nunique()
        print(ringkasan.rename("jumlah_baris").to_string())

        if bad_ratio > max_bad_ratio:
            print(f"\nMelebihi batas {max_bad_ratio:.2%}, pipeline dihentikan.")
            raise SystemExit(1)

        # 3. Masih di bawah batas: pisahkan ke karantina, lanjut dengan data bersih
        out = Path(quarantine_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.loc[bad_idx].to_csv(out, index=False)
        print(f"\nBaris bermasalah dikarantina ke {out}, pipeline lanjut.")
        return churn_schema.validate(df.drop(index=bad_idx), lazy=True)