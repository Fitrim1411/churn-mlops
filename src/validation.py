from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

YES_NO = ["Yes", "No"]
ADDON = ["Yes", "No", "No internet service"]
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]


# ---------- Cross-column rules (evaluated per row) ----------

def zero_tenure_has_zero_charges(df):
    """Per row: if tenure is 0, TotalCharges must be 0."""
    return (df["tenure"] != 0) | (df["TotalCharges"] == 0)


def phone_lines_consistent(df):
    """Per row: no phone service <=> MultipleLines is 'No phone service'."""
    return (df["PhoneService"] == "No") == (df["MultipleLines"] == "No phone service")


def addons_consistent(df):
    """Per row: no internet service <=> every add-on is 'No internet service'."""
    no_internet = df["InternetService"] == "No"
    ok = pd.Series(True, index=df.index)
    for col in ADDON_COLS:
        ok = ok & (no_internet == (df[col] == "No internet service"))
    return ok


# ---------- Schema ----------

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
              error="Too few rows (minimum 1000)"),
        Check(zero_tenure_has_zero_charges,
              error="Customers with tenure 0 must have TotalCharges 0"),
        Check(phone_lines_consistent,
              error="PhoneService 'No' must pair with MultipleLines 'No phone service'"),
        Check(addons_consistent,
              error="InternetService 'No' must pair with 'No internet service' in all add-ons"),
    ],
    strict=True,
)


# ---------- Validation with quarantine ----------

def validate(df: pd.DataFrame, max_bad_ratio: float, quarantine_path: str) -> pd.DataFrame:
    """Validate the data. Structural errors stop the pipeline. Row-level errors
    are quarantined, as long as they stay within max_bad_ratio."""
    try:
        return churn_schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        cases = err.failure_cases

        # 1. Structural errors are not tied to any row -> always stop
        structural = cases[cases["index"].isna()]
        if len(structural) > 0:
            print("VALIDATION FAILED (structural), stopping the pipeline:\n")
            print(structural[["column", "check", "failure_case"]].to_string(index=False))
            raise SystemExit(1)

        # 2. Row-level errors: count how many rows are affected
        bad_idx = cases["index"].dropna().unique()
        bad_ratio = len(bad_idx) / len(df)
        print(f"WARNING: {len(bad_idx)} invalid rows ({bad_ratio:.2%} of the data)")
        summary = cases.dropna(subset=["index"]).groupby("check")["index"].nunique()
        print(summary.rename("rows").to_string())

        if bad_ratio > max_bad_ratio:
            print(f"\nExceeds the {max_bad_ratio:.2%} limit, stopping the pipeline.")
            raise SystemExit(1)

        # 3. Within the limit: quarantine the invalid rows, continue with the rest
        out = Path(quarantine_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.loc[bad_idx].to_csv(out, index=False)
        print(f"\nInvalid rows quarantined to {out}, pipeline continues.")
        return churn_schema.validate(df.drop(index=bad_idx), lazy=True)
