import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

YES_NO = ["Yes", "No"]
ADDON = ["Yes", "No", "No internet service"]

churn_schema = DataFrameSchema(
    columns={
        # Identitas dan label
        "customerID": Column(nullable=False, unique=True),
        "Churn": Column(checks=Check.isin(YES_NO)),
        # Demografi
        "gender": Column(checks=Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "Partner": Column(checks=Check.isin(YES_NO)),
        "Dependents": Column(checks=Check.isin(YES_NO)),
        # Layanan
        "PhoneService": Column(checks=Check.isin(YES_NO)),
        "MultipleLines": Column(checks=Check.isin(["Yes", "No", "No phone service"])),
        "InternetService": Column(checks=Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": Column(checks=Check.isin(ADDON)),
        "OnlineBackup": Column(checks=Check.isin(ADDON)),
        "DeviceProtection": Column(checks=Check.isin(ADDON)),
        "TechSupport": Column(checks=Check.isin(ADDON)),
        "StreamingTV": Column(checks=Check.isin(ADDON)),
        "StreamingMovies": Column(checks=Check.isin(ADDON)),
        # Kontrak dan pembayaran
        "Contract": Column(checks=Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": Column(checks=Check.isin(YES_NO)),
        "PaymentMethod": Column(checks=Check.isin([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])),
        # Numerik
        "tenure": Column(int, Check.in_range(0, 120)),
        "MonthlyCharges": Column(float, Check.gt(0)),
        "TotalCharges": Column(float, Check.ge(0), nullable=False),
    },
    checks=[
        Check(lambda df: len(df) >= 1000,
              error="Jumlah baris terlalu sedikit (minimal 1000)"),
        Check(lambda df: (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all(),
              error="Pelanggan dengan tenure 0 harus punya TotalCharges 0"),
    ],
    strict=True,
)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validasi data bersih. Hentikan pipeline jika ada pelanggaran."""
    try:
        return churn_schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        cases = err.failure_cases[["column", "check", "failure_case"]]
        print(f"VALIDASI GAGAL: {len(cases)} pelanggaran ditemukan\n")
        print(cases.to_string(index=False, max_rows=30))
        raise SystemExit(1)