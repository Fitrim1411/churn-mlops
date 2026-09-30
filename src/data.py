import urllib.request
from pathlib import Path

import pandas as pd


def download_data(url: str, out_path: str) -> Path:
    """Download the raw dataset from url to out_path."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out)
    return out


def load_raw(path: str) -> pd.DataFrame:
    """Read the raw data from a CSV file."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data. Returns a new DataFrame; the input is not modified."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # New customers (tenure 0) have never been billed, so their total charges are 0
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0
    return df


def split_xy(df: pd.DataFrame, target: str, id_column: str):
    """Split the data into features (X) and label (y)."""
    y = (df[target] == "Yes").astype(int)
    X = df.drop(columns=[target, id_column])
    return X, y
