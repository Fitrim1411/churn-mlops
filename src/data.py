import urllib.request
from pathlib import Path

import pandas as pd


def download_data(url: str, out_path: str) -> Path:
    """Unduh data mentah dari URL ke out_path."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out)
    return out


def load_raw(path: str) -> pd.DataFrame:
    """Baca data mentah dari CSV."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan data mentah. Tidak mengubah DataFrame aslinya."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Pelanggan baru (tenure 0) belum pernah ditagih, jadi total tagihannya 0
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0
    return df


def split_xy(df: pd.DataFrame, target: str, id_column: str):
    """Pisahkan fitur (X) dan label (y)."""
    y = (df[target] == "Yes").astype(int)
    X = df.drop(columns=[target, id_column])
    return X, y