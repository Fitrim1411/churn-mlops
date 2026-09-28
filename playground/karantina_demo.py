import sys

from src.config import load_config
from src.data import clean, load_raw
from src.validation import validate

cfg = load_config()
df = clean(load_raw(cfg["data"]["raw_path"]))

# Berapa baris yang mau dirusak? Bisa diatur dari terminal, default 3
n_rusak = int(sys.argv[1]) if len(sys.argv) > 1 else 3
df.loc[df.index[:n_rusak], "Contract"] = "Three year"

hasil = validate(df, **cfg["validation"])
print(f"\nData yang lanjut ke training: {len(hasil)} baris")