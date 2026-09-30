import sys

from src.config import load_config
from src.data import clean, load_raw
from src.validation import validate

cfg = load_config()
df = clean(load_raw(cfg["data"]["raw_path"]))

# How many rows to corrupt? Pass a number from the terminal (default: 3)
n_bad = int(sys.argv[1]) if len(sys.argv) > 1 else 3
df.loc[df.index[:n_bad], "Contract"] = "Three year"

result = validate(df, **cfg["validation"])
print(f"\nRows passed on to training: {len(result)}")
