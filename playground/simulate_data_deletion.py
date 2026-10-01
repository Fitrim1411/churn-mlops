import pandas as pd

from src.config import load_config

cfg = load_config()
path = cfg["data"]["raw_path"]

# Read everything as text so the file keeps its original raw format
df = pd.read_csv(path, dtype=str, keep_default_na=False)

# Simulate 500 customers asking for their personal data to be deleted
deleted = df.sample(n=500, random_state=7)
df = df.drop(deleted.index)

df.to_csv(path, index=False)
print(f"Deleted {len(deleted)} customers, {len(df)} remain")