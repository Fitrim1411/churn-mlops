from pathlib import Path

from src.config import load_config
from src.data import clean, load_raw
from src.validation import validate


def main():
    cfg = load_config()
    df = validate(clean(load_raw(cfg["data"]["raw_path"])), **cfg["validation"])

    out = Path(cfg["data"]["processed_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Prepared data: {len(df)} rows saved to {out}")


if __name__ == "__main__":
    main()