from src.config import load_config
from src.data import clean, load_raw
from src.validation import validate


def main():
    cfg = load_config()
    df = validate(clean(load_raw(cfg["data"]["raw_path"])), **cfg["validation"])
    print(f"Data is valid: {len(df)} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    main()
