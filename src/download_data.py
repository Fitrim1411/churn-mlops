from src.config import load_config
from src.data import download_data


def main():
    cfg = load_config()
    out = download_data(cfg["data"]["url"], cfg["data"]["raw_path"])
    print(f"Data saved to {out}")


if __name__ == "__main__":
    main()
