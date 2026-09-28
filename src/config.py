import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    """Baca file konfigurasi YAML dan kembalikan sebagai dictionary."""
    with open(path) as f:
        return yaml.safe_load(f)