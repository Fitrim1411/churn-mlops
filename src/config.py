import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load the YAML config file and return it as a dictionary."""
    with open(path) as f:
        return yaml.safe_load(f)
