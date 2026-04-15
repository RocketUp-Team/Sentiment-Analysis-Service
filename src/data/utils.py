from typing import Any, Dict

import yaml


def load_params(path: str = "params.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
