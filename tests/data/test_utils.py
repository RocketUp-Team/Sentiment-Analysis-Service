import os
import yaml
from src.data.utils import load_params


def test_load_params(tmp_path):
    config_file = tmp_path / "test_params.yaml"
    config_data = {"data": {"split_seed": 42}}
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    params = load_params(str(config_file))
    assert params["data"]["split_seed"] == 42
