import json

import pytest

from pdft_core.run_opt_config import load_run_config, validate_run_config


@pytest.mark.parametrize("config_path", ["run_config.json"])
def test_example_configs_pass_schema(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    validate_run_config(config)


def test_concatenated_json_objects_report_extra_data(tmp_path):
    config_path = tmp_path / "run_config.json"
    config_path.write_text("{}{}", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_run_config(str(config_path))

    message = str(excinfo.value)
    assert "line 1 column 3" in message
    assert "파일에 JSON 객체가 두 개 이상 있음" in message


def _has_key(mapping, key):
    if isinstance(mapping, dict):
        if key in mapping:
            return True
        return any(_has_key(value, key) for value in mapping.values())
    if isinstance(mapping, list):
        return any(_has_key(item, key) for item in mapping)
    return False


def test_run_config_does_not_include_d3_command():
    with open("run_config.json", "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    assert not _has_key(config, "d3_command")
