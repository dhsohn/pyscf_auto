import json

import pytest

from run_opt_config import load_run_config, validate_run_config


@pytest.mark.parametrize("config_path", ["run_config_ase.json", "run_config_ts.json"])
def test_example_configs_pass_schema(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    validate_run_config(config)


def test_ts_placeholder_d3_command_rejected():
    with open("run_config_ts.json", "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    config["optimizer"]["ase"]["d3_command"] = "/path/to/dftd3"

    with pytest.raises(ValueError) as excinfo:
        validate_run_config(config)

    assert "placeholder '/path/to/dftd3'" in str(excinfo.value)


def test_concatenated_json_objects_report_extra_data(tmp_path):
    config_path = tmp_path / "run_config.json"
    config_path.write_text("{}{}", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_run_config(str(config_path))

    message = str(excinfo.value)
    assert "line 1 column 3" in message
    assert "파일에 JSON 객체가 두 개 이상 있음" in message
