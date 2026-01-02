import json

import pytest

import run_opt_dispersion


@pytest.mark.parametrize(
    "d3_params, expected_key, expected_value",
    [
        ({"parameters": {"s8": 1.2}}, "s8", 1.2),
        ({"params_tweaks": {"a1": 0.3}}, "a1", 0.3),
        ({"damping": {"parameters": {"a2": 4.5}}}, "a2", 4.5),
    ],
)
def test_parse_dispersion_settings_dftd3_param_shapes(
    monkeypatch, d3_params, expected_key, expected_value
):
    class DummyDFTD3:
        def __init__(self, method=None, damping=None, params_tweaks=None, **kwargs):
            pass

    def fake_loader(prefer_backend=None):
        return DummyDFTD3, "dftd3"

    monkeypatch.setattr(run_opt_dispersion, "load_d3_calculator", fake_loader)

    result = run_opt_dispersion.parse_dispersion_settings(
        "d3bj", xc="b3lyp", d3_params=d3_params, prefer_d3_backend="dftd3"
    )

    settings = result["settings"]
    assert settings["params_tweaks"] == {expected_key: expected_value}


def test_parse_dispersion_settings_dftd3_backend(monkeypatch):
    class DummyDFTD3:
        def __init__(self, method=None, damping=None, params_tweaks=None, **kwargs):
            pass

    def fake_loader(prefer_backend=None):
        return DummyDFTD3, "dftd3"

    monkeypatch.setattr(run_opt_dispersion, "load_d3_calculator", fake_loader)

    result = run_opt_dispersion.parse_dispersion_settings(
        "d3bj", xc="b3lyp", prefer_d3_backend="dftd3"
    )

    settings = result["settings"]
    assert result["backend"] == "d3"
    assert settings["damping"] == "d3bj"
    assert settings["method"] == "b3lyp"


def test_parse_dispersion_settings_dftd3_damping_tweaks(monkeypatch):
    class DummyDFTD3:
        def __init__(self, method=None, damping=None, params_tweaks=None, **kwargs):
            pass

    def fake_loader(prefer_backend=None):
        return DummyDFTD3, "dftd3"

    monkeypatch.setattr(run_opt_dispersion, "load_d3_calculator", fake_loader)

    d3_params = {
        "damping": {"s6": 1.0, "s8": 1.2, "a1": 0.3, "a2": 4.5},
    }

    result = run_opt_dispersion.parse_dispersion_settings(
        "d3bj", xc="b3lyp", d3_params=d3_params, prefer_d3_backend="dftd3"
    )

    settings = result["settings"]
    assert settings["params_tweaks"] == {"s6": 1.0, "s8": 1.2, "a1": 0.3, "a2": 4.5}


@pytest.mark.parametrize("config_path", ["run_config.json"])
def test_parse_dispersion_settings_templates(monkeypatch, config_path):
    class DummyDFTD3:
        def __init__(self, method=None, damping=None, params_tweaks=None, **kwargs):
            pass

    def fake_loader(prefer_backend=None):
        return DummyDFTD3, "dftd3"

    monkeypatch.setattr(run_opt_dispersion, "load_d3_calculator", fake_loader)

    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    dispersion = config["dispersion"]
    xc = config["xc"]
    optimizer_ase = config["optimizer"]["ase"]
    d3_params = optimizer_ase["d3_params"]

    result = run_opt_dispersion.parse_dispersion_settings(
        dispersion,
        xc=xc,
        d3_params=d3_params,
        prefer_d3_backend=None,
    )

    settings = result["settings"]
    assert settings["damping"] == "d3bj"
    assert settings["params_tweaks"] == d3_params["damping"]
