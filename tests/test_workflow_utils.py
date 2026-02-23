import pytest

from execution.utils import _normalize_frequency_dispersion_mode, _xc_includes_dispersion


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "numerical"),
        ("numerical", "numerical"),
        ("fd", "numerical"),
        ("energy", "energy"),
        ("energy_only", "energy"),
        ("none", "none"),
        ("off", "none"),
    ],
)
def test_normalize_frequency_dispersion_mode(value, expected):
    assert _normalize_frequency_dispersion_mode(value) == expected


def test_normalize_frequency_dispersion_mode_rejects_unknown():
    with pytest.raises(ValueError):
        _normalize_frequency_dispersion_mode("analytic")


@pytest.mark.parametrize(
    "value, expected",
    [
        ("b3lyp-d3", True),
        ("wb97x-v", True),
        ("b97m-v", True),
        ("r2scan-v", True),
        ("b97m-vv10", True),
        ("\u03c9B97X-V", True),
        ("pbe", False),
    ],
)
def test_xc_includes_dispersion(value, expected):
    assert _xc_includes_dispersion(value) is expected
