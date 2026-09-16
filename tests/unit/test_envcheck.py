"""A wrong-environment run must say so, not surface as an AttributeError."""
import sys

import pytest

from src.utils.envcheck import EnvironmentError_, assert_environment, describe


def test_current_environment_is_usable():
    assert assert_environment(strict=False) == []


def test_old_numpy_is_reported_clearly(monkeypatch):
    import numpy
    monkeypatch.setattr(numpy, "__version__", "1.26.4", raising=False)
    with pytest.raises(EnvironmentError_) as e:
        assert_environment()
    msg = str(e.value)
    assert "numpy 1.26.4" in msg
    assert "StringDType" in msg, "must name the symptom the user will see"
    assert "prefix" in msg, "must show which interpreter is running"


def test_conda_base_gets_an_activate_hint(monkeypatch):
    import numpy
    monkeypatch.setattr(numpy, "__version__", "1.26.4", raising=False)
    monkeypatch.setattr(sys, "prefix", "/home/u/anaconda3")
    with pytest.raises(EnvironmentError_, match="conda activate wsi"):
        assert_environment()


def test_describe_reports_the_interpreter():
    d = describe()
    assert "python" in d and "prefix" in d and "numpy" in d
