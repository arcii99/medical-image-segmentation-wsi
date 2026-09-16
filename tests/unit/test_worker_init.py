"""worker_init is shared by both dataset backends and must not assume either."""
import numpy as np
import pytest

from src.data.dataset import worker_init
from src.data.transforms import TrainTransform


class _FakeInfo:
    def __init__(self, ds):
        self.dataset = ds


class _SlideLike:
    def __init__(self):
        self._readers = {("pid", "a"): object()}
        self._geoms = {"a": object()}
        self._pid = -1
        self.rng = np.random.default_rng(0)
        self.tf = TrainTransform(0)


class _PatchSetLike:
    """No _readers, no _geoms -- exactly what broke in Colab."""
    def __init__(self):
        self.rng = np.random.default_rng(0)
        self.tf = TrainTransform(0)


class _Minimal:
    """Not even an rng or a transform."""


def _run(ds, worker_id, monkeypatch):
    """Drive worker_init without torch installed.

    worker_init imports torch lazily, so a stub module in sys.modules is
    enough. This keeps the test runnable in environments without torch --
    which is where most of this project's checks run.
    """
    import sys
    import types
    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch.utils")
    data = types.ModuleType("torch.utils.data")
    data.get_worker_info = lambda: _FakeInfo(ds)
    utils.data = data
    torch.utils = utils
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", data)
    worker_init(worker_id)


def test_slide_backed_state_is_reset(monkeypatch):
    ds = _SlideLike()
    _run(ds, 3, monkeypatch)
    assert ds._readers == {} and ds._geoms == {}
    assert ds._pid != -1


def test_patchset_dataset_does_not_crash(monkeypatch):
    ds = _PatchSetLike()
    _run(ds, 3, monkeypatch)        # would raise AttributeError before the fix


def test_dataset_without_rng_or_transform(monkeypatch):
    _run(_Minimal(), 0, monkeypatch)


def test_workers_get_different_augmentation_streams(monkeypatch):
    a, b = _PatchSetLike(), _PatchSetLike()
    _run(a, 0, monkeypatch)
    _run(b, 1, monkeypatch)
    da = a.tf.rng.integers(0, 1_000_000, 8).tolist()
    db = b.tf.rng.integers(0, 1_000_000, 8).tolist()
    assert da != db, "workers must not share an augmentation seed"
