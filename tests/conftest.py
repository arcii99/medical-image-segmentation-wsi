import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIX = ROOT / "tests" / "fixtures" / "mini"


@pytest.fixture(scope="session")
def fixtures():
    if not (FIX / "synth_tumor.tif").exists():
        pytest.skip("run: python scripts/make_fixtures.py --out tests/fixtures/mini")
    return FIX
