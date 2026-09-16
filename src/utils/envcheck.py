"""Fail fast, and legibly, when the interpreter is not the project's.

The failure this exists for: running a script from the wrong conda
environment. The symptom was

    AttributeError: module 'numpy.dtypes' has no attribute 'StringDType'

raised from inside tiffslide, wrapped in a SlideReadError, wrapped again in a
ProcessPoolExecutor traceback. Nothing in that chain says "wrong environment",
and the actual cause -- numpy 1.26 in Anaconda's base env, where the project
requires numpy 2 -- is several inferences away from the message.
"""
from __future__ import annotations

import sys
from pathlib import Path

MIN_NUMPY = (2, 0)


class EnvironmentError_(RuntimeError):
    pass


def describe() -> str:
    try:
        import numpy
        nv = numpy.__version__
    except Exception:                                    # noqa: BLE001
        nv = "missing"
    return (f"python  {sys.version.split()[0]}\n"
            f"prefix  {sys.prefix}\n"
            f"numpy   {nv}")


def assert_environment(strict: bool = True) -> list[str]:
    """Check the interpreter can actually run this project. Returns problems."""
    problems: list[str] = []
    try:
        import numpy
        ver = tuple(int(x) for x in numpy.__version__.split(".")[:2])
        if ver < MIN_NUMPY:
            problems.append(
                f"numpy {numpy.__version__} is installed, but this project "
                f"needs >= {'.'.join(map(str, MIN_NUMPY))}. tiffslide will "
                "fail with \"module 'numpy.dtypes' has no attribute "
                "'StringDType'\".")
    except ImportError:
        problems.append("numpy is not installed")

    for mod, why in (("tiffslide", "reading slides"),
                     ("shapely", "annotation geometry"),
                     ("cv2", "image operations")):
        try:
            __import__(mod)
        except Exception as e:                           # noqa: BLE001
            problems.append(f"{mod} unusable ({type(e).__name__}: {e}) "
                            f"-- needed for {why}")

    if problems and strict:
        env = Path(sys.prefix).name
        hint = ("\n\nThe interpreter is:\n  " + describe().replace("\n", "\n  ")
                + f"\n\nThis looks like the '{env}' environment."
                + ("\nDid you forget:  conda activate wsi"
                   if env in ("base", "anaconda3", "miniconda3") else ""))
        raise EnvironmentError_(
            "environment is not usable for this project:\n  - "
            + "\n  - ".join(problems) + hint)
    return problems
