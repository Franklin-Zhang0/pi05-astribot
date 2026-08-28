"""Preload the standard-library typing module before spawned data workers start.

OpenCV temporarily places its ``cv2`` package directory near the front of
``sys.path`` while importing the native extension. A Python multiprocessing
worker can start during that window and resolve ``typing`` to ``cv2/typing``.
That accidental import also pulls in GUI OpenCV and fails in the headless ACP
image because ``libGL.so.1`` is absent. Caching stdlib ``typing`` during Python
startup makes subsequent imports deterministic without reducing data-loader
parallelism.
"""

import importlib.util
import sys

_typing_path = f"{sys.base_prefix}/lib/python{sys.version_info.major}.{sys.version_info.minor}/typing.py"
_typing_spec = importlib.util.spec_from_file_location("typing", _typing_path)
if _typing_spec is None or _typing_spec.loader is None:
    raise RuntimeError(f"Unable to load standard-library typing from {_typing_path}")
_stdlib_typing = importlib.util.module_from_spec(_typing_spec)
sys.modules["typing"] = _stdlib_typing
_typing_spec.loader.exec_module(_stdlib_typing)
