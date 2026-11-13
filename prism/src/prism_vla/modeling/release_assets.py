"""Verified, local architecture assets used only during bundle construction."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType

_ROOT = ContextVar("prism_release_asset_root", default=None)


@contextmanager
def architecture_assets(root):
    """Bind local architecture definitions without changing global HF loaders."""
    root = Path(root).resolve()
    token = _ROOT.set(root)
    try:
        yield
    finally:
        _ROOT.reset(token)


def asset_path(name):
    root = _ROOT.get()
    if root is None:
        raise RuntimeError("Construct PRISM through load_bundle(), which verifies local assets first")
    path = root / name
    if not path.is_dir():
        raise FileNotFoundError(f"Missing architecture asset directory: {name}")
    return path


def build_cradio_architecture():
    """Create the pinned visual graph; final weights come from the bundle."""
    path = asset_path("cradio")
    # Loading the local package directly retains upstream relative imports.
    # Transformers' dynamic cache only copies first-level local dependencies
    # and can miss transitive files such as utils.py in a clean installation.
    name = "prism_cradio_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(path)]
        package.__package__ = name
        sys.modules[name] = package
    previous = sys.dont_write_bytecode
    try:
        # Keep verified bundles immutable, including on read-only model mounts.
        sys.dont_write_bytecode = True
        module = importlib.import_module(name + ".hf_model")
    finally:
        sys.dont_write_bytecode = previous
    config = module.RADIOConfig.from_dict(json.loads((path / "config.json").read_text()))
    return module.RADIOModel(config)
