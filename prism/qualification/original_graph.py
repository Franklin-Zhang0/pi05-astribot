"""Developer-only reference factory; never imported by the serving runtime.

Set PRISM_PARITY_REFERENCE_SOURCE to the original ``src`` directory and
PRISM_PARITY_REFERENCE_CHECKPOINT to the original completed checkpoint. These
paths are deliberately not stored in the public package or capture receipts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def load_bundle(directory, device="cuda:0"):
    """Construct the original graph, then restore the exported final weights."""
    source = Path(os.environ["PRISM_PARITY_REFERENCE_SOURCE"]).resolve()
    checkpoint = Path(os.environ["PRISM_PARITY_REFERENCE_CHECKPOINT"]).resolve()
    if not (source / "prism_vla/modeling/model.py").is_file():
        raise ValueError("Reference source does not contain the original model")
    if any(key == "prism_vla" or key.startswith("prism_vla.") for key in sys.modules):
        raise RuntimeError("Reference source must be selected before importing prism_vla")
    sys.path.insert(0, str(source))
    from prism_engram.bundle import LoadedBundle, restore_model_weights, verify_bundle

    path, manifest, _, normalization = verify_bundle(directory)
    revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={source.parent}", "-C", str(source.parent), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != manifest["source_revision"]:
        raise ValueError("Reference source revision differs from the exported checkpoint contract")
    config = json.loads((checkpoint / "config.json").read_text())
    import prism_vla
    from prism_vla.modeling.config import PrismVLAConfig
    from prism_vla.modeling.model import PrismVLA

    if not Path(prism_vla.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Reference model resolved to the standalone package")
    # Existing base/tokenizer/vision assets must be available locally. Only
    # original construction occurs here; all final tensors are replaced next.
    model = PrismVLA(PrismVLAConfig.from_dict(config))
    restore_model_weights(model, path / "pytorch_model.bin", device=device)
    model.qwen.config.use_cache = True
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path / "assets/qwen"), local_files_only=True)
    return LoadedBundle(path, model, tokenizer, config, normalization, manifest)
