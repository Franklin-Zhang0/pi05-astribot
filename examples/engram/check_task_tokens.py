"""Check all selected task prompts against every discrete π0.5 state bin."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.models import pi0_config
from openpi.models.tokenizer import PaligemmaTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    spec = json.loads(Path(args.manifest).read_text())
    max_len = pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=64).max_token_len
    tokenizer = PaligemmaTokenizer(max_len=10_000)
    worst = 0
    worst_prompt = ""
    for source in spec["sources"]:
        for row in pq.read_table(Path(source["root"]) / "meta/tasks.parquet").to_pylist():
            prompt = row["task"]
            if not source["has_gripper"]:
                prompt = "End-effector state unavailable. " + prompt
            # Every possible tokenized state value is -1..255. The tokenizer
            # separates state values with spaces, so the longest repeated value
            # bounds any 22-component combination of those values.
            for bin_value in range(-1, 256):
                state = np.full(22, -2.0 if bin_value == -1 else (bin_value + 0.5) / 128.0 - 1.0)
                _, mask = tokenizer.tokenize(prompt, state)
                length = int(mask.sum())
                if length > worst:
                    worst, worst_prompt = length, prompt
    print(f"max_tokens={worst} configured_max={max_len} prompt={worst_prompt!r}")
    if worst > max_len:
        raise ValueError("A task/state prompt would be truncated")


if __name__ == "__main__":
    main()
