"""Bound mixed JSON/plain prompts against all possible discrete 64D state bins."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies.engram_multi_policy import format_prompt
from openpi.shared.engram_multi import EMBODIMENTS
from openpi.shared.engram_multi import ROUTE_NAMES
from openpi.shared.engram_multi import route_id


def audit_prompts(catalog, max_len):
    tokenizer = PaligemmaTokenizer(max_len=10000, preserve_text=True, reject_truncation=True)
    worst = {"tokens": 0}
    prompts = 0
    for index, spec in enumerate(EMBODIMENTS):
        for instruction in catalog[ROUTE_NAMES[index]]:
            for available in (False, True):
                for structured in (False, True):
                    text = format_prompt(
                        instruction, spec.robot_type, spec.end_effector_type, available, structured=structured
                    )
                    # Discrete values are separated by spaces. Repeating every possible
                    # -1..255 value bounds all combinations of the 64 state components.
                    for value in range(-1, 256):
                        state = np.full(64, -2.0 if value == -1 else (value + 0.5) / 128 - 1)
                        _, mask = tokenizer.tokenize(text, state)
                        length = int(mask.sum())
                        if length > worst["tokens"]:
                            worst = {
                                "tokens": length,
                                "route": spec.name,
                                "prompt": text,
                                "structured": structured,
                                "available": available,
                                "state_bin": value,
                            }
                    prompts += 1
    report = {
        "max_token_len": max_len,
        "prompt_variants": prompts,
        "worst_case": worst,
        "truncation_free": worst["tokens"] <= max_len,
    }
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report["truncation_free"]:
        raise ValueError("Mixed prompt/state conditioning exceeds the model token budget")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Saved typed-stats provenance (retained prompts)")
    parser.add_argument(
        "--source-manifest", action="store_true", help="Conservatively include every selected-source task"
    )
    parser.add_argument("--max-token-len", type=int, default=352)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text())
    if args.source_manifest:
        catalog = {name: set() for name in ROUTE_NAMES}
        for source in spec["sources"]:
            if not source["selected_episode_indices"]:
                continue
            name = ROUTE_NAMES[route_id(source["robot_type"], source["end_effector_type"])]
            catalog[name].update(
                row["task"] for row in pq.read_table(Path(source["root"]) / "meta/tasks.parquet").to_pylist()
            )
    else:
        catalog = spec["prompts_by_route"]
    audit_prompts(catalog, args.max_token_len)


if __name__ == "__main__":
    main()
