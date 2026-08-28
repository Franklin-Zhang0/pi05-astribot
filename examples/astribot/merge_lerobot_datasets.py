"""Merge two LeRobot datasets into one.

This script merges two LeRobot datasets that have compatible structures.
It reads episodes from both source datasets and writes them to a new merged dataset.

Usage:

python openpi/examples/astribot/merge_lerobot_datasets.py \
    --dataset1-repo-id local/mm22_5_4_lerobot \
    --dataset2-repo-id local/converted_astribot_dataset \
    --output-repo-id local/merged_dataset
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset, LeRobotDatasetMetadata

try:
    import tqdm
except Exception:
    tqdm = None


def load_episode_metadata(dataset_path: Path) -> tuple[dict, list[dict], list[dict]]:
    """Load info.json, episodes.jsonl, and tasks.jsonl from a LeRobot dataset."""
    meta_dir = dataset_path / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"

    info = {}
    if info_path.exists():
        info = json.loads(info_path.read_text())

    episodes = []
    if episodes_path.exists():
        for line in episodes_path.read_text().strip().split("\n"):
            if line.strip():
                episodes.append(json.loads(line))

    tasks = []
    if tasks_path.exists():
        for line in tasks_path.read_text().strip().split("\n"):
            if line.strip():
                tasks.append(json.loads(line))

    return info, episodes, tasks


def copy_episode_frames(
    source_path: Path,
    target_dataset: LeRobotDataset,
    ep_idx: int,
    ep_meta: dict,
) -> int:
    """Copy frames from a source episode to target dataset. Returns frames copied."""
    chunk_idx = ep_idx // 1000
    parquet_path = source_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"

    if not parquet_path.exists():
        print(f"Warning: Episode {ep_idx} parquet not found: {parquet_path}")
        return 0

    try:
        import pandas as pd

        df = pd.read_parquet(parquet_path)

        # Extract task string
        ep_tasks = ep_meta.get("tasks", [])
        task_str = ep_tasks[0] if ep_tasks else "unknown task"

        frames_copied = 0
        # Copy frames
        for frame_idx in range(len(df)):
            frame = {}
            for col in df.columns:
                value = df.iloc[frame_idx][col]
                # Handle different data types
                if isinstance(value, np.ndarray):
                    frame[col] = value.copy()
                elif isinstance(value, (list, tuple)):
                    frame[col] = np.array(value)
                elif pd.notna(value):
                    frame[col] = value

            # Ensure task is set
            if "task" not in frame and "task_index" not in frame:
                frame["task"] = task_str

            target_dataset.add_frame(frame)
            frames_copied += 1

        return frames_copied
    except Exception as e:
        print(f"Error copying episode {ep_idx}: {e}")
        import traceback

        traceback.print_exc()
        return 0


def merge_datasets(
    dataset1_repo_id: str,
    dataset2_repo_id: str,
    output_repo_id: str,
    max_episodes_per_dataset: int | None = None,
) -> None:
    """Merge two LeRobot datasets into one."""
    dataset1_path = HF_LEROBOT_HOME / dataset1_repo_id
    dataset2_path = HF_LEROBOT_HOME / dataset2_repo_id
    output_path = HF_LEROBOT_HOME / output_repo_id

    if not dataset1_path.exists():
        raise ValueError(f"Dataset 1 not found: {dataset1_path}")
    if not dataset2_path.exists():
        raise ValueError(f"Dataset 2 not found: {dataset2_path}")

    # Load metadata from both datasets
    info1, episodes1, tasks1 = load_episode_metadata(dataset1_path)
    info2, episodes2, tasks2 = load_episode_metadata(dataset2_path)

    print(f"Dataset 1: {len(episodes1)} episodes")
    print(f"Dataset 2: {len(episodes2)} episodes")

    # Verify feature compatibility
    features1 = info1.get("features", {})
    features2 = info2.get("features", {})

    # Use features from dataset1 as base (assuming they're compatible)
    # Merge unique features if needed
    merged_features = features1.copy()
    for key, value in features2.items():
        if key not in merged_features:
            merged_features[key] = value
        elif merged_features[key] != value:
            print(f"Warning: Feature {key} differs between datasets, using dataset1 version")
            print(f"  Dataset 1: {merged_features[key]}")
            print(f"  Dataset 2: {value}")

    # Clean output directory
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create merged dataset
    merged_dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        robot_type=info1.get("robot_type", info2.get("robot_type", "astribot")),
        fps=info1.get("fps", info2.get("fps", 30)),
        features=merged_features,
        use_videos=True,  # Assuming video format
        image_writer_threads=10,
        image_writer_processes=32,
    )

    # Copy episodes from dataset1
    print("\nCopying episodes from dataset 1...")
    episodes_to_copy_1 = episodes1
    if max_episodes_per_dataset:
        episodes_to_copy_1 = episodes1[:max_episodes_per_dataset]

    iterator1 = episodes_to_copy_1
    if tqdm is not None:
        iterator1 = tqdm.tqdm(episodes_to_copy_1, desc="Dataset 1", unit="episode")

    total_frames_1 = 0
    for ep_meta in iterator1:
        ep_idx = ep_meta["episode_index"]
        frames = copy_episode_frames(dataset1_path, merged_dataset, ep_idx, ep_meta)
        if frames > 0:
            merged_dataset.save_episode()
            total_frames_1 += frames

    # Copy episodes from dataset2
    print("\nCopying episodes from dataset 2...")
    episodes_to_copy_2 = episodes2
    if max_episodes_per_dataset:
        episodes_to_copy_2 = episodes2[:max_episodes_per_dataset]

    iterator2 = episodes_to_copy_2
    if tqdm is not None:
        iterator2 = tqdm.tqdm(episodes_to_copy_2, desc="Dataset 2", unit="episode")

    total_frames_2 = 0
    for ep_meta in iterator2:
        ep_idx = ep_meta["episode_index"]
        frames = copy_episode_frames(dataset2_path, merged_dataset, ep_idx, ep_meta)
        if frames > 0:
            merged_dataset.save_episode()
            total_frames_2 += frames

    # Consolidate if available
    if hasattr(merged_dataset, "consolidate"):
        merged_dataset.consolidate()

    print(f"\nMerged dataset written to: {output_path}")
    print(f"Total episodes: {len(episodes_to_copy_1) + len(episodes_to_copy_2)}")
    print(f"Total frames: {total_frames_1 + total_frames_2} (Dataset1: {total_frames_1}, Dataset2: {total_frames_2})")


@tyro.dataclass
class Args:
    dataset1_repo_id: str
    dataset2_repo_id: str
    output_repo_id: str
    max_episodes_per_dataset: int | None = None


def main(args: Args) -> None:
    merge_datasets(
        dataset1_repo_id=args.dataset1_repo_id,
        dataset2_repo_id=args.dataset2_repo_id,
        output_repo_id=args.output_repo_id,
        max_episodes_per_dataset=args.max_episodes_per_dataset,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
