#!/usr/bin/env python3
"""
LeRobot Dataset Inspector

Inspects LeRobot datasets (v2.0 format) stored locally or on HuggingFace Hub.
Provides a human-readable report of dataset structure, episodes, frames, modalities,
video streams, and optional statistics without loading everything into memory.

Usage Examples:
    # Inspect local dataset (default path)
    python openpi/astribot_scripts/inspect_lerobot_dataset.py

    # Inspect local dataset with custom path
    python openpi/astribot_scripts/inspect_lerobot_dataset.py --root ./my_dataset

    # Inspect with sampling statistics
    python openpi/astribot_scripts/inspect_lerobot_dataset.py --max_episodes_per_subdataset 5 --max_steps_per_episode 100

    # Inspect HuggingFace dataset (requires lerobot/datasets installed)
    python openpi/astribot_scripts/inspect_lerobot_dataset.py --hf_repo openpi/airdust_lerobot

    # Run as module
    python -m openpi.astribot_scripts.inspect_lerobot_dataset --help
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings

# Optional dependencies (graceful fallback)
try:
    import pyarrow.parquet as pq

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class Colors:
    """ANSI color codes for terminal output"""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}\n")


def print_section(text: str):
    """Print formatted section"""
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}{text}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'-' * len(text)}{Colors.ENDC}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}ERROR: {text}{Colors.ENDC}", file=sys.stderr)


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}WARNING: {text}{Colors.ENDC}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}{text}{Colors.ENDC}")


def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file with error handling"""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print_warning(f"Failed to load {path}: {e}")
        return None


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file with error handling"""
    try:
        episodes = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
        return episodes
    except Exception as e:
        print_warning(f"Failed to load {path}: {e}")
        return []


def detect_dataset_format(root: Path) -> str:
    """Detect the dataset format based on directory structure"""
    if not root.exists():
        return "NOT_FOUND"

    # Check for LeRobot v2.0 format indicators
    has_meta = (root / "meta").exists()
    has_info = (root / "meta" / "info.json").exists()
    has_episodes = (root / "meta" / "episodes.jsonl").exists()

    if has_meta and has_info and has_episodes:
        return "LEROBOT_V2"

    # Check for subdatasets (parent directory containing multiple lerobot datasets)
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    has_subdatasets = any((d / "meta" / "info.json").exists() for d in subdirs)

    if has_subdatasets:
        return "LEROBOT_V2_MULTI"

    return "UNKNOWN"


def inspect_subdataset(
    subdataset_path: Path, max_episodes: Optional[int] = None, max_steps: Optional[int] = None
) -> Dict[str, Any]:
    """Inspect a single LeRobot subdataset and return summary statistics"""

    result = {
        "name": subdataset_path.name,
        "path": str(subdataset_path),
        "format": "LEROBOT_V2",
        "num_episodes": 0,
        "num_frames": 0,
        "fps": None,
        "robot_type": None,
        "features": {},
        "video_streams": [],
        "action_features": [],
        "state_features": [],
        "observation_features": [],
        "statistics": {},
        "errors": [],
    }

    # Load info.json
    info_path = subdataset_path / "meta" / "info.json"
    info = load_json(info_path)

    if info is None:
        result["errors"].append(f"Failed to load {info_path}")
        return result

    result["num_episodes"] = info.get("total_episodes", 0)
    result["num_frames"] = info.get("total_frames", 0)
    result["fps"] = info.get("fps")
    result["robot_type"] = info.get("robot_type")
    result["codebase_version"] = info.get("codebase_version")

    # Parse features
    features = info.get("features", {})
    result["features"] = features

    for key, feature_info in features.items():
        dtype = feature_info.get("dtype")
        shape = feature_info.get("shape", [])

        # Categorize features
        if dtype == "video":
            video_info = feature_info.get("info", {})
            result["video_streams"].append(
                {
                    "name": key,
                    "shape": shape,
                    "resolution": f"{video_info.get('video.width', '?')}x{video_info.get('video.height', '?')}",
                    "fps": video_info.get("video.fps"),
                    "codec": video_info.get("video.codec"),
                    "is_depth": video_info.get("video.is_depth_map", False),
                }
            )
            result["observation_features"].append(key)
        elif "command" in key.lower() or "action" in key.lower():
            result["action_features"].append(key)
        elif "state" in key.lower():
            result["state_features"].append(key)
        else:
            result["observation_features"].append(key)

    # Load episode metadata
    episodes_path = subdataset_path / "meta" / "episodes.jsonl"
    episodes = load_jsonl(episodes_path)

    if episodes:
        result["episode_lengths"] = [ep.get("length", 0) for ep in episodes]
        result["tasks"] = list(set(task for ep in episodes for task in ep.get("tasks", [])))
        result["success_rate"] = (
            sum(ep.get("is_episode_success", 0) for ep in episodes) / len(episodes) if episodes else 0
        )

    # Sample parquet data for statistics (if pyarrow available)
    if HAS_PYARROW and max_episodes and max_episodes > 0:
        result["statistics"] = sample_parquet_statistics(
            subdataset_path, result["action_features"], max_episodes, max_steps
        )

    return result


def sample_parquet_statistics(
    subdataset_path: Path, action_features: List[str], max_episodes: int, max_steps: Optional[int] = None
) -> Dict[str, Any]:
    """Sample parquet files to compute statistics without loading everything"""

    stats = {}
    data_dir = subdataset_path / "data"

    if not data_dir.exists():
        return {"error": "No data directory found"}

    # Find parquet files
    parquet_files = sorted(data_dir.glob("**/*.parquet"))

    if not parquet_files:
        return {"error": "No parquet files found"}

    # Sample up to max_episodes files
    sampled_files = parquet_files[:max_episodes] if max_episodes else parquet_files

    try:
        all_values = {feature: [] for feature in action_features}

        for pq_file in sampled_files:
            try:
                # Read parquet file
                table = pq.read_table(pq_file)
                df = table.to_pandas()

                # Limit steps if specified
                if max_steps and len(df) > max_steps:
                    df = df.head(max_steps)

                # Collect values for each action feature
                for feature in action_features:
                    if feature in df.columns:
                        values = df[feature].values
                        # Handle list/array columns
                        if len(values) > 0 and hasattr(values[0], "__len__") and not isinstance(values[0], str):
                            # Flatten arrays
                            all_values[feature].extend([float(v) for row in values for v in row])
                        else:
                            all_values[feature].extend([float(v) for v in values])

            except Exception as e:
                print_warning(f"Failed to read {pq_file.name}: {e}")
                continue

        # Compute statistics
        if HAS_NUMPY:
            for feature, values in all_values.items():
                if values:
                    arr = np.array(values)
                    stats[feature] = {
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "samples": len(values),
                    }
        else:
            # Fallback to Python built-ins
            for feature, values in all_values.items():
                if values:
                    stats[feature] = {
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "samples": len(values),
                    }

        stats["_sampled_episodes"] = len(sampled_files)

    except Exception as e:
        stats["error"] = str(e)

    return stats


def print_subdataset_report(result: Dict[str, Any], verbose: bool = False):
    """Print formatted report for a single subdataset"""

    print_section(f"📁 Dataset: {result['name']}")

    print(f"  Format:        {result['format']}")
    print(f"  Path:          {result['path']}")
    if result.get("codebase_version"):
        print(f"  Version:       {result['codebase_version']}")
    if result.get("robot_type"):
        print(f"  Robot Type:    {result['robot_type']}")
    print(f"  Episodes:      {result['num_episodes']}")
    print(f"  Total Frames:  {result['num_frames']:,}")
    if result["fps"]:
        print(f"  FPS:           {result['fps']}")
        duration_sec = result["num_frames"] / result["fps"]
        print(f"  Duration:      {duration_sec:.1f}s ({duration_sec / 60:.1f}min)")

    if result.get("tasks"):
        print(f"  Tasks:         {', '.join(result['tasks'])}")

    if "success_rate" in result:
        print(f"  Success Rate:  {result['success_rate']:.1%}")

    if result.get("episode_lengths"):
        lengths = result["episode_lengths"]
        print(f"  Episode Len:   min={min(lengths)}, max={max(lengths)}, avg={sum(lengths) / len(lengths):.1f}")

    # Video streams
    if result["video_streams"]:
        print(f"\n  🎥 Video Streams ({len(result['video_streams'])}):")
        for vs in result["video_streams"]:
            depth_marker = " [DEPTH]" if vs["is_depth"] else ""
            print(f"    • {vs['name']}: {vs['resolution']} @ {vs['fps']}fps ({vs['codec']}){depth_marker}")

    # Action features
    if result["action_features"]:
        print(f"\n  🎮 Action Features ({len(result['action_features'])}):")
        for af in result["action_features"]:
            feat_info = result["features"].get(af, {})
            shape = feat_info.get("shape", [])
            dtype = feat_info.get("dtype", "?")
            shape_str = f"shape={tuple(shape)}" if shape else ""
            print(f"    • {af}: {dtype} {shape_str}")

    # State features
    if result["state_features"]:
        print(f"\n  📊 State Features ({len(result['state_features'])}):")
        for sf in result["state_features"]:
            feat_info = result["features"].get(sf, {})
            shape = feat_info.get("shape", [])
            dtype = feat_info.get("dtype", "?")
            shape_str = f"shape={tuple(shape)}" if shape else ""
            print(f"    • {sf}: {dtype} {shape_str}")

    # Statistics (if available)
    if result.get("statistics") and not result["statistics"].get("error"):
        stats = result["statistics"]
        sampled = stats.pop("_sampled_episodes", 0)
        if stats:
            print(f"\n  📈 Action Statistics (sampled {sampled} episodes):")
            for feature, stat in sorted(stats.items()):
                if isinstance(stat, dict) and "min" in stat:
                    print(f"    • {feature}:")
                    print(f"        min={stat['min']:.6f}, max={stat['max']:.6f}, mean={stat['mean']:.6f}")
                    if "std" in stat:
                        print(f"        std={stat['std']:.6f}, samples={stat['samples']}")

    if result.get("errors"):
        print(f"\n  ⚠️  Errors:")
        for err in result["errors"]:
            print(f"    • {err}")


def inspect_local_dataset(
    root_path: Path, max_episodes_per_subdataset: Optional[int] = None, max_steps_per_episode: Optional[int] = None
) -> Dict[str, Any]:
    """Inspect local LeRobot dataset(s)"""

    print_header(f"🔍 Inspecting Local LeRobot Dataset: {root_path}")

    # Detect format
    format_type = detect_dataset_format(root_path)

    if format_type == "NOT_FOUND":
        print_error(f"Dataset path not found: {root_path}")
        return {"error": "Dataset path not found", "results": []}

    if format_type == "UNKNOWN":
        print_error(f"Unknown dataset format at: {root_path}")
        return {"error": "Unknown dataset format", "results": []}

    print_success(f"✓ Detected format: {format_type}")

    results = []

    if format_type == "LEROBOT_V2":
        # Single dataset
        result = inspect_subdataset(root_path, max_episodes_per_subdataset, max_steps_per_episode)
        results.append(result)
        print_subdataset_report(result)

    elif format_type == "LEROBOT_V2_MULTI":
        # Multiple subdatasets
        subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])

        for subdir in subdirs:
            if (subdir / "meta" / "info.json").exists():
                result = inspect_subdataset(subdir, max_episodes_per_subdataset, max_steps_per_episode)
                results.append(result)
                print_subdataset_report(result)

    # Aggregate summary
    if len(results) > 1:
        print_aggregate_summary(results)

    return {"format": format_type, "results": results}


def print_aggregate_summary(results: List[Dict[str, Any]]):
    """Print aggregate summary across all subdatasets"""

    print_section("📊 Aggregate Summary")

    total_episodes = sum(r["num_episodes"] for r in results)
    total_frames = sum(r["num_frames"] for r in results)
    total_subdatasets = len(results)

    print(f"  Total Subdatasets:  {total_subdatasets}")
    print(f"  Total Episodes:     {total_episodes:,}")
    print(f"  Total Frames:       {total_frames:,}")

    # Collect all unique features
    all_action_features = set()
    all_state_features = set()
    all_video_streams = set()
    all_tasks = set()

    for r in results:
        all_action_features.update(r["action_features"])
        all_state_features.update(r["state_features"])
        all_video_streams.update(vs["name"] for vs in r["video_streams"])
        all_tasks.update(r.get("tasks", []))

    print(f"  Unique Actions:     {len(all_action_features)}")
    print(f"  Unique States:      {len(all_state_features)}")
    print(f"  Unique Videos:      {len(all_video_streams)}")
    print(f"  Unique Tasks:       {len(all_tasks)}")

    if all_tasks:
        print(f"\n  All Tasks:")
        for task in sorted(all_tasks):
            print(f"    • {task}")

    # Average statistics
    avg_episodes = total_episodes / total_subdatasets
    avg_frames = total_frames / total_subdatasets

    print(f"\n  Avg Episodes/Dataset:  {avg_episodes:.1f}")
    print(f"  Avg Frames/Dataset:    {avg_frames:.1f}")


def inspect_hf_dataset(repo_id: str):
    """Inspect HuggingFace dataset (requires datasets library)"""

    if not HAS_DATASETS:
        print_error("datasets library not installed. Install with: pip install datasets")
        return {"error": "datasets library not available"}

    print_header(f"🔍 Inspecting HuggingFace Dataset: {repo_id}")

    try:
        # This is a placeholder - actual HF inspection would depend on the specific format
        print_warning("HuggingFace dataset inspection not fully implemented yet.")
        print("  You can use: datasets.load_dataset('{}') for manual inspection.".format(repo_id))

        return {"error": "HF inspection not implemented", "repo_id": repo_id}

    except Exception as e:
        print_error(f"Failed to load HuggingFace dataset: {e}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Inspect LeRobot datasets (local or HuggingFace)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect default local dataset
  %(prog)s

  # Inspect custom path
  %(prog)s --root ./my_dataset

  # Sample statistics from first 5 episodes, max 100 steps each
  %(prog)s --max_episodes_per_subdataset 5 --max_steps_per_episode 100

  # Inspect HuggingFace repo (requires datasets library)
  %(prog)s --hf_repo openpi/airdust_lerobot
        """,
    )

    parser.add_argument(
        "--root",
        type=str,
        default="./openpi/airdust_lerobot",
        help="Root path to local LeRobot dataset (default: ./openpi/airdust_lerobot)",
    )

    parser.add_argument(
        "--hf_repo",
        type=str,
        default=None,
        help="HuggingFace repository ID (e.g., 'openpi/airdust_lerobot'). If specified, inspects HF dataset instead of local.",
    )

    parser.add_argument(
        "--max_episodes_per_subdataset",
        type=int,
        default=None,
        help="Maximum number of episodes to sample per subdataset for statistics (default: None, no sampling)",
    )

    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=None,
        help="Maximum number of steps to sample per episode for statistics (default: None, use all steps)",
    )

    args = parser.parse_args()

    # Check dependencies for sampling
    if args.max_episodes_per_subdataset:
        if not HAS_PYARROW:
            print_warning("pyarrow not installed. Sampling statistics will be skipped.")
            print_warning("Install with: pip install pyarrow")
            args.max_episodes_per_subdataset = None
        if not HAS_NUMPY:
            print_warning("numpy not installed. Statistics will use Python built-ins (slower).")

    try:
        if args.hf_repo:
            # Inspect HuggingFace dataset
            result = inspect_hf_dataset(args.hf_repo)
        else:
            # Inspect local dataset
            root_path = Path(args.root)
            result = inspect_local_dataset(root_path, args.max_episodes_per_subdataset, args.max_steps_per_episode)

        if result.get("error"):
            sys.exit(1)

        print_header("✅ Inspection Complete")

    except KeyboardInterrupt:
        print_error("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
