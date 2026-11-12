from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import hashlib
import json
import logging
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def _engram_video_contract(data_config):
    return {
        "version": 1,
        "mode": data_config.engram_video_mode,
        "profiling_subset_only": data_config.engram_proxy_subset_only,
        "proxy_identity": data_config.engram_video_identity,
    }


def _engram_multi_contract(data_config, manifest):
    from openpi.policies.engram_multi_policy import camera_size
    from openpi.shared.engram_multi import ROUTES

    return {
        "format": "engram-multi-policy-v1",
        "action_dim": 64,
        "action_horizon": manifest["action_horizon"],
        "action_head_names": manifest["action_head_names"],
        "max_token_len": manifest["max_token_len"],
        "routes": ROUTES,
        "structured_prompt_probability": data_config.engram_multi_structured_probability,
        "prompt_keys": ["instruction", "robot_type", "end_effector_type", "end_effector_state_available"],
        "preserve_prompt_text": True,
        "camera_sizes": {robot: {role: list(camera_size(robot, role))
                                 for role in ("head", "left_wrist", "right_wrist")}
                         for robot in ("dex_w1", "galbot")},
        "video": _engram_video_contract(data_config),
        "stats_sha256": manifest["stats_sha256"],
    }


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        if data_config.engram_multi_data_dir is not None:
            destination = directory / data_config.asset_id
            destination.mkdir(parents=True, exist_ok=True)
            stats = epath.Path(data_config.engram_multi_stats_path).read_bytes()
            manifest_bytes = epath.Path(data_config.engram_multi_manifest_path).read_bytes()
            if (hashlib.sha256(stats).hexdigest() != data_config.engram_multi_stats_sha256
                    or hashlib.sha256(manifest_bytes).hexdigest() != data_config.engram_multi_manifest_sha256):
                raise ValueError("Multi-embodiment statistics/provenance changed while the trainer was running")
            manifest = json.loads(manifest_bytes)
            if manifest.get("stats_sha256") != hashlib.sha256(stats).hexdigest():
                raise ValueError("Typed checkpoint statistics differ from their provenance")
            (destination / "engram_multi_stats.json").write_bytes(stats)
            (destination / "engram_multi_manifest.json").write_bytes(manifest_bytes)
            (destination / "engram_multi_contract.json").write_text(
                json.dumps(_engram_multi_contract(data_config, manifest), indent=2) + "\n"
            )
            prompts = sorted({prompt for values in manifest.get("prompts_by_route", {}).values() for prompt in values})
            (destination / "prompt.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2) + "\n")
            return
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)
            if data_config.engram_stats_manifest is not None:
                manifest = json.loads(epath.Path(data_config.engram_stats_manifest).read_text())
                manifest["norm_stats_sha256"] = hashlib.sha256(
                    (directory / data_config.asset_id / "norm_stats.json").read_bytes()
                ).hexdigest()
                (directory / data_config.asset_id / "engram_stats_manifest.json").write_text(json.dumps(manifest, indent=2))
                (directory / data_config.asset_id / "engram_video_contract.json").write_text(
                    json.dumps(_engram_video_contract(data_config), indent=2)
                )

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    data_config = data_loader.data_config()
    if data_config.engram_multi_data_dir is not None:
        restored_step = checkpoint_manager.latest_step() if step is None else step
        if restored_step is None:
            raise ValueError("No multi-embodiment checkpoint is available to resume")
        saved = checkpoint_manager.directory / str(restored_step) / "assets" / data_config.asset_id
        current_manifest = epath.Path(data_config.engram_multi_manifest_path).read_bytes()
        current_stats = epath.Path(data_config.engram_multi_stats_path).read_bytes()
        if ((saved / "engram_multi_manifest.json").read_bytes() != current_manifest
                or (saved / "engram_multi_stats.json").read_bytes() != current_stats):
            raise ValueError("Cannot resume with changed multi-embodiment statistics or source selection")
        if json.loads((saved / "engram_multi_contract.json").read_text()) != _engram_multi_contract(
            data_config, json.loads(current_manifest)
        ):
            raise ValueError("Cannot resume with changed multi-embodiment prompt, route, or camera contract")
    if data_config.engram_stats_manifest is not None:
        restored_step = checkpoint_manager.latest_step() if step is None else step
        if restored_step is None:
            raise ValueError("No checkpoint is available to resume")
        saved_dir = checkpoint_manager.directory / str(restored_step) / "assets" / data_config.asset_id
        try:
            checkpoint_stats = _normalize.load(saved_dir)
            checkpoint_manifest = json.loads((saved_dir / "engram_stats_manifest.json").read_text())
            current_manifest = json.loads(epath.Path(data_config.engram_stats_manifest).read_text())
        except FileNotFoundError as exc:
            raise ValueError("Engram checkpoint normalization or provenance is missing") from exc
        if (
            _normalize.serialize_json(checkpoint_stats) != _normalize.serialize_json(data_config.norm_stats)
            or checkpoint_manifest != current_manifest
        ):
            raise ValueError("Cannot resume Engram training with changed normalization statistics or provenance")
        video_contract_path = saved_dir / "engram_video_contract.json"
        if video_contract_path.exists():
            video_contract = json.loads(video_contract_path.read_text())
            if video_contract != _engram_video_contract(data_config):
                raise ValueError("Cannot resume Engram training with changed source-video/proxy identity")
        elif data_config.engram_video_mode != "original" or data_config.engram_proxy_subset_only:
            raise ValueError("Legacy checkpoint has no proxy identity; resume requires the full original-video set")

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
