"""Latest-value ZMQ server for the Astribot policy client."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any

import numpy as np
from openpi_client import msgpack_numpy
import zmq

from openpi.serving.astribot_policy_schema import ACTION_DIM
from openpi.serving.astribot_policy_schema import CANONICAL_ACTION_JOINTS
from openpi.serving.astribot_policy_schema import CANONICAL_BODY_JOINTS
from openpi.serving.astribot_policy_schema import HAND_CONTROL_JOINTS
from openpi.serving.astribot_policy_schema import POLICY_PROTOCOL_VERSION
from openpi.serving.astribot_policy_schema import POLICY_SCHEMA_HASH

logger = logging.getLogger(__name__)


@dataclass
class _Session:
    identity: bytes
    session_id: str
    control_epoch: int = 0
    streaming: bool = False
    latest_observation: Mapping[str, object] | None = None
    latest_observation_sequence: int = -1
    last_inferred_sequence: int = -1
    latest_received_at: float = 0.0
    action_sequence: int = 0


class ZmqPolicyServer:
    """Serve a single-action policy stream over a ROUTER socket.

    Observations are intentionally latest-value data.  The socket and server
    each keep a bounded queue, and the receive loop drains all available
    frames before selecting the newest sequence for inference.  A policy is
    run at most ``max_inference_hz`` and each observation sequence is inferred
    at most once.
    """

    def __init__(
        self,
        policy: Any,
        *,
        host: str = "0.0.0.0",
        port: int = 5555,
        bind_uri: str | None = None,
        metadata: Mapping[str, object] | None = None,
        max_inference_hz: float = 10.0,
    ) -> None:
        if not np.isfinite(max_inference_hz) or max_inference_hz <= 0.0:
            raise ValueError("max_inference_hz must be finite and positive")
        self._policy = policy
        self._bind_uri = bind_uri or f"tcp://{host}:{port}"
        self._metadata = {
            "astribot_policy_protocol": POLICY_PROTOCOL_VERSION,
            "action_schema_hash": POLICY_SCHEMA_HASH,
            "checkpoint_id": "unknown",
            "transport": "zmq",
            "action_semantics": "absolute_joint_position_single_frame",
            "max_inference_hz": float(max_inference_hz),
            **dict(metadata or {}),
        }
        self._max_inference_hz = float(max_inference_hz)
        self._period = 1.0 / self._max_inference_hz
        self._ready = threading.Event()
        self._bound_uri: str | None = None

    @property
    def bound_uri(self) -> str | None:
        """Return the concrete endpoint after ``run`` has bound the socket."""

        return self._bound_uri

    @property
    def max_inference_hz(self) -> float:
        return self._max_inference_hz

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        return self._ready.wait(timeout)

    def serve_forever(self) -> None:
        self.run()

    def run(self, stop_event: threading.Event | None = None) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(self._bind_uri)
        self._bound_uri = socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self._ready.set()
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        sessions: dict[bytes, _Session] = {}
        next_inference_at = time.monotonic() + self._period
        logger.info(
            "ZMQ policy server listening on %s (max inference %.3f Hz)",
            self._bound_uri,
            self._max_inference_hz,
        )

        try:
            while stop_event is None or not stop_event.is_set():
                now = time.monotonic()
                if next_inference_at > now:
                    wait_seconds = next_inference_at - now
                    timeout_ms = min(50, int(wait_seconds * 1000.0))
                    if timeout_ms == 0:
                        timeout_ms = 1
                else:
                    # Keep an already-due deadline while idle. This avoids
                    # pushing a newly received observation into the future
                    # just because no observation was queued for one poll.
                    timeout_ms = 50
                poller.poll(timeout_ms)
                self._drain_messages(socket, sessions)

                now = time.monotonic()
                if now < next_inference_at:
                    continue
                session = self._newest_pending_session(sessions)
                if session is None:
                    continue
                inference_started_at = time.monotonic()
                self._infer_and_send(socket, session)
                # Rate-limit starts, rather than adding a full period after
                # inference completes. If inference overruns the period, the
                # next deadline is already due and the latest observation can
                # run as soon as the current inference returns.
                next_inference_at = inference_started_at + self._period
        finally:
            self._ready.clear()
            socket.close(0)
            context.term()

    @staticmethod
    def _newest_pending_session(sessions: Mapping[bytes, _Session]) -> _Session | None:
        pending = [
            session
            for session in sessions.values()
            if session.streaming
            and session.latest_observation is not None
            and session.latest_observation_sequence > session.last_inferred_sequence
        ]
        if not pending:
            return None
        return max(pending, key=lambda session: session.latest_received_at)

    def _drain_messages(self, socket: zmq.Socket, sessions: dict[bytes, _Session]) -> None:
        while True:
            try:
                parts = socket.recv_multipart(flags=zmq.DONTWAIT)
            except zmq.Again:
                return
            if len(parts) != 2:
                logger.warning("discarding ZMQ message with %d frames", len(parts))
                continue
            identity, raw = parts
            try:
                payload = msgpack_numpy.unpackb(raw)
            except Exception as exc:
                logger.warning("discarding undecodable ZMQ message: %s", exc)
                continue
            if not isinstance(payload, Mapping):
                logger.warning("discarding non-mapping ZMQ message")
                continue
            self._handle_message(socket, identity, payload, sessions)

    def _handle_message(
        self,
        socket: zmq.Socket,
        identity: bytes,
        payload: Mapping[str, object],
        sessions: dict[bytes, _Session],
    ) -> None:
        message_type = str(payload.get("type", ""))
        session_id = str(payload.get("session_id", ""))
        if message_type == "hello":
            if (
                not session_id
                or payload.get("protocol_version") != POLICY_PROTOCOL_VERSION
                or payload.get("schema_hash") != POLICY_SCHEMA_HASH
            ):
                return
            session = _Session(identity=identity, session_id=session_id)
            try:
                session.control_epoch = int(payload.get("control_epoch", 0))
            except (TypeError, ValueError):
                session.control_epoch = 0
            sessions[identity] = session
            self._send(socket, identity, self._metadata)
            return

        session = sessions.get(identity)
        if session is None or session.session_id != session_id:
            return
        if message_type == "control":
            self._handle_control(session, payload)
            return
        if message_type != "observation":
            return

        try:
            if payload.get("protocol_version") != POLICY_PROTOCOL_VERSION:
                return
            if payload.get("schema_hash") != POLICY_SCHEMA_HASH:
                return
            epoch = int(payload.get("control_epoch", session.control_epoch))
            sequence = int(payload.get("request_sequence", -1))
        except (TypeError, ValueError):
            return
        if epoch != session.control_epoch or sequence < 0 or not session.streaming:
            return
        if sequence <= session.latest_observation_sequence:
            return
        # Replacing this reference is the discard-old-observation policy.
        session.latest_observation = payload
        session.latest_observation_sequence = sequence
        session.latest_received_at = time.monotonic()

    @staticmethod
    def _handle_control(session: _Session, payload: Mapping[str, object]) -> None:
        if payload.get("protocol_version") != POLICY_PROTOCOL_VERSION:
            return
        if payload.get("schema_hash") != POLICY_SCHEMA_HASH:
            return
        try:
            epoch = int(payload.get("control_epoch", session.control_epoch))
        except (TypeError, ValueError):
            return
        if epoch != session.control_epoch:
            session.control_epoch = epoch
            session.latest_observation = None
            session.latest_observation_sequence = -1
            session.last_inferred_sequence = -1
            session.action_sequence = 0
        command = str(payload.get("command", "")).strip().lower()
        if command == "arm":
            session.streaming = True
        elif command in {"hold", "shadow", "stop"}:
            session.streaming = False
            session.latest_observation = None

    def _infer_and_send(self, socket: zmq.Socket, session: _Session) -> None:
        observation = session.latest_observation
        if observation is None:
            return
        sequence = session.latest_observation_sequence
        # Consume the selected sequence even when model preprocessing fails;
        # the next observation can then be tested without retrying bad data.
        session.last_inferred_sequence = sequence
        started = time.perf_counter()
        try:
            action, timing = self._policy.infer_observation(observation)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
                raise ValueError(f"policy returned invalid action shape/values: {action.shape}")
            mask = np.asarray(observation.get("enabled_action_mask"), dtype=np.bool_).reshape(-1)
            if mask.shape != (ACTION_DIM,):
                raise ValueError(f"enabled_action_mask must contain {ACTION_DIM} values")
            timing_payload = dict(timing or {})
            # The adapter also reports its own end-to-end duration. Keep that
            # value under a distinct key so the server's scheduler timing is
            # not silently overwritten by a duplicate ``infer_ms`` field.
            if "infer_ms" in timing_payload:
                timing_payload["adapter_infer_ms"] = timing_payload.pop("infer_ms")
            response = {
                "type": "action",
                "protocol_version": POLICY_PROTOCOL_VERSION,
                "schema_hash": POLICY_SCHEMA_HASH,
                "session_id": session.session_id,
                "control_epoch": session.control_epoch,
                "action_sequence": session.action_sequence,
                "source_observation_sequence": sequence,
                "body_joint_names": list(CANONICAL_BODY_JOINTS),
                "hand_control_joint_names": list(HAND_CONTROL_JOINTS),
                "action_joint_names": list(CANONICAL_ACTION_JOINTS),
                "enabled_action_mask": mask,
                "action": action,
                "server_timing": {
                    "infer_ms": (time.perf_counter() - started) * 1000.0,
                    **timing_payload,
                },
            }
            gripper_action = getattr(self._policy, "last_gripper_action", None)
            if gripper_action is not None:
                values = np.asarray(gripper_action, dtype=np.float32).reshape(-1)
                if values.shape != (2,) or not np.isfinite(values).all():
                    raise ValueError("policy returned an invalid gripper action")
                response["gripper"] = {
                    "joint_names": ["astribot_gripper_left", "astribot_gripper_right"],
                    "position_mm": values,
                }
            self._send(socket, session.identity, response)
            session.action_sequence += 1
        except Exception as exc:
            logger.warning("inference failed for session %s: %s", session.session_id, exc)

    @staticmethod
    def _send(socket: zmq.Socket, identity: bytes, payload: Mapping[str, object]) -> None:
        try:
            socket.send_multipart(
                [identity, msgpack_numpy.packb(dict(payload))],
                flags=zmq.DONTWAIT,
            )
        except zmq.Again:
            logger.debug("dropping action because ZMQ send queue is full")
