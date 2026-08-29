import threading
import time

import numpy as np
from openpi_client import msgpack_numpy
import zmq

from openpi.serving import astribot_policy_schema as schema
from openpi.serving import zmq_policy_server


class _RecordingPolicy:
    def __init__(self, inference_seconds: float = 0.0) -> None:
        self.calls: list[tuple[int, float]] = []
        self.inference_seconds = inference_seconds

    def infer_observation(self, request: dict[str, object]) -> tuple[np.ndarray, dict[str, float]]:
        sequence = int(request["request_sequence"])
        self.calls.append((sequence, time.monotonic()))
        if self.inference_seconds:
            time.sleep(self.inference_seconds)
        return np.full(schema.ACTION_DIM, sequence, dtype=np.float32), {
            "infer_ms": 0.1,
            "policy_infer_ms": 0.05,
        }


def _message(session_id: str, sequence: int, mask: np.ndarray) -> dict[str, object]:
    return {
        "type": "observation",
        "protocol_version": schema.POLICY_PROTOCOL_VERSION,
        "schema_hash": schema.POLICY_SCHEMA_HASH,
        "session_id": session_id,
        "request_sequence": sequence,
        "control_epoch": 0,
        "enabled_action_mask": mask,
        "observation": {},
    }


def _recv(socket: zmq.Socket, timeout_ms: int = 2000) -> dict[str, object]:
    assert socket.poll(timeout_ms), "timed out waiting for ZMQ policy response"
    payload = msgpack_numpy.unpackb(socket.recv())
    assert isinstance(payload, dict)
    return payload


def test_zmq_server_uses_latest_observation_and_caps_inference_frequency() -> None:
    policy = _RecordingPolicy()
    stop_event = threading.Event()
    server = zmq_policy_server.ZmqPolicyServer(
        policy,
        bind_uri="tcp://127.0.0.1:0",
        metadata={"checkpoint_id": "test-zmq"},
        max_inference_hz=10.0,
    )
    thread = threading.Thread(target=server.run, args=(stop_event,), daemon=True)
    thread.start()
    assert server.wait_until_ready()
    assert server.bound_uri is not None

    context = zmq.Context()
    client = context.socket(zmq.DEALER)
    client.setsockopt(zmq.SNDHWM, 100)
    client.connect(server.bound_uri)
    mask = np.ones(schema.ACTION_DIM, dtype=np.bool_)
    session_id = "integration-session"
    try:
        client.send(
            msgpack_numpy.packb(
                {
                    "type": "hello",
                    "protocol_version": schema.POLICY_PROTOCOL_VERSION,
                    "schema_hash": schema.POLICY_SCHEMA_HASH,
                    "session_id": session_id,
                    "control_epoch": 0,
                }
            )
        )
        metadata = _recv(client)
        assert metadata["astribot_policy_protocol"] == 1
        assert metadata["action_schema_hash"] == schema.POLICY_SCHEMA_HASH

        client.send(
            msgpack_numpy.packb(
                {
                    "type": "control",
                    "command": "arm",
                    "protocol_version": schema.POLICY_PROTOCOL_VERSION,
                    "schema_hash": schema.POLICY_SCHEMA_HASH,
                    "session_id": session_id,
                    "control_epoch": 0,
                }
            )
        )
        for sequence in range(5):
            client.send(msgpack_numpy.packb(_message(session_id, sequence, mask)))

        first = _recv(client)
        action, _, source_sequence = schema.validate_single_action_response(
            first,
            session_id=session_id,
            expected_mask=mask,
        )
        assert source_sequence == 4
        assert np.all(action == 4.0)
        assert first["server_timing"]["infer_ms"] >= first["server_timing"]["policy_infer_ms"]  # type: ignore[index]
        assert first["server_timing"]["adapter_infer_ms"] == 0.1  # type: ignore[index]
        assert [sequence for sequence, _ in policy.calls] == [4]

        started = time.monotonic()
        for sequence in range(5, 12):
            client.send(msgpack_numpy.packb(_message(session_id, sequence, mask)))
            time.sleep(0.01)
        time.sleep(0.25)
        elapsed = time.monotonic() - started
        assert len(policy.calls) >= 2
        assert len(policy.calls) <= int(elapsed * 10.0) + 1
        intervals = [right - left for (_, left), (_, right) in zip(policy.calls, policy.calls[1:], strict=False)]
        assert all(interval >= 0.075 for interval in intervals)
    finally:
        stop_event.set()
        client.close(0)
        context.term()
        thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_zmq_server_does_not_add_period_after_slow_inference() -> None:
    policy = _RecordingPolicy(inference_seconds=0.12)
    stop_event = threading.Event()
    server = zmq_policy_server.ZmqPolicyServer(
        policy,
        bind_uri="tcp://127.0.0.1:0",
        max_inference_hz=10.0,
    )
    thread = threading.Thread(target=server.run, args=(stop_event,), daemon=True)
    thread.start()
    assert server.wait_until_ready()
    assert server.bound_uri is not None

    context = zmq.Context()
    client = context.socket(zmq.DEALER)
    client.connect(server.bound_uri)
    mask = np.ones(schema.ACTION_DIM, dtype=np.bool_)
    session_id = "slow-inference-session"
    try:
        client.send(
            msgpack_numpy.packb(
                {
                    "type": "hello",
                    "protocol_version": schema.POLICY_PROTOCOL_VERSION,
                    "schema_hash": schema.POLICY_SCHEMA_HASH,
                    "session_id": session_id,
                    "control_epoch": 0,
                }
            )
        )
        _recv(client)
        client.send(
            msgpack_numpy.packb(
                {
                    "type": "control",
                    "command": "arm",
                    "protocol_version": schema.POLICY_PROTOCOL_VERSION,
                    "schema_hash": schema.POLICY_SCHEMA_HASH,
                    "session_id": session_id,
                    "control_epoch": 0,
                }
            )
        )
        client.send(msgpack_numpy.packb(_message(session_id, 0, mask)))
        _recv(client)

        second_sent_at = time.monotonic()
        client.send(msgpack_numpy.packb(_message(session_id, 1, mask)))
        second = _recv(client, timeout_ms=1000)
        assert second["source_observation_sequence"] == 1

        assert len(policy.calls) == 2
        start_interval = policy.calls[1][1] - policy.calls[0][1]
        assert start_interval >= 0.09
        assert start_interval < 0.20
        assert time.monotonic() - second_sent_at < 0.20
    finally:
        stop_event.set()
        client.close(0)
        context.term()
        thread.join(timeout=2.0)
    assert not thread.is_alive()
