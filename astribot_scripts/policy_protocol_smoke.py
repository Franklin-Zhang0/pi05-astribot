#!/usr/bin/env python3
"""Run a hardware-free ZMQ protocol smoke test for both end-effectors."""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import zmq
from openpi_client import msgpack_numpy

from openpi.serving import astribot_policy_schema as schema
from openpi.serving.zmq_policy_server import ZmqPolicyServer


class _FakePolicy:
    def __init__(self, *, gripper: bool) -> None:
        self.gripper = gripper
        self.last_gripper_action = np.asarray([25.0, 35.0], dtype=np.float32) if gripper else None

    def infer_observation(self, request: dict[str, object]) -> tuple[np.ndarray, dict[str, float]]:
        sequence = int(request["request_sequence"])
        return np.full(schema.ACTION_DIM, sequence, dtype=np.float32), {"infer_ms": 0.0}


def _observation(session_id: str, sequence: int) -> dict[str, object]:
    return {
        "type": "observation",
        "protocol_version": schema.POLICY_PROTOCOL_VERSION,
        "schema_hash": schema.POLICY_SCHEMA_HASH,
        "session_id": session_id,
        "request_sequence": sequence,
        "control_epoch": 0,
        "enabled_action_mask": np.ones(schema.ACTION_DIM, dtype=np.bool_),
        "observation": {},
    }


def _recv(socket: zmq.Socket) -> dict[str, object]:
    if not socket.poll(2000):
        raise TimeoutError("timed out waiting for policy response")
    payload = msgpack_numpy.unpackb(socket.recv())
    if not isinstance(payload, dict):
        raise TypeError("policy response is not a mapping")
    return payload


def _run_case(end_effector: str) -> None:
    stop = threading.Event()
    policy = _FakePolicy(gripper=end_effector == "gripper")
    server = ZmqPolicyServer(policy, bind_uri="tcp://127.0.0.1:0", max_inference_hz=100.0)
    thread = threading.Thread(target=server.run, args=(stop,), daemon=True)
    thread.start()
    if not server.wait_until_ready() or server.bound_uri is None:
        raise RuntimeError("policy server did not become ready")
    context = zmq.Context()
    client = context.socket(zmq.DEALER)
    client.connect(server.bound_uri)
    session_id = f"protocol-smoke-{end_effector}"
    try:
        client.send(msgpack_numpy.packb({
            "type": "hello",
            "protocol_version": schema.POLICY_PROTOCOL_VERSION,
            "schema_hash": schema.POLICY_SCHEMA_HASH,
            "session_id": session_id,
            "control_epoch": 0,
        }))
        metadata = _recv(client)
        schema.validate_server_metadata(metadata)
        client.send(msgpack_numpy.packb({
            "type": "control",
            "command": "arm",
            "protocol_version": schema.POLICY_PROTOCOL_VERSION,
            "schema_hash": schema.POLICY_SCHEMA_HASH,
            "session_id": session_id,
            "control_epoch": 0,
        }))
        # Send a burst; latest-value semantics must return the newest frame.
        for sequence in range(3):
            client.send(msgpack_numpy.packb(_observation(session_id, sequence)))
        response = _recv(client)
        action, _, source = schema.validate_single_action_response(
            response,
            session_id=session_id,
            expected_mask=np.ones(schema.ACTION_DIM, dtype=np.bool_),
        )
        if source != 2 or not np.all(action == 2.0):
            raise AssertionError(f"latest-value contract failed for {end_effector}: source={source}")
        if end_effector == "gripper":
            gripper = response.get("gripper")
            if not isinstance(gripper, dict) or not np.array_equal(
                np.asarray(gripper.get("position_mm")), np.asarray([25.0, 35.0], dtype=np.float32)
            ):
                raise AssertionError("stock gripper envelope missing or malformed")
        print(f"{end_effector}: ok (source_sequence={source})")
    finally:
        client.send(msgpack_numpy.packb({
            "type": "control",
            "command": "stop",
            "protocol_version": schema.POLICY_PROTOCOL_VERSION,
            "schema_hash": schema.POLICY_SCHEMA_HASH,
            "session_id": session_id,
            "control_epoch": 0,
        }))
        time.sleep(0.02)
        stop.set()
        client.close(0)
        context.term()
        thread.join(timeout=2.0)
        if thread.is_alive():
            raise RuntimeError("policy server did not stop")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-effector", choices=("both", "apexhand", "gripper"), default="both")
    args = parser.parse_args()
    cases = ("apexhand", "gripper") if args.end_effector == "both" else (args.end_effector,)
    for end_effector in cases:
        _run_case(end_effector)


if __name__ == "__main__":
    main()
