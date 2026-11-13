"""Wire compatibility and per-connection episode ownership."""

import importlib.util
from pathlib import Path
from threading import Thread

import numpy as np
import pytest
from prism_engram import codec
from prism_engram.client import PrismClient, server_uri
from prism_engram.server import connection_handler
from websockets.sync.server import serve


def test_wire_format_matches_existing_openpi_numpy_codec():
    path = Path(__file__).resolve().parents[2] / "packages/openpi-client/src/openpi_client/msgpack_numpy.py"
    spec = importlib.util.spec_from_file_location("original_codec", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = {"image": np.arange(90, dtype=np.uint8).reshape(5, 6, 3), "scalar": np.float32(0.5)}
    for encoder, decoder in [(codec.packb, module.unpackb), (module.packb, codec.unpackb)]:
        decoded = decoder(encoder(value))
        np.testing.assert_array_equal(decoded["image"], value["image"])
        assert decoded["scalar"] == 0.5
    with pytest.raises(ValueError):
        codec.packb(np.array([object()]))


@pytest.mark.parametrize(
    "host,port,expected",
    [
        ("127.0.0.1", None, "ws://127.0.0.1:8000"),
        ("127.0.0.1", 9000, "ws://127.0.0.1:9000"),
        ("ws://localhost:8000/path", None, "ws://localhost:8000/path"),
        ("wss://example.org/path", None, "wss://example.org/path"),
        ("ws://localhost/path", 9000, "ws://localhost:9000/path"),
    ],
)
def test_client_url_does_not_duplicate_or_misplace_ports(host, port, expected):
    assert server_uri(host, port) == expected


class Session:
    def __init__(self):
        self.reset()

    def reset(self, seed=0):
        self.steps = 0
        return {"reset": True, "seed": seed}

    def infer(self, value, decode=True):
        self.steps += 1
        return {"step": self.steps, "decoded": decode, "actions": value.get("actions")}


class Runtime:
    def new_session(self):
        return Session()


def test_websocket_roundtrip_reset_observe_and_isolation():
    runtime = Runtime()
    with serve(lambda connection: connection_handler(runtime, connection), "127.0.0.1", 0, compression=None) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.socket.getsockname()[1]
        try:
            with PrismClient(port=port) as a, PrismClient(port=port) as b:
                assert a.metadata["stateful"] is True
                result = a.infer({"actions": np.eye(3)})
                assert result["step"] == 1
                np.testing.assert_array_equal(result["actions"], np.eye(3))
                assert a.observe({})["step"] == 2
                assert b.infer({})["step"] == 1
                assert a.reset(12)["seed"] == 12
                assert a.infer({})["step"] == 1
                with pytest.raises(RuntimeError):
                    a.request({"op": "unknown"})
                assert b.infer({})["step"] == 2
        finally:
            server.shutdown()
            thread.join(timeout=5)
