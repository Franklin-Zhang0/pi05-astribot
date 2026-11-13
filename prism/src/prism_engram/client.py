"""Small OpenPI-compatible client with explicit PRISM episode resets."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from websockets.sync.client import connect

from prism_engram import codec


class PrismClient:
    def __init__(self, host: str = "127.0.0.1", port: int | None = None):
        self.connection = connect(server_uri(host, port), compression=None, max_size=64 * 1024**2)
        self.metadata = codec.unpackb(self.connection.recv())

    def request(self, message: dict) -> dict:
        self.connection.send(codec.packb(message))
        reply = self.connection.recv()
        if isinstance(reply, str):
            raise RuntimeError(reply)
        return codec.unpackb(reply)

    def reset(self, seed: int = 0) -> dict:
        return self.request({"op": "reset", "seed": seed})

    def infer(self, observation: dict) -> dict:
        return self.request(observation)

    def observe(self, observation: dict) -> dict:
        return self.request({**observation, "op": "observe"})

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def server_uri(host: str, port: int | None = None) -> str:
    explicit_uri = host.startswith(("ws://", "wss://"))
    parts = urlsplit(host if explicit_uri else f"ws://{host}")
    if not parts.hostname:
        raise ValueError("A server hostname is required")
    if parts.port is not None:
        if port is not None and port != parts.port:
            raise ValueError("Conflicting URL and explicit ports")
        return urlunsplit(parts)
    selected = port if port is not None else (None if explicit_uri else 8000)
    if selected is not None:
        if isinstance(selected, bool) or not isinstance(selected, int) or not 1 <= selected <= 65535:
            raise ValueError("port must be in [1,65535]")
        parts = parts._replace(netloc=f"{parts.netloc}:{selected}")
    return urlunsplit(parts)
