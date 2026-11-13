"""OpenPI MessagePack WebSocket serving, one recurrent session per connection."""

from __future__ import annotations

import argparse
import logging
import time

from prism_engram import codec

METADATA = {
    "model": "engram_prism",
    "protocol_version": 1,
    "stateful": True,
    "reset": {"op": "reset", "seed": 0},
    "observation_hz": 10.0,
    "action_horizon": 32,
    "action_hz": 30.0,
    "camera_order": ["head", "left_wrist", "right_wrist"],
    "output_contract": "W1 absolute EEF/body/available-hand; Galbot absolute joints",
}


def connection_handler(runtime, connection):
    session = runtime.new_session()
    connection.send(codec.packb(METADATA))
    try:
        for payload in connection:
            started = time.perf_counter()
            generation = getattr(session, "episode_generation", None)
            try:
                if not isinstance(payload, bytes):
                    raise ValueError("Requests must be OpenPI MessagePack binary frames")
                request = codec.unpackb(payload)
                if not isinstance(request, dict):
                    raise ValueError("Requests must be dictionaries")
                operation = request.get("op", "infer")
                if operation == "reset":
                    extra = set(request) - {"op", "seed"}
                    if extra:
                        raise ValueError("reset accepts only op and seed")
                    response = session.reset(request.get("seed", 0))
                elif operation in {"infer", "observe"}:
                    response = session.infer(request, decode=operation == "infer")
                else:
                    raise ValueError("Supported operations: infer, observe, reset")
                response["server_timing"] = {"infer_ms": 1000 * (time.perf_counter() - started)}
                connection.send(codec.packb(response))
            except (ValueError, TypeError, KeyError) as error:
                reset_notice = (
                    " Episode was reset; restart the observation stream."
                    if getattr(session, "episode_generation", None) != generation
                    else ""
                )
                connection.send(f"Request failed: {error}.{reset_notice}")
            except Exception:
                session.reset()
                logging.exception("Inference failed; this connection's episode was reset")
                connection.send("Inference failed; episode reset. Check server log and send a reset request.")
    finally:
        session.reset()


def serve(runtime, host: str = "127.0.0.1", port: int = 8000):
    from websockets.sync.server import serve as websocket_serve

    with websocket_serve(
        lambda connection: connection_handler(runtime, connection), host, port, compression=None, max_size=64 * 1024**2
    ) as server:
        logging.info("Serving Engram PRISM on %s:%d", host, port)
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Exported, portable PRISM inference bundle")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    from prism_engram.runtime import Runtime

    runtime = Runtime.from_bundle(args.checkpoint, device=args.device)
    serve(runtime, args.host, args.port)


if __name__ == "__main__":
    main()
