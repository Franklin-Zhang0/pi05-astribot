#!/usr/bin/env python3
"""Add the /data2 AFS volume to the standard authenticated-AOSS ACP helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

HELPER = Path("/data/250010203/franklin/.codex/skills/sco-control/scripts/acp_create_with_aoss.py")
DATA2_VOLUME_ID = "019eba92-c2ee-78ef-953f-e2d087fbb861"


def main() -> int:
    spec = importlib.util.spec_from_file_location("acp_create_with_aoss", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load SCO helper: {HELPER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    original_build_request_body = module.build_request_body

    def build_request_body(args: Any, scope: Any, access_key: str, secret_key: str) -> dict[str, Any]:
        body = original_build_request_body(args, scope, access_key, secret_key)
        body["mount"].insert(
            1,
            {
                "type": "PV_AFS",
                "id": DATA2_VOLUME_ID,
                "name": "",
                # This AFS volume is ACP-mountable through its established /202608
                # subdirectory contract; mounting the volume root fails before startup.
                "mount_path": "/mnt/afs",
                "subdir": "/202608",
                "zone": scope.compute_zone,
                "endpoint": "",
                "mount_options": "",
            },
        )
        return body

    module.build_request_body = build_request_body
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
