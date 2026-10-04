"""What leaves the machine when a cloud provider is used.

Only skill *result summaries* (results.json content: counts, statistics, top-k
lists) are sent - never expression matrices, per-cell/per-spot tables, or files.
Absolute paths are reduced to ``~``-relative form so the user name is not sent,
long lists are truncated, and the size of each tool result is capped. The user
must acknowledge cloud use once (configs/llm.yaml ``cloud.acknowledged`` or
NEUROSTIAS_ALLOW_CLOUD=1).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

HOME = str(Path.home())
MAX_LIST = 20
MAX_CHARS = 6000
DROP_KEYS = {"artifacts"}


def _clean(obj, cloud: bool):
    if isinstance(obj, dict):
        return {k: _clean(v, cloud) for k, v in obj.items() if k not in DROP_KEYS}
    if isinstance(obj, list):
        out = [_clean(v, cloud) for v in obj[:MAX_LIST]]
        if len(obj) > MAX_LIST:
            out.append(f"... {len(obj) - MAX_LIST} more items truncated")
        return out
    if isinstance(obj, str) and cloud:
        return obj.replace(HOME, "~")
    return obj


def sanitize_result(result: dict, cloud: bool) -> str:
    text = json.dumps(_clean(result, cloud), default=str)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + ' ... [truncated]'
    return text


def cloud_allowed(cfg: dict) -> bool:
    return bool(cfg.get("cloud", {}).get("acknowledged")) or os.environ.get("NEUROSTIAS_ALLOW_CLOUD") == "1"


CLOUD_NOTICE = ("Cloud provider selected. Only skill result summaries (aggregate statistics, top-k lists, "
                "home directory redacted) will be sent; no expression matrices or per-cell tables. "
                "Acknowledge by setting cloud.acknowledged: true in configs/llm.yaml or NEUROSTIAS_ALLOW_CLOUD=1.")
