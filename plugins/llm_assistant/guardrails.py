"""Numeric grounding check: every number in the answer should come from a tool result or the question.

Numbers are matched with tolerance for rounding to the precision written in the
answer (e.g. 0.8361 in a tool result supports "0.84" and "83.6%"). Small integers
(0-10) and four-digit years are not checked. Unverified numbers are listed below
the answer so the reader knows which figures are not traceable to a computation.
"""

from __future__ import annotations

import json
import re

NUM = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?%?")


def _numbers_in_text(text: str) -> list[str]:
    return [m.group(0) for m in NUM.finditer(text)]


def _collect(obj, out: set):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _collect(v, out)
            _collect_str(str(k), out)
    elif isinstance(obj, list):
        for v in obj:
            _collect(v, out)
    elif isinstance(obj, str):
        try:
            _collect(json.loads(obj), out)
        except (json.JSONDecodeError, ValueError):
            _collect_str(obj, out)


def _collect_str(s: str, out: set):
    for tok in _numbers_in_text(s):
        try:
            out.add(float(tok.replace(",", "").rstrip("%")))
        except ValueError:
            pass


def _supported(token: str, sources: set) -> bool:
    is_pct = token.endswith("%")
    t = token.replace(",", "").rstrip("%")
    v = float(t)
    decimals = len(t.split(".")[1]) if "." in t else 0
    tol = 0.5 * 10 ** (-decimals) + 1e-12
    for s in sources:
        for cand in ((s, s * 100) if is_pct else (s, s * 100, s / 100)):
            if abs(cand - v) <= tol or (abs(v) > 0 and abs(cand - v) / abs(v) < 1e-3):
                return True
    return False


def check(answer: str, tool_results: list[str], question: str = "") -> dict:
    sources: set = set()
    for r in tool_results:
        _collect(r, sources)
    _collect_str(question, sources)
    unverified, verified = [], []
    for tok in _numbers_in_text(answer):
        t = tok.replace(",", "").rstrip("%")
        try:
            v = float(t)
        except ValueError:
            continue
        if ("." not in t and 0 <= v <= 10) or ("." not in t and 1900 <= v <= 2100):
            continue
        (verified if _supported(tok, sources) else unverified).append(tok)
    return {"n_numbers_checked": len(verified) + len(unverified), "verified": verified, "unverified": unverified,
            "grounded": not unverified}
