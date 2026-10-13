"""Configuration loading for the humanwrite pipeline.

The config file is YAML. To keep the package stdlib-only, a minimal YAML
subset loader (nested dicts, scalar lists, comments) is used when PyYAML is
not importable. All tunables live in config.yaml so users can edit one file.
"""

from __future__ import annotations

import copy
import os
from typing import Any

DEFAULTS: dict[str, Any] = {
    "endpoint": "http://10.60.50.11:9292/v1",
    "model": "Qwen3.8-27B",
    "api_key": "not-needed",
    "timeout_seconds": 1800,
    "max_retries": 3,
    # Extra keys merged verbatim into every chat request body. Empty by
    # default. For reasoning models, thinking stays ON (the server's
    # thinking budget is large enough) and sampling.max_tokens must be big
    # enough to hold thinking + content. E.g. to force thinking OFF on
    # Qwen3-style models: {"chat_template_kwargs": {"enable_thinking": false}}.
    "extra_body": {},
    "sampling": {
        "temperature": 1.1,
        "top_p": 0.95,
        "presence_penalty": 1.0,
        "n_candidates": 3,
        # Generous: reasoning models spend part of this budget on hidden
        # thinking before any content token is produced (the server-side
        # thinking budget alone is 32768, so this must cover it + content).
        "max_tokens": 65536,
    },
    "gate": {
        "fk_tolerance": 2.0,
        "msl_ratio_low": 0.80,
        "msl_ratio_high": 1.25,
        "ttr_ratio_low": 0.85,
        "ttr_ratio_high": 1.15,
        "contraction_rel_diff_max": 0.50,
        "contraction_abs_diff_max": 1.0,
        "contraction_low_threshold": 1.0,
        "anchor_min": 0.30,
        "anchor_jaccard_min": 0.85,
        "length_ratio_low": 0.9,
        "length_ratio_high": 1.5,
        "max_repair_iterations": 3,
        # --notes word cap; 0 = unlimited (endpoint context window is big
        # enough to hold the full source material).
        "max_notes_words": 0,
    },
    "tell_scan": {
        # T05: a sentence-opening word repeated this many times in the whole
        # text is a tic. Default 7; 3-5 is normal in real student essays
        # (and matches the rough draft's own voice), so it should not flag.
        "repeated_opening_min": 7,
        "vocabulary": [
            "delve", "landscape", "tapestry", "testament", "pivotal",
            "seamless", "robust", "elevate", "foster", "leverage",
            "crucial", "realm", "showcasing", "underscore", "unlock",
            "navigate", "intricate", "multifaceted", "utilize",
            "meticulous", "paramount", "synergy", "holistic", "game-changer",
        ],
        "ing_riders": [
            "symbolizing", "reflecting", "showcasing", "underscoring",
            "highlighting", "demonstrating", "exemplifying", "signifying",
            "emphasizing", "emphasising",
        ],
        "extra_words": [],
        "disabled_patterns": [],
    },
    "constrain": {
        "section_proportion_tolerance": 0.20,
        "error_budget_per_500_words": 3,
        "humanizer_length_tolerance": 0.10,
    },
}


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(token: str) -> Any:
    token = token.strip()
    if token == "":
        return None
    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
        return token[1:-1]
    low = token.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~"):
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def _strip_comment(line: str) -> str:
    out = []
    in_quote = ""
    for ch in line:
        if in_quote:
            out.append(ch)
            if ch == in_quote:
                in_quote = ""
            continue
        if ch in ("'", '"'):
            in_quote = ch
            out.append(ch)
            continue
        if ch == "#":
            break
        out.append(ch)
    return "".join(out).rstrip()


def _parse_inline_list(text: str) -> list[Any]:
    inner = text.strip()[1:-1].strip()
    if not inner:
        return []
    parts: list[str] = []
    depth = 0
    quote = ""
    buf: list[str] = []
    for ch in inner:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
        elif ch == "[":
            depth += 1
            buf.append(ch)
        elif ch == "]":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return [_coerce(p) for p in parts if p.strip() != ""]


def _parse_scalar_list(items: list[str]) -> list[Any]:
    return [_coerce(i) for i in items]


def parse_yaml(text: str) -> dict[str, Any]:
    """Parse the YAML subset used by humanwrite config files.

    Supports: nested mappings (2-space indentation per level), inline lists
    ``[a, b]``, block lists (``- item``), scalars, and ``#`` comments.
    """
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        lines.append((indent, line.strip()))

    root: dict[str, Any] = {}
    # stack of (indent, container); root sits at indent -1 so it never pops
    stack: list[tuple[int, Any]] = [(-1, root)]

    idx = 0
    while idx < len(lines):
        indent, content = lines[idx]
        while indent < stack[-1][0]:
            stack.pop()
        if not content.startswith("- "):
            # a mapping key: any entry at its own indent or deeper is stale
            while len(stack) > 1 and indent <= stack[-1][0]:
                stack.pop()
        container = stack[-1][1]

        if content.startswith("- "):
            while len(stack) > 1 and stack[-1][0] > indent:
                stack.pop()
            if isinstance(stack[-1][1], list):
                stack[-1][1].append(_coerce(content[2:]))
            idx += 1
            continue

        if ":" not in content:
            idx += 1
            continue
        key, _, value = content.partition(":")
        key = key.strip().strip('"').strip("'")
        value = value.strip()
        if value == "":
            # look ahead: is the next line a block list at this or deeper indent?
            if idx + 1 < len(lines) and lines[idx + 1][0] >= indent and \
               lines[idx + 1][1].startswith("- "):
                new: Any = []
            else:
                new = {}
            if isinstance(container, dict):
                container[key] = new
            stack.append((indent, new))
        else:
            if value.startswith("[") and value.endswith("]"):
                item: Any = _parse_inline_list(value)
            else:
                item = _coerce(value)
            if isinstance(container, dict):
                container[key] = item
        idx += 1

    return root


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load config.yaml (searching upward from CWD / this package) and merge
    over DEFAULTS so every key always exists.
    """
    if path is None:
        path = find_config_path()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text) or {}
        except ImportError:
            data = parse_yaml(text)
    else:
        data = {}
    return _deep_merge(DEFAULTS, data if isinstance(data, dict) else {})


def find_config_path() -> str | None:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "config.yaml"),
        os.path.join(here, "..", ".claude", "skills", "humanwrite", "config.yaml"),
        os.path.join(os.getcwd(), "config.yaml"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None


def get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
