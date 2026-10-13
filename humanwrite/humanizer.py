"""Pass 2: humanizer.

Sends the winning candidate plus the AI-tell pattern list through the local
model and asks it to fix every tell found — wording and rhythm only, facts,
claims, and citations preserved exactly, no sections added or removed,
length within +/-10%.

The prompt content (the pattern list) is read from
references/ai_tells.md so the model and tell_scan.py share one list.

CLI:  python -m humanwrite.humanizer --draft candidate.md --out humanized.md
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from humanwrite.config import load_config, get
from humanwrite.llm import client_from_config


def pattern_list_text(cfg: dict[str, Any] | None = None) -> str:
    """Read references/ai_tells.md next to this package."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "references", "ai_tells.md")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def build_messages(draft: str, cfg: dict[str, Any],
                   pattern_list: str | None = None) -> list[dict[str, str]]:
    tol = get(cfg or {}, "constrain.humanizer_length_tolerance", 0.10)
    system = (
        "You are proofreading a student's draft. Your only job is to remove the "
        "specific writing patterns listed in the pattern list below. Change "
        "wording and rhythm only. You must preserve every fact, claim, number, "
        "quote, and page number exactly as written. Do not add or remove "
        "sections, paragraphs, or bullet points. Keep the total length within "
        f"{tol:.0%} of the input. Output only the revised draft, no commentary."
    )
    user = (
        "--- DRAFT TO REVISE ---\n"
        f"{draft}\n\n"
        "--- PATTERN LIST (fix every occurrence of these) ---\n"
        f"{pattern_list if pattern_list is not None else pattern_list_text()}\n\n"
        "Rules:\n"
        "1. Fix every pattern listed above. Where a pattern is a word, replace "
        "it with an ordinary word the student would use.\n"
        "2. Where a pattern is a construction, restate the sentence in plain "
        "words. Do not just delete the sentence — keep its content.\n"
        "3. Do not change any fact, claim, number, quote, or page number.\n"
        "4. Do not add sections, drop sections, or restructure paragraphs.\n"
        f"5. Keep the student's voice and the draft's length within {tol:.0%}.\n"
        "6. Output only the revised draft."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def run(draft: str, client, cfg: dict[str, Any], log=print) -> str:
    s = cfg.get("sampling", {})
    messages = build_messages(draft, cfg)
    out = client.chat(
        messages,
        temperature=s.get("temperature"),
        top_p=s.get("top_p"),
        presence_penalty=s.get("presence_penalty"),
        max_tokens=s.get("max_tokens"),
    )
    log(f"humanizer: input {len(draft)} chars -> output {len(out)} chars")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Pass 2: remove AI tells via the local model")
    ap.add_argument("--draft", required=True)
    ap.add_argument("--out", default="humanized.md")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    with open(args.draft, encoding="utf-8") as fh:
        draft = fh.read()
    client = client_from_config(cfg, log=print)
    result = run(draft, client, cfg)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(result)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
