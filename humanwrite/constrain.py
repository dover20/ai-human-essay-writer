"""Pass 3: constrain the draft back to the rough draft.

Sends (current draft, rough draft, author profile) through the local model
with six hard constraints stated in the prompt:

  1. same section structure and order as the rough draft
  2. anchor preservation >= anchor_min (default 0.30), near-identical =
     token Jaccard >= anchor_jaccard_min (default 0.85)
  3. error budget: re-introduce min(3, errors per 500 words) of the profile's
     inventory errors, in the same sections
  4. section length proportions within 20% of the rough draft's
  5. no remaining AI-tell vocabulary or constructions
  6. no new sections, topics, or citations not in the notes

CLI:  python -m humanwrite.constrain --draft humanized.md --rough rough.md
        --profile author_profile.json --out constrained.md
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from humanwrite.config import load_config, get
from humanwrite.llm import client_from_config
from humanwrite.profile import error_budget_size_from_profile


def profile_summary(profile: dict[str, Any]) -> str:
    s = profile["stats"]
    lines = [
        f"words: {s['word_count']}; sentences: {s['sentence_count']}",
        f"FK grade: {s['fk_grade']}; mean sentence length: {s['mean_sentence_length']}",
        f"TTR: {s['ttr']}; contractions per 100 words: {s['contraction_rate']}",
        "section structure:",
    ]
    struct = s["structure"]
    if struct["granularity"] == "section":
        for sec, prop in zip(struct["sections"], struct["section_proportions"]):
            lines.append(f"  - {sec['title']} ({sec['words']} words, {prop:.0%} of the draft)")
    else:
        lines.append(f"  - {struct['n_paragraphs']} paragraphs; proportions: "
                     + ", ".join(f"{p:.2f}" for p in struct["paragraph_proportions"]))
    errs = profile["error_inventory"]
    lines.append(f"error slips found in the rough draft ({len(errs)}):")
    for e in errs[:12]:
        loc = f"line {e['line']}" if e.get("line") else "anywhere"
        lines.append(f"  - [{e['type']}] {e.get('note', '')} ({loc})"
                     + (f" e.g. {e['sentence']!r}" if e.get("sentence") else ""))
    return "\n".join(lines)


def build_messages(draft: str, rough: str, profile: dict[str, Any],
                   cfg: dict[str, Any]) -> list[dict[str, str]]:
    g = cfg.get("gate", {})
    anchor_pct = f"{g.get('anchor_min', 0.30):.0%}"
    budget = error_budget_size_from_profile(profile)
    vocab = ", ".join(cfg.get("tell_scan", {}).get("vocabulary", []))
    system = (
        "You are the student again, making final edits to your own draft. You "
        "are bringing the draft back in line with the rough draft you started "
        "from. Follow the hard constraints exactly. Output only the revised "
        "draft."
    )
    user = (
        "--- CURRENT DRAFT ---\n"
        f"{draft}\n\n"
        "--- YOUR ROUGH DRAFT (the anchor) ---\n"
        f"{rough}\n\n"
        "--- YOUR WRITING PROFILE ---\n"
        f"{profile_summary(profile)}\n\n"
        "Hard constraints — every one must hold:\n"
        "1. Same sections, same order as the rough draft. No new sections, no "
        "new topics. No citation that is not in the rough draft or your notes.\n"
        f"2. Anchor preservation: at least {anchor_pct} of the sentences in your "
        "revised draft must be word-for-word identical to, or nearly identical "
        "to, a sentence in the rough draft. Real students leave most of their "
        "own draft in place. Copy strong rough-draft sentences back in.\n"
        f"3. Error budget: re-introduce exactly {budget} of the small errors "
        "listed in the profile, in the same sections where the rough draft had "
        "them. Leave the rest of the grammar clean.\n"
        "4. Section lengths within 20% of the rough draft's proportions.\n"
        "5. Remove every AI-tell word and construction: no "
        f"{vocab}; no 'not just X but Y'; no 'In conclusion'; no "
        "'It is worth noting'; no em-dash glue.\n"
        "6. Keep the rough draft's rhythm: a mix of short and long sentences.\n\n"
        "Make the smallest edits that satisfy all six constraints. Output only "
        "the revised draft."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def run(draft: str, rough: str, profile: dict[str, Any], client,
        cfg: dict[str, Any], log=print) -> str:
    s = cfg.get("sampling", {})
    messages = build_messages(draft, rough, profile, cfg)
    out = client.chat(
        messages,
        temperature=s.get("temperature"),
        top_p=s.get("top_p"),
        presence_penalty=s.get("presence_penalty"),
        max_tokens=s.get("max_tokens"),
    )
    log(f"constrain: input {len(draft)} chars -> output {len(out)} chars")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Pass 3: constrain back to the rough draft")
    ap.add_argument("--draft", required=True)
    ap.add_argument("--rough", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", default="constrained.md")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    with open(args.draft, encoding="utf-8") as fh:
        draft = fh.read()
    with open(args.rough, encoding="utf-8") as fh:
        rough = fh.read()
    with open(args.profile, encoding="utf-8") as fh:
        profile = json.load(fh)
    client = client_from_config(cfg, log=print)
    result = run(draft, rough, profile, client, cfg)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(result)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
