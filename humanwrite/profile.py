"""Author profile: everything the pipeline needs about the rough draft's
author, computed from the rough draft alone.

The profile has two parts:

* deterministic stats — readability numbers, structure map, sentence list
  (all from humanwrite.metrics)
* qualitative inventories — error slips, opinion material, and voice
  exemplars — found by lightweight heuristics

The qualitative part is intentionally conservative: it feeds the generation
prompt and the error budget, so a false positive costs little, but a miss is
worse. Heuristics here are simple and local.

CLI:  python -m humanwrite.profile --rough rough.md [--out author_profile.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Any

from humanwrite import metrics

# --------------------------------------------------------------------------
# error inventory heuristics
# --------------------------------------------------------------------------

BE_VERBS = {"am", "is", "are", "was", "were", "been", "being", "be"}
MODALS = {"can", "could", "will", "would", "shall", "should", "may", "might",
          "must", "do", "does", "did", "have", "has", "had"}

#: subject + finite-verb marker: does this clause stand on its own?
INDEPENDENT_RE = re.compile(
    r"\b(?:I|you|he|she|it|we|they|who|that|what|everyone|everything|something|"
    r"no one|nobody|somebody|people|parents|teachers|kids|students|girls|"
    r"boys|children|families|society|government)\s+"
    r"(?:am|is|are|was|were|have|has|had|do|does|did|can|could|will|would|"
    r"shall|should|may|might|must|know|knows|knew|think|thinks|thought|"
    r"want|wants|wanted|need|needs|needed|believe|believes|believed|"
    r"feel|feels|felt|see|sees|saw|get|gets|got|go|goes|went|make|makes|"
    r"made|take|takes|took|say|says|said|work|works|worked|live|lives|"
    r"lived|come|comes|came)\b",
    re.IGNORECASE,
)

INFORMAL_RE = re.compile(
    r"\b(?:gonna|wanna|gotta|kinda|sorta|stuff|thing(s)?\s+like|a lot|"
    r"lots of|basically|honestly|I mean|you know|oh well|yeah,?|no,? like|"
    r"super|reallly|really really)\b",
    re.IGNORECASE,
)

AGREEMENT_SLIPS = [
    (re.compile(r"\b(?:he|she|it)\s+(?:were|are|have|do|does)\b", re.I), "singular subject, plural verb"),
    (re.compile(r"\bthey\s+(?:was|has|does)\b", re.I), "plural subject, singular verb"),
    (re.compile(r"\bI\s+(?:was|has)\b", re.I), "'I was/has' slip"),
    (re.compile(r"\b(?:everyone|nobody|somebody|each)\s+(?:were|are|have|do)\b", re.I),
     "indefinite singular subject, plural verb"),
    (re.compile(r"\b(?:the|a|an|my|his|her|their) people\s+(?:was|is|has)\b", re.I),
     "plural noun, singular verb"),
]


def _sentence_bounds(text: str) -> list[tuple[int, int, str]]:
    """(offset, line, sentence_text) for every sentence."""
    out = []
    cursor = 0
    for sent in metrics.split_sentences(text):
        idx = text.find(sent, cursor)
        if idx < 0:
            idx = text.find(sent)
        cursor = idx + len(sent) if idx >= 0 else cursor
        if idx < 0:
            continue
        line = text[:idx].count("\n")
        out.append((idx, line, sent))
    return out


def find_comma_splices(text: str) -> list[dict[str, Any]]:
    out = []
    for off, line, sent in _sentence_bounds(text):
        commas = [m.start() for m in re.finditer(r",", sent)]
        for i, c in enumerate(commas):
            if i + 1 < len(commas) and commas[i + 1] - c < 45:
                continue  # multiple commas = probably a list, not a splice
            left, right = sent[:c], sent[c + 1:]
            if INDEPENDENT_RE.search(left) and INDEPENDENT_RE.search(right) and \
               len(right.split()) >= 3 and not right.lower().startswith(
                   ("and ", "but ", "so ", "which ", "that ", "because ",
                    "although ", "though ", "when ", "if ", "while ")):
                out.append({
                    "type": "comma splice",
                    "line": line + 1,
                    "sentence": sent[:90],
                    "note": "two independent clauses joined by a comma",
                })
                break
    return out


def find_fragments(text: str) -> list[dict[str, Any]]:
    out = []
    for off, line, sent in _sentence_bounds(text):
        words = metrics.word_tokens(sent)
        if len(words) < 3:
            continue
        has_finite = False
        for w in words:
            if w in BE_VERBS or w in MODALS:
                has_finite = True
                break
            if re.match(r"^[a-z]{3,}(?:s|ed)$", w) and w not in (
                    "this", "that", "these", "those", "has", "was", "is",
                    "its", "its", "lets", "gets", "must", "lets"):
                # a plain verb ending only counts as finite with a subject
                # before it; keep the heuristic loose (we allow false misses)
                has_finite = True
                break
        if not has_finite:
            out.append({
                "type": "fragment",
                "line": line + 1,
                "sentence": sent[:90],
                "note": "no finite verb found",
            })
    return out


def find_informal(text: str) -> list[dict[str, Any]]:
    out = []
    for off, line, sent in _sentence_bounds(text):
        for m in INFORMAL_RE.finditer(sent):
            out.append({
                "type": "informal phrasing",
                "line": line + 1,
                "sentence": sent[:90],
                "note": f"informal: {m.group(0)!r}",
            })
    return out


def find_agreement_slips(text: str) -> list[dict[str, Any]]:
    out = []
    for off, line, sent in _sentence_bounds(text):
        for rx, note in AGREEMENT_SLIPS:
            for m in rx.finditer(sent):
                out.append({
                    "type": "agreement slip",
                    "line": line + 1,
                    "sentence": sent[:90],
                    "note": f"{note}: {m.group(0)!r}",
                })
    return out


def find_overused_words(text: str, top_n: int = 5) -> list[dict[str, Any]]:
    words = metrics.word_tokens(text)
    if not words:
        return []
    counts = Counter(w for w in words if len(w) >= 4 and w not in {
        "that", "this", "with", "have", "from", "they", "them", "been",
        "would", "could", "should", "because", "about", "there", "their",
        "which", "when", "where", "what", "people", "would",
    })
    threshold = max(3, len(words) / 60)
    out = []
    for word, n in counts.most_common(top_n):
        if n >= threshold:
            out.append({
                "type": "overused word",
                "line": None,
                "sentence": None,
                "note": f"{word!r} appears {n} times",
            })
    return out


def error_inventory(text: str) -> list[dict[str, Any]]:
    errs: list[dict[str, Any]] = []
    errs += find_comma_splices(text)
    errs += find_fragments(text)
    errs += find_informal(text)
    errs += find_agreement_slips(text)
    errs += find_overused_words(text)
    return errs


# --------------------------------------------------------------------------
# opinion inventory
# --------------------------------------------------------------------------

FIRST_PERSON_RE = re.compile(r"\b(?:I|me|my|mine|myself)\b")
ANECDOTE_RE = re.compile(
    r"\b(?:when I|I (?:remember|was|were|saw|heard|felt|knew|thought)|"
    r"my (?:mom|mother|dad|father|mommy|papa|family|neighbor|teacher|"
    r"grandma|grandpa|sister|brother)|last (?:year|summer|winter|fall))\b",
    re.IGNORECASE,
)
EXAMPLE_RE = re.compile(
    r"\b(?:for example|for instance|in chapter|on page|pages? \d|"
    r"\d{1,2}\s*(?:year|percent|%|dollar|\$))\b",
    re.IGNORECASE,
)


def opinion_inventory(text: str) -> dict[str, Any]:
    first_person = []
    anecdotes = []
    examples = []
    for off, line, sent in _sentence_bounds(text):
        if FIRST_PERSON_RE.search(sent):
            first_person.append({"line": line + 1, "sentence": sent[:120]})
        if ANECDOTE_RE.search(sent):
            anecdotes.append({"line": line + 1, "sentence": sent[:120]})
        if EXAMPLE_RE.search(sent):
            examples.append({"line": line + 1, "sentence": sent[:120]})
    return {
        "first_person_claims": first_person,
        "anecdotes": anecdotes,
        "concrete_examples": examples,
    }


# --------------------------------------------------------------------------
# voice exemplars
# --------------------------------------------------------------------------

def pick_voice_exemplars(text: str, n: int = 3) -> list[dict[str, Any]]:
    """Pick structurally diverse paragraphs as voice exemplars.

    Selection: the longest paragraph, the shortest multi-sentence paragraph,
    and the paragraph with the most error signals (the "imperfect" one).
    Deduplicated by index; at most ``n``.
    """
    paras = [p for p in text.split("\n\n") if metrics.tokenize_words(p)]
    if not paras:
        return []
    stats = [
        {"index": i, "text": p,
         "words": len(metrics.tokenize_words(p)),
         "sentences": len(metrics.split_sentences(p)),
         "errors": len(_para_errors(p))}
        for i, p in enumerate(paras)
    ]
    chosen: dict[int, dict[str, Any]] = {}
    ranked = sorted(stats, key=lambda s: -s["words"])
    if ranked:
        chosen[ranked[0]["index"]] = {**ranked[0], "reason": "longest paragraph"}
    multi = [s for s in stats if s["sentences"] >= 2]
    if multi:
        short = min(multi, key=lambda s: s["words"])
        if short["index"] not in chosen:
            chosen[short["index"]] = {**short, "reason": "shortest multi-sentence paragraph"}
    with_err = [s for s in stats if s["errors"] > 0]
    if with_err:
        rough = max(with_err, key=lambda s: s["errors"])
        if rough["index"] not in chosen:
            chosen[rough["index"]] = {**rough, "reason": "most imperfect (kept deliberately)"}
    if len(chosen) < n:
        for s in stats:
            if len(chosen) >= n:
                break
            if s["index"] not in chosen:
                chosen[s["index"]] = {**s, "reason": "additional voice sample"}
    out = [chosen[i] for i in sorted(chosen)]
    return out[:n]


def _para_errors(p: str) -> list[dict[str, Any]]:
    return error_inventory(p)


# --------------------------------------------------------------------------
# profile assembly
# --------------------------------------------------------------------------

def build_profile(text: str) -> dict[str, Any]:
    stats = metrics.compute_stats(text)
    sentences = stats.pop("sentences")
    profile: dict[str, Any] = {
        "stats": stats,
        "error_inventory": error_inventory(text),
        "opinion_inventory": opinion_inventory(text),
        "voice_exemplars": pick_voice_exemplars(text),
        "sentence_count": len(sentences),
    }
    return profile


def error_budget_size(text: str, max_errors: int = 3) -> int:
    """min(3, errors per 500 words) — how many slips to re-introduce."""
    words = len(metrics.word_tokens(text))
    errs = len(error_inventory(text))
    per_500 = errs / max(words / 500.0, 1.0)
    return max(0, min(max_errors, int(round(per_500))))


def write_profile(profile: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2, ensure_ascii=False)


def print_summary(profile: dict[str, Any], log=print) -> None:
    s = profile["stats"]
    log("author profile:")
    log(f"  words: {s['word_count']}   sentences: {s['sentence_count']}")
    log(f"  FK grade: {s['fk_grade']}   Flesch ease: {s['flesch_ease']}")
    log(f"  mean sentence length: {s['mean_sentence_length']} words"
        f"   (CoV {s['sentence_length_cov']})")
    log(f"  TTR: {s['ttr']}   contractions: {s['contraction_rate']}/100 words"
        f"   passive: {s['passive_rate']}")
    struct = s["structure"]
    if struct["granularity"] == "section":
        for sec, prop in zip(struct["sections"], struct["section_proportions"]):
            log(f"  section: {sec['title']!r} — {sec['words']} words ({prop:.0%}), "
                f"{sec['paragraphs']} paragraphs")
    else:
        log(f"  paragraphs: {struct['n_paragraphs']} "
            f"(proportions: "
            f"{', '.join(f'{p:.2f}' for p in struct['paragraph_proportions'][:12])}"
            f"{'...' if len(struct['paragraph_proportions']) > 12 else ''})")
    errs = profile["error_inventory"]
    log(f"  error inventory: {len(errs)} slips "
        f"({', '.join(sorted({e['type'] for e in errs})) or 'none found'})")
    opp = profile["opinion_inventory"]
    log(f"  opinion material: {len(opp['first_person_claims'])} first-person, "
        f"{len(opp['anecdotes'])} anecdote, {len(opp['concrete_examples'])} example")
    log(f"  error budget (to re-introduce): {error_budget_size_from_profile(profile)}")


def error_budget_size_from_profile(profile: dict[str, Any]) -> int:
    words = profile["stats"]["word_count"]
    errs = len(profile["error_inventory"])
    per_500 = errs / max(words / 500.0, 1.0)
    return max(0, min(3, int(round(per_500))))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an author profile from a rough draft")
    ap.add_argument("--rough", required=True)
    ap.add_argument("--out", default="author_profile.json")
    args = ap.parse_args()
    with open(args.rough, encoding="utf-8") as fh:
        text = fh.read()
    profile = build_profile(text)
    write_profile(profile, args.out)
    print_summary(profile)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
