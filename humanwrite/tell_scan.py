"""AI-tell vocabulary and construction scanner.

Each pattern has a stable id (T01, T02, ...). The same ids are used in
references/ai_tells.md so the humanizer pass and this scanner share one
canonical list. The word-lists are config-driven (``tell_scan`` in
config.yaml); the structural patterns are built-in.

Output: ``scan_text(text, cfg) -> list[violation]`` where each violation is
``{"line": int, "col": int, "pattern": id, "text": matched, "note": ...}``.
"""

from __future__ import annotations

import re
from typing import Any, Callable

Pattern = tuple[str, Callable[[str], list[tuple[int, int, str, str]]]]
# id -> matcher: line, col, matched_text, note

# --------------------------------------------------------------------------
# structural pattern matchers (all return spans on one line of text)
# --------------------------------------------------------------------------


def _contrast_matcher():
    """Staged "not just X but Y" / "more than just X" constructions.

    Plain "not only" and short "X, not Y" (two words or fewer between the
    comma and "not") are normal English and do not count.
    """
    rxs = [
        (re.compile(r"\bnot\s+(?:just|merely|simply|only)\s+[^.?!]{0,45}?\s+but\s+(?:also\s+)?\w", re.I),
         "'not just X but Y' contrast"),
        (re.compile(r"\bbut\s+not\s+just\s+\w", re.I), "'but not just X' contrast"),
        (re.compile(r"\bmuch\s+more\s+than\s+just\s+\w", re.I), "'more than just X'"),
        (re.compile(r"\bmore\s+than\s+just\s+(?:a|an|the)\s+\w[\w' -]{0,30}$", re.I),
         "'more than just X' (sentence-final)"),
        (re.compile(r"\bnot\s+merely\s+[^.?!]{0,40}?\s+but\s+\w", re.I), "'not merely X but Y'"),
    ]
    def match(line: str):
        out = []
        for rx, note in rxs:
            for m in rx.finditer(line):
                out.append((m.start(), m.end(), m.group(0), note))
        return out
    return match


def _staged_opener_matcher():
    rxs = [
        (re.compile(r"\bhere'?s the (?:thing|truth|beauty|point|real)\b", re.I),
         "staged run-up opener"),
        (re.compile(r"^\s*(?:honestly|to be honest|really|look)[,]?$", re.I),
         "staged run-up opener"),
        (re.compile(r"\bthe best part[,:]\s*", re.I), "'the best part:' run-up"),
        (re.compile(r"\b(?!let\s+that\s+sink\s+in)let me tell you\b", re.I), "staged run-up opener"),
        (re.compile(r"\bthe bottom line\b[,:]", re.I), "staged run-up opener"),
    ]
    def match(line: str):
        out = []
        for rx, note in rxs:
            for m in rx.finditer(line):
                out.append((m.start(), m.end(), m.group(0), note))
        return out
    return match


def _dramatic_closer_matcher():
    rxs = [
        (re.compile(r"\blet that sink in\b", re.I), "one-line dramatic closer"),
        (re.compile(r"\bthat'?s (?:the|what makes this) (?:point|difference|hard part)\b", re.I),
         "one-line dramatic closer"),
        (re.compile(r"\bthat is (?:what|where) (?:it|this) (?:all|really) (?:is|comes)\b", re.I),
         "one-line dramatic closer"),
        (re.compile(r"^\s*(?:So|That)[\s,]+that'?s\b", re.I), "one-line dramatic closer"),
        (re.compile(r"\bkeep in mind that\b", re.I), "one-line dramatic closer"),
        (re.compile(r"\bthis is (?:what|why) (?:it|all) (?:matters|counts|really)\b", re.I),
         "one-line dramatic closer"),
    ]
    def match(line: str):
        out = []
        for rx, note in rxs:
            for m in rx.finditer(line):
                out.append((m.start(), m.end(), m.group(0), note))
        return out
    return match


def _triad_matcher():
    """Forced rule of three: 'X, Y, and Z' with all three short (<=4 words)."""
    rx = re.compile(
        r"\b(\w[\w' -]{0,20}?)\s*,\s*(\w[\w' -]{0,20}?)\s*,?\s+and\s+(\w[\w' -]{0,20}?)\b"
    )
    def match(line: str):
        out = []
        for m in rx.finditer(line):
            parts = (m.group(1).strip(), m.group(2).strip(), m.group(3).strip())
            lens = [len(p.split()) for p in parts]
            if all(1 <= l <= 3 for l in lens) and all(p for p in parts):
                out.append((m.start(), m.end(), m.group(0), "forced rule-of-three list"))
        return out
    return match


def _repeated_openings_matcher(config: dict):
    """Factory: sentence openings repeated ``repeated_opening_min`` times
    (default 7) in the whole text.

    A handful of repeated openers (3-5) is normal in real student essays and
    even a hallmark of a consistent voice, so the threshold is set high enough
    that only a genuine opening *tic* flags. Config-driven via
    ``tell_scan.repeated_opening_min``.
    """
    min_count = int(config.get("repeated_opening_min", 7))

    def match(text: str):
        """Returns (line_index, col, text, note) tuples."""
        from humanwrite.metrics import split_sentences
        from collections import Counter

        line_offsets = [0]
        for ln in text.split("\n"):
            line_offsets.append(line_offsets[-1] + len(ln) + 1)

        def linecol(off: int) -> tuple[int, int]:
            lo, hi = 0, len(line_offsets) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if line_offsets[mid] <= off:
                    lo = mid
                else:
                    hi = mid - 1
            return lo, off - line_offsets[lo]

        seen: Counter[str] = Counter()
        positions: dict[str, list[tuple[int, int, str, str]]] = {}
        for sent in split_sentences(text):
            idx = text.find(sent)
            if idx < 0:
                continue
            tokens = sent.strip().split()
            if not tokens:
                continue
            first = re.sub(r"[^a-z0-9']", "", tokens[0]).lower()
            if len(first) < 3:
                continue
            seen[first] += 1
            ln, col = linecol(idx)
            positions.setdefault(first, []).append((ln, col, sent[:24], "repeated sentence opening"))
        out = []
        for word, spans in positions.items():
            if seen[word] >= min_count:
                out.extend(spans)
        return out

    return match


def _dash_matcher():
    def match(line: str):
        out = []
        for m in re.finditer(r" — | – | - ", line):
            out.append((m.start(), m.end(), m.group(0), "em/en-dash as connector"))
        return out
    return match


def _stacked_qualifiers_matcher():
    rxs = [
        (re.compile(r"\b(?:it|this|they|it'?s) (?:was|is|were|be) (?:arguably|essentially|fundamentally|virtually) (?:a|an|the|really|truly)\b", re.I),
         "stacked qualifiers"),
        (re.compile(r"\b(very|quite|really|truly)\s+(very|quite|really|truly)\b", re.I),
         "stacked qualifiers"),
    ]
    def match(line: str):
        out = []
        for rx, note in rxs:
            for m in rx.finditer(line):
                out.append((m.start(), m.end(), m.group(0), note))
        return out
    return match


def _ing_rider_matcher(config: dict):
    riders = set(config.get("ing_riders", []))
    if not riders:
        return lambda line: []
    rx = re.compile(r"\b([a-z]+)(?:ing|ising|izing)\b", re.I)
    def match(line: str):
        out = []
        for m in rx.finditer(line):
            word = m.group(0).lower()
            if word in riders:
                out.append((m.start(), m.end(), m.group(0), "shallow -ing rider"))
        return out
    return match


def _vocabulary_matcher(config: dict):
    words = set(w.lower() for w in config.get("vocabulary", []))
    words |= set(w.lower() for w in config.get("extra_words", []))
    if not words:
        return lambda line: []
    # expand to common inflections so "delves", "leveraging", "navigated" hit
    expanded = set(words)
    for w in words:
        for suf in ("s", "es", "ed", "ing", "ly"):
            expanded.add(w + suf)
    rx = re.compile(r"\b([a-z][a-z'-]+)\b", re.I)
    def match(line: str):
        out = []
        for m in rx.finditer(line):
            if m.group(1).lower() in expanded:
                out.append((m.start(), m.end(), m.group(0), "AI vocabulary"))
        return out
    return match


def _serves_as_matcher():
    rxs = [
        (re.compile(r"\bserves (?:as|to) (?:a|an|the)\b", re.I), "'serves as' instead of is/has"),
        (re.compile(r"\bboasts (?:a|an|the)\b", re.I), "'boasts' instead of has"),
    ]
    def match(line: str):
        out = []
        for rx, note in rxs:
            for m in rx.finditer(line):
                out.append((m.start(), m.end(), m.group(0), note))
        return out
    return match


def _vague_authority_matcher():
    rxs = [
        (re.compile(r"\bexperts (?:agree|say|insist|claim)\b", re.I), "vague authority"),
        (re.compile(r"\bit is worth noting\b", re.I), "vague authority"),
        (re.compile(r"\bmany (?:people|experts|writers|scholars) (?:agree|believe|say|would say)\b", re.I),
         "vague authority"),
        (re.compile(r"\b(?!worthwhile)(?:of course,? it|clearly,? it|undeniably,? it)\b", re.I),
         "vague authority"),
        (re.compile(r"\b(?!in no small way)(?:in many ways,?|in one way or another,?|when it comes down to it,?)\b", re.I),
         "vague authority"),
    ]
    def match(line: str):
        out = []
        for rx, note in rxs:
            for m in rx.finditer(line):
                out.append((m.start(), m.end(), m.group(0), note))
        return out
    return match


def _conclusion_closer_matcher():
    rxs = [
        (re.compile(r"^\s*(?:in|to|as)\s+(?:conclusion|a conclusion|a final word|a close)\b[,:]", re.I),
         "'in conclusion' closer"),
        (re.compile(r"\bin closing\b[,:]", re.I), "'in conclusion' closer"),
        (re.compile(r"\bfinal(ly)?,?\s+(?:it|this|we)\b[,:]", re.I), "'in conclusion' closer"),
        (re.compile(r"\bto summarize\b[,:]", re.I), "'in conclusion' closer"),
    ]
    def match(line: str):
        out = []
        for rx, note in rxs:
            for m in rx.finditer(line):
                out.append((m.start(), m.end(), m.group(0), note))
        return out
    return match


def _restate_heading_matcher(text: str):
    """First sentence of a section restates the heading almost verbatim."""
    from humanwrite.metrics import _HEADING_RE, split_sentences
    out = []
    line_offsets = [0]
    for ln in text.split("\n"):
        line_offsets.append(line_offsets[-1] + len(ln) + 1)

    def linecol(off: int):
        lo, hi = 0, len(line_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_offsets[mid] <= off:
                lo = mid
            else:
                hi = mid - 1
        return lo, off - line_offsets[lo]

    for m in _HEADING_RE.finditer(text):
        heading = m.group(2).lower()
        # find the first sentence after the heading
        rest = text[m.end():]
        sents = split_sentences(rest)
        if not sents:
            continue
        first = sents[0].lower()
        # token overlap >= 0.7 means it's basically a restatement
        ht = {w for w in re.findall(r"[a-z0-9']+", heading) if len(w) > 2}
        ft = {w for w in re.findall(r"[a-z0-9']+", first) if len(w) > 2}
        if ht and ft and len(ht & ft) / min(len(ht), len(ft)) >= 0.7:
            abs_off = m.end() + rest.find(sents[0])
            ln, col = linecol(abs_off)
            out.append((ln, col, sents[0][:40], "heading restated as first sentence"))
    return out


def _decorative_bold_matcher(text: str):
    """**Label:** bold used decoratively (short label before a colon)."""
    rx = re.compile(r"^\s*\*\*([^*]{1,40})\*\*\s*:", re.MULTILINE)
    out = []
    for m in rx.finditer(text):
        ln = text[:m.start()].count("\n")
        col = m.start() - (text.rfind("\n", 0, m.start()) + 1)
        out.append((ln, col, m.group(0).strip(), "decorative bold label"))
    return out


def _title_case_headings_matcher(text: str):
    """Markdown headings written in Title Case decoratively."""
    rx = re.compile(r"^\s{0,3}#{1,6}\s+([A-Z][a-zA-Z0-9' -]{2,})\s*$", re.MULTILINE)
    out = []
    for m in rx.finditer(text):
        words = m.group(1).split()
        uppered = sum(1 for w in words if w[0].isupper())
        if len(words) >= 3 and uppered / len(words) >= 0.6 and \
           not all(w[0].isupper() and len(w) == 1 for w in words):
            ln = text[:m.start()].count("\n")
            col = m.start() - (text.rfind("\n", 0, m.start()) + 1)
            out.append((ln, col, m.group(0).strip(), "Title-Case decorative heading"))
    return out


def _curly_quotes_matcher(text: str):
    out = []
    for m in re.finditer(r"[“”‘’]", text):
        ln = text[:m.start()].count("\n")
        col = m.start() - (text.rfind("\n", 0, m.start()) + 1)
        out.append((ln, col, m.group(0), "curly quotes"))
    return out


def _chatbot_residue_matcher():
    rxs = [
        (re.compile(r"\bgreat question\b", re.I), "chatbot residue"),
        (re.compile(r"\bi hope this (?:helps|answers|clarifies)\b", re.I), "chatbot residue"),
        (re.compile(r"\bhere (?:is|are) (?:the )?(?:answer|summary)\b", re.I), "chatbot residue"),
        (re.compile(r"\bcertainly[!.,]?\s", re.I), "chatbot residue"),
        (re.compile(r"\b(?!certainly a)(?:as an ai|as a language model)\b", re.I), "chatbot residue"),
    ]
    def match(line: str):
        out = []
        for rx, note in rxs:
            for m in rx.finditer(line):
                out.append((m.start(), m.end(), m.group(0), note))
        return out
    return match


def _hedging_matcher():
    rxs = [
        (re.compile(r"\b(?:it should be noted that|it can be argued that|it could be said that)\b", re.I),
         "hedging disclaimer"),
        (re.compile(r"\bwhile (?:this|it) (?:may|might) (?:seem|appear|be|have)\b", re.I),
         "hedging disclaimer"),
        (re.compile(r"\b(?:one could argue that|one might argue that|it is important to remember that)\b", re.I),
         "hedging disclaimer"),
    ]
    def match(line: str):
        out = []
        for rx, note in rxs:
            for m in rx.finditer(line):
                out.append((m.start(), m.end(), m.group(0), note))
        return out
    return match


def _reexplain_matcher():
    rxs = [
        (re.compile(r"\bfor (?:those of you|anyone) who (?:is|are|may be) (?:unfamiliar|new to|not familiar)\b", re.I),
         "re-explaining the obvious"),
        (re.compile(r"\bas (?:many of you|we all) (?:know|know well)\b", re.I),
         "re-explaining the obvious"),
        (re.compile(r"\bas a (?:quick|brief) (?:background|refresher|reminder)\b", re.I),
         "re-explaining the obvious"),
        (re.compile(r"\bfor context[,:]?\s", re.I), "re-explaining the obvious"),
    ]
    def match(line: str):
        out = []
        for rx, note in rxs:
            for m in rx.finditer(line):
                out.append((m.start(), m.end(), m.group(0), note))
        return out
    return match


# --------------------------------------------------------------------------
# pattern table
# --------------------------------------------------------------------------

def build_patterns(cfg: dict[str, Any], scan_cfg: dict[str, Any] | None = None):
    scan_cfg = scan_cfg or {}
    disabled = set(scan_cfg.get("disabled_patterns", []))

    table: list[tuple[str, str, Callable]] = [
        ("T01", "contrast", _contrast_matcher()),
        ("T02", "staged_opener", _staged_opener_matcher()),
        ("T03", "dramatic_closer", _dramatic_closer_matcher()),
        ("T04", "rule_of_three", _triad_matcher()),
        ("T05", "repeated_openings", _repeated_openings_matcher(scan_cfg)),
        ("T06", "dash_connector", _dash_matcher()),
        ("T07", "stacked_qualifiers", _stacked_qualifiers_matcher()),
        ("T08", "vocabulary", _vocabulary_matcher(scan_cfg)),
        ("T09", "ing_rider", _ing_rider_matcher(scan_cfg)),
        ("T10", "serves_as", _serves_as_matcher()),
        ("T11", "vague_authority", _vague_authority_matcher()),
        ("T12", "conclusion_closer", _conclusion_closer_matcher()),
        ("T13", "restate_heading", _restate_heading_matcher),
        ("T14", "decorative_bold", _decorative_bold_matcher),
        ("T15", "title_case_heading", _title_case_headings_matcher),
        ("T16", "curly_quotes", _curly_quotes_matcher),
        ("T17", "chatbot_residue", _chatbot_residue_matcher()),
        ("T18", "hedging", _hedging_matcher()),
        ("T19", "reexplain", _reexplain_matcher()),
    ]
    return [(pid, name, fn) for pid, name, fn in table if pid not in disabled]


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

WHOLE_TEXT_PATTERNS = ("repeated_openings", "restate_heading", "decorative_bold",
                       "title_case_heading", "curly_quotes")


def scan_text(text: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Scan text and return one dict per violation:
    {"line": 1-based line number, "col": column, "pattern": id, "name": str,
     "text": matched text, "note": human description}.
    """
    scan_cfg = cfg.get("tell_scan", {})
    violations: list[dict[str, Any]] = []
    lines = text.split("\n")
    for pid, name, fn in build_patterns(cfg, scan_cfg):
        spans: list[tuple[int, int, str, str]] = []
        try:
            if name in WHOLE_TEXT_PATTERNS:
                spans = fn(text)  # (line_index, col, text, note)
                for ln, col, matched, note in spans:
                    violations.append(_violation(ln, col, pid, name, matched, note))
            else:
                for idx, line in enumerate(lines):
                    for start, _, matched, note in fn(line):
                        violations.append(_violation(idx, start, pid, name, matched, note))
        except Exception as err:  # a broken pattern must not kill the scan
            violations.append(_violation(0, 0, pid, name, "", f"PATTERN ERROR: {err}"))
    violations.sort(key=lambda v: (v["line"], v["col"]))
    return violations


def _violation(line_index: int, col: int, pid: str, name: str,
               matched: str, note: str) -> dict[str, Any]:
    return {
        "line": line_index + 1,
        "col": col,
        "pattern": pid,
        "name": name,
        "text": matched,
        "note": note,
    }


def scan_report(violations: list[dict[str, Any]]) -> str:
    if not violations:
        return "tell scan: 0 violations"
    lines = [f"tell scan: {len(violations)} violations"]
    for v in violations:
        lines.append(f"  line {v['line']}: [{v['pattern']} {v['name']}] {v['text']!r}"
                     + (f"  ({v['note']})" if v["note"] else ""))
    return "\n".join(lines)


def main() -> None:
    import argparse
    from humanwrite.config import load_config
    ap = argparse.ArgumentParser(description="Scan text for AI-tell patterns")
    ap.add_argument("text_file")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    with open(args.text_file, encoding="utf-8") as fh:
        text = fh.read()
    print(scan_report(scan_text(text, cfg)))


if __name__ == "__main__":
    main()
