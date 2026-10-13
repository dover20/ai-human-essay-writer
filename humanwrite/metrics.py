"""Readability and structure statistics, computed from scratch (stdlib only).

Everything the gate needs lives here:

* sentence splitting with abbreviation awareness
* syllable counting (vowel-group heuristic with silent-e stripping)
* Flesch-Kincaid grade and Flesch Reading Ease
* mean sentence length (words) and its coefficient of variation
* type-token ratio (TTR)
* contraction rate per 100 words (explicit contraction list)
* passive-voice rate (heuristic, reported only)
* a structure map (sections / paragraphs and their length proportions)

Numbers are intentionally conservative: for the gate what matters is that the
same code is applied to both the rough draft and the final draft, so a final
draft that came from the same author's hand lands in-band.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------
# tokenizing
# --------------------------------------------------------------------------

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
VOWELS = "aeiouy"

#: words/abbreviations whose trailing period never ends a sentence
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "capt", "col",
    "sgt", "rev", "pres", "sen", "rep", "gov", "gen", "sgt", "ft", "approx",
    "approx.", "dept", "inc", "corp", "ltd", "co", "vs", "cf", "al", "misc",
    "etc", "eg", "ie", "no", "vol", "fig", "sec", "para", "p", "pp", "ch",
    "ed", "eds", "trans", "univ", "inst", "assn", "bros", "min", "max",
    "dept.", "u.s", "u.k", "e.g", "i.e", "st.", "mr.",
}

#: common be-forms used by the passive heuristic
BE_FORMS = {"am", "is", "are", "was", "were", "been", "being", "be"}
#: common past participles the "-ed" heuristic misses
PARTICIPLES = {
    "done", "made", "known", "given", "seen", "taken", "written", "found",
    "built", "sent", "kept", "left", "put", "set", "got", "told", "shown",
    "chosen", "driven", "frozen", "broken", "spoken", "worn", "born",
    "taken", "run", "understood", "meant", "brought", "thought", "left",
    "paid", "said", "taught", "caught", "sought", "sold", "told", "woken",
}


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def word_tokens(text: str) -> list[str]:
    """Words lowercased (used for TTR and contraction counting)."""
    return [w.lower() for w in tokenize_words(text)]


# --------------------------------------------------------------------------
# syllables
# --------------------------------------------------------------------------

def count_syllables(word: str) -> int:
    """Vowel-group heuristic with silent-e stripping.

    Rules (in order):
      1. keep only a-z
      2. single letter -> 1
      3. strip a trailing "es" or "ed" only when preceded by s/x/z (es) or a
         consonant (ed), to avoid counting inflectional endings
      4. strip one trailing "e" unless the word ends in "le" (people, table)
      5. count vowel groups (y counts as a vowel only when not the first letter)
      6. minimum 1
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) == 1:
        return 1
    # drop inflectional endings that would add a fake vowel group
    if w.endswith("es") and len(w) > 3 and w[-3] in "sxz":
        w = w[:-2]
    elif w.endswith("ed") and len(w) > 3 and w[-3] not in VOWELS and w[-3] not in ("e",):
        w = w[:-2]
    if w.endswith("e") and len(w) > 2 and not w.endswith("le"):
        w = w[:-1]
    if not w:
        return 1
    count = 0
    prev_vowel = False
    for i, ch in enumerate(w):
        is_vowel = ch in "aeiou" or (ch == "y" and i > 0)
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(1, count)


def total_syllables(words: list[str]) -> int:
    return sum(count_syllables(w) for w in words if any(c.isalpha() for c in w))


# --------------------------------------------------------------------------
# sentences
# --------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Handles: multiple terminal punctuation ("?!" / "?!"), abbreviations
    (Mr., e.g., i.e., approx., ...), decimals (3.5), acronyms (U.S., A.),
    and stray periods. A boundary is a run of .!? followed by whitespace or
    end-of-string, unless the character before it makes the dot belong to an
    abbreviation, number, or single-letter initial.
    """
    text = text.replace("\n", " ")
    sentences: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in ".!?":
            # absorb runs of punctuation: "?!." etc.
            j = i
            while j < n and text[j] in ".!?":
                j += 1
            # a boundary requires whitespace (or end) after the run
            if j >= n or text[j].isspace():
                if not _abbreviation_dot(text, i):
                    piece = text[start:i + 1].strip()
                    if piece:
                        sentences.append(piece)
                    start = j
                    i = j
                    continue
        i += 1
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    # drop pure-punctuation or empty fragments
    return [s for s in sentences if re.search(r"[A-Za-z0-9]", s)]


def _abbreviation_dot(text: str, dot_index: int) -> bool:
    """True if the character at dot_index (a '.') is part of an
    abbreviation/number/initial rather than a sentence boundary."""
    # walk back over the punctuation run to find the dot in question
    k = dot_index
    while k > 0 and text[k - 1] in ".!?" and not text[k - 1].isspace():
        k -= 1
    # scan backwards for the token (letters + inner dots, e.g. "e.g")
    token_end = k
    token_start = token_end
    while token_start > 0 and not text[token_start - 1].isspace():
        token_start -= 1
    raw = text[token_start:token_end]
    token = raw.lower()
    if not token:
        return True
    compact = re.sub(r"[.\-]", "", token)
    if token in ABBREVIATIONS or compact in ABBREVIATIONS:
        return True
    # decimals: digit . digit
    after = text[token_end:token_end + 1]
    before = text[token_start - 1] if token_start > 0 else ""
    if token[-1].isdigit() and after.isdigit():
        return True
    if before.isdigit() and after.isdigit():
        return True
    # initials and acronyms: "J. Smith", "U.S. policy", "A.B. something"
    if raw.isupper():
        # inside/after an acronym: the token itself holds dots ("U.S"),
        # or the next char is another dot ("U." followed by "S.")
        if "." in raw and re.fullmatch(r"[A-Z.]+", raw):
            return True
        if after == ".":
            return True
        # this token continues an acronym: char before it is a dot ("...S. policy")
        if before == ".":
            return True
        # personal initial: next word starts with a capital ("J. Smith")
        rest = text[token_end + 1:]
        m = re.match(r"\s*([A-Za-z])", rest)
        if m and m.group(1).isupper():
            return True
    return False


# --------------------------------------------------------------------------
# the metrics
# --------------------------------------------------------------------------

CONTRACTIONS = {
    "i'm", "i've", "i'll", "i'd", "you're", "you've", "you'll", "you'd",
    "he's", "he'd", "he'll", "she's", "she'd", "she'll", "it's", "it'll",
    "we're", "we've", "we'll", "we'd", "they're", "they've", "they'll",
    "they'd", "don't", "doesn't", "didn't", "can't", "won't", "wouldn't",
    "shouldn't", "couldn't", "isn't", "aren't", "wasn't", "were't",
    "wasn't", "weren't", "hasn't", "haven't", "hadn't", "let's", "that's",
    "there's", "what's", "who's", "where's", "here's", "how's",
    "ain't", "im", "dont", "cant", "wont",
}
CONTRACTIONS.discard("were't")


def flesch_kincaid_grade(text: str) -> float:
    words = word_tokens(text)
    sentences = split_sentences(text)
    if not words or not sentences:
        return 0.0
    asl = len(words) / len(sentences)
    asp = total_syllables(words) / len(words)
    return 0.39 * asl + 11.8 * asp - 15.59


def flesch_reading_ease(text: str) -> float:
    words = word_tokens(text)
    sentences = split_sentences(text)
    if not words or not sentences:
        return 0.0
    asl = len(words) / len(sentences)
    asp = total_syllables(words) / len(words)
    return 206.835 - 1.015 * asl - 84.6 * asp


def mean_sentence_length(text: str) -> float:
    words = word_tokens(text)
    sentences = split_sentences(text)
    if not sentences:
        return 0.0
    return len(words) / len(sentences)


def sentence_length_cov(text: str) -> float:
    """Coefficient of variation of sentence lengths (burstiness)."""
    words = word_tokens(text)
    sentences = split_sentences(text)
    lengths = [len(tokenize_words(s)) for s in sentences]
    lengths = [l for l in lengths if l > 0]
    if len(lengths) < 2:
        return 0.0
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 0.0
    var = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    return math.sqrt(var) / mean


def type_token_ratio(text: str) -> float:
    words = word_tokens(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def contraction_rate(text: str) -> float:
    """Contractions per 100 words."""
    words = word_tokens(text)
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in CONTRACTIONS)
    return 100.0 * hits / len(words)


def passive_rate(text: str) -> float:
    """Fraction of sentences with a be-form + participle pairing (rough)."""
    sentences = split_sentences(text)
    if not sentences:
        return 0.0
    passive = 0
    for s in sentences:
        words = word_tokens(s)
        for idx in range(len(words) - 1):
            if words[idx] in BE_FORMS:
                nxt = words[idx + 1]
                if nxt.endswith("ed") or nxt in PARTICIPLES or nxt in BE_FORMS:
                    passive += 1
                    break
    return passive / len(sentences)


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$", re.MULTILINE)


@dataclass
class StructureMap:
    headings: list[str] = field(default_factory=list)
    sections: list[dict[str, Any]] = field(default_factory=list)
    n_paragraphs: int = 0
    paragraph_words: list[int] = field(default_factory=list)
    paragraph_proportions: list[float] = field(default_factory=list)
    section_proportions: list[float] = field(default_factory=list)
    granularity: str = "paragraph"

    def to_dict(self) -> dict[str, Any]:
        return {
            "headings": self.headings,
            "sections": self.sections,
            "n_sections": len(self.sections),
            "section_proportions": self.section_proportions,
            "n_paragraphs": self.n_paragraphs,
            "paragraph_words": self.paragraph_words,
            "paragraph_proportions": self.paragraph_proportions,
            "granularity": self.granularity,
        }


def structure_map(text: str) -> StructureMap:
    """Map the document's structure: sections (by markdown headings when
    present, otherwise paragraphs) and their length proportions.

    ``granularity`` says which level carries the real unevenness signal:
    "section" when the document has 2+ headings, else "paragraph".
    """
    sm = StructureMap()
    headings = [(m.start(), m.group(1), m.group(2)) for m in _HEADING_RE.finditer(text)]
    sm.headings = [h[2] for h in headings]

    if len(headings) >= 2:
        sm.granularity = "section"
        bounds = [0] + [h[0] for h in headings] + [len(text)]
        for i in range(len(headings)):
            chunk = text[headings[i][0]: headings[i + 1][0] if i + 1 < len(headings) else len(text)]
            title = headings[i][2]
            words = len(tokenize_words(chunk))
            paras = [p for p in chunk.split("\n\n") if tokenize_words(p)]
            sm.sections.append({"title": title, "words": words, "paragraphs": len(paras)})
        total = sum(s["words"] for s in sm.sections)
        sm.section_proportions = [s["words"] / total if total else 0.0 for s in sm.sections]
        # paragraph stats over the whole doc
        paras = [p for p in text.split("\n\n") if tokenize_words(p)]
        sm.n_paragraphs = len(paras)
        sm.paragraph_words = [len(tokenize_words(p)) for p in paras]
    else:
        sm.granularity = "paragraph"
        paras = [p for p in text.split("\n\n") if tokenize_words(p)]
        sm.n_paragraphs = len(paras)
        sm.paragraph_words = [len(tokenize_words(p)) for p in paras]
        total = sum(sm.paragraph_words)
        sm.paragraph_proportions = [w / total if total else 0.0 for w in sm.paragraph_words]
        # whole document is one section
        sm.sections = [{"title": "(whole document)", "words": sum(sm.paragraph_words),
                        "paragraphs": len(paras)}]
        sm.section_proportions = [1.0] if sm.paragraph_words else []
    return sm


# --------------------------------------------------------------------------
# full profile computation (deterministic part)
# --------------------------------------------------------------------------

def compute_stats(text: str) -> dict[str, Any]:
    """All deterministic statistics for one document."""
    words = word_tokens(text)
    sentences = split_sentences(text)
    sm = structure_map(text)
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "syllable_count": total_syllables(words),
        "fk_grade": round(flesch_kincaid_grade(text), 2),
        "flesch_ease": round(flesch_reading_ease(text), 2),
        "mean_sentence_length": round(mean_sentence_length(text), 2),
        "sentence_length_cov": round(sentence_length_cov(text), 3),
        "ttr": round(type_token_ratio(text), 4),
        "contraction_rate": round(contraction_rate(text), 3),
        "passive_rate": round(passive_rate(text), 3),
        "structure": sm.to_dict(),
        "sentences": sentences,
    }
