"""Tell scan: one positive and one negative fixture for every pattern id.

The positive fixture must produce at least one violation with the expected
pattern id; the negative fixture must produce none at all (it is plain,
ordinary prose that a human would write).
"""
from __future__ import annotations

import copy

import pytest

from humanwrite.config import load_config
from humanwrite.tell_scan import scan_report, scan_text


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def hits(text: str, cfg) -> list[dict]:
    return scan_text(text, cfg)


def assert_pattern_hit(text: str, cfg, pattern_id: str) -> None:
    found = {v["pattern"] for v in hits(text, cfg)}
    assert pattern_id in found, (
        f"expected {pattern_id} in {found}; violations: "
        f"{[(v['pattern'], v['text']) for v in hits(text, cfg)]}")


def assert_clean(text: str, cfg) -> None:
    violations = hits(text, cfg)
    assert violations == [], [(v["pattern"], v["text"]) for v in violations]


# --------------------------------------------------------------------------
# one (positive, negative) fixture per pattern
# --------------------------------------------------------------------------

CASES = {
    "T01": (
        "This is not just a test but a real exam for everyone.",
        "The test was hard. I passed it anyway.",
    ),
    "T02": (
        "Here's the thing: it actually matters a lot.",
        "Here is the thing I did yesterday morning.",
    ),
    "T03": (
        "Let that sink in.",
        "Let the week pass. It was not fun at all.",
    ),
    "T04": (
        "It was bold, risky, and smart.",
        "We had a test and a quiz that week.",
    ),
    "T05": (
        # 7 identical openers is a genuine tic (default threshold is 7);
        # 3-5 is normal student prose and must NOT flag.
        "That is hard. That is real. That is true. That is why. "
        "That is it. That is now. That is all.",
        "That was bad. It was worse. He was the worst of them.",
    ),
    "T06": (
        "The plan — bold, risky — worked out fine.",
        "The plan worked out fine in the end.",
    ),
    "T07": (
        "It was essentially a great day for everyone.",
        "It was a great day for everyone.",
    ),
    "T08": (
        "This is a pivotal moment for the whole town.",
        "This is a key moment for the whole town.",
    ),
    "T09": (
        "The crowd cheered, symbolizing their joy at the win.",
        "The crowd cheered loudly at the win.",
    ),
    "T10": (
        "The river serves as a boundary between the two towns.",
        "The river is a boundary between the two towns.",
    ),
    "T11": (
        "Experts agree that the treatment works well.",
        "My teacher agrees that the treatment works well.",
    ),
    "T12": (
        "In conclusion, we should act before it is too late.",
        "The conclusion of the meeting was clear to all.",
    ),
    "T13": (
        "## Why I Believe This\n"
        "Why I believe this is simple and easy to explain.",
        "## why I believe this\n"
        "The evidence for my position is strong and clear.",
    ),
    "T14": (
        "**Key idea**: the point is simple.",
        "The **key** idea is simple and easy to see.",
    ),
    "T15": (
        "## The Importance of Failure\n"
        "Body text goes here.",
        "## importance of the failure\n"
        "Body text goes here.",
    ),
    "T16": (
        "He said “hello” to the whole room.",
        'He said "hello" to the whole room.',
    ),
    "T17": (
        "Great question! Here is the answer you want.",
        "That is a good question and I will answer it.",
    ),
    "T18": (
        "It should be noted that the results were mixed.",
        "The results were mixed, as expected.",
    ),
    "T19": (
        "For context, the book is quite long and dense.",
        "The book is quite long and dense to read.",
    ),
}


@pytest.mark.parametrize("pattern_id", sorted(CASES))
def test_positive_hits(pattern_id: str, cfg) -> None:
    positive, _ = CASES[pattern_id]
    assert_pattern_hit(positive, cfg, pattern_id)


@pytest.mark.parametrize("pattern_id", sorted(CASES))
def test_negative_clean(pattern_id: str, cfg) -> None:
    _, negative = CASES[pattern_id]
    assert_clean(negative, cfg)


# --------------------------------------------------------------------------
# config-driven behavior
# --------------------------------------------------------------------------

def test_extra_words_add_to_vocabulary(cfg) -> None:
    cfg2 = copy.deepcopy(cfg)
    cfg2["tell_scan"]["extra_words"] = ["fluffy"]
    assert_pattern_hit("The cat is very fluffy today.", cfg2, "T08")
    assert_clean("The cat is very fluffy today.", cfg)


def test_disabled_patterns_are_skipped(cfg) -> None:
    cfg2 = copy.deepcopy(cfg)
    cfg2["tell_scan"]["disabled_patterns"] = ["T08"]
    assert_clean("This is a pivotal moment for the whole town.", cfg2)


def test_vocabulary_inflections_expand(cfg) -> None:
    # "delves" is an inflection of "delve" from the default vocabulary
    assert_pattern_hit("The report delves into the details.", cfg, "T08")


def test_scan_report_format(cfg) -> None:
    assert scan_report([]).strip() == "tell scan: 0 violations"
    report = scan_report(hits("This is a pivotal moment.", cfg))
    assert "violations" in report
    assert "T08" in report


def test_violation_shape(cfg) -> None:
    v = hits("This is a pivotal moment.", cfg)[0]
    for key in ("line", "col", "pattern", "name", "text", "note"):
        assert key in v
    assert v["line"] >= 1
