"""Metrics: hand-computed fixtures, sentence-splitter edge cases, and an
optional textstat cross-check (skipped when textstat or its data is
unavailable).
"""
from __future__ import annotations

import math

import pytest

from humanwrite.metrics import (
    count_syllables,
    contraction_rate,
    flesch_kincaid_grade,
    flesch_reading_ease,
    mean_sentence_length,
    sentence_length_cov,
    split_sentences,
    total_syllables,
    type_token_ratio,
    word_tokens,
)

# --------------------------------------------------------------------------
# syllables — chosen to match the vowel-group heuristic's actual behavior
# --------------------------------------------------------------------------

SYLLABLE_FIXTURES = {
    "the": 1, "cat": 1, "sat": 1, "red": 1, "blue": 1, "jump": 1,
    "time": 1, "care": 1, "fine": 1, "know": 1, "i": 1,
    "easy": 2, "water": 2, "table": 2, "happy": 2, "again": 2,
    "window": 2, "people": 2, "running": 2,
    "history": 3, "computer": 3, "banana": 3, "elephant": 3,
    "beautiful": 3,
    "school": 1,
}


@pytest.mark.parametrize("word,expected", sorted(SYLLABLE_FIXTURES.items()))
def test_syllables(word: str, expected: int) -> None:
    assert count_syllables(word) == expected


def test_total_syllables() -> None:
    assert total_syllables(["the", "cat", "water", "history"]) == 1 + 1 + 2 + 3
    assert total_syllables([]) == 0
    assert total_syllables(["123", "cat"]) == 1  # digits carry no syllables


# --------------------------------------------------------------------------
# sentence splitting
# --------------------------------------------------------------------------

def test_split_basic() -> None:
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_split_no_final_period() -> None:
    assert split_sentences("End of doc no period") == ["End of doc no period"]


def test_split_multiple_terminal_punctuation() -> None:
    assert split_sentences("What?! She left.") == ["What?", "She left."]


@pytest.mark.parametrize("text,expected", [
    ("Mr. Smith said hi. He left.",
     ["Mr. Smith said hi.", "He left."]),
    ("It cost 3.5 dollars. We paid.",
     ["It cost 3.5 dollars.", "We paid."]),
    ("U.S. policy matters.", ["U.S. policy matters."]),
    ("e.g. apples and oranges. Done.",
     ["e.g. apples and oranges.", "Done."]),
    ("I.e. we stay. Yes we do.", ["I.e. we stay.", "Yes we do."]),
    ("Approx. 5 miles. Done.", ["Approx. 5 miles.", "Done."]),
    ("See p. 12 for details. Then more.",
     ["See p. 12 for details.", "Then more."]),
    ("I said 'go' and went. Then I stopped.",
     ["I said 'go' and went.", "Then I stopped."]),
])
def test_split_abbreviations_and_numbers(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


def test_split_initial_then_lower_case_is_boundary() -> None:
    # "A. person" — lone capital before a lowercase word is a real boundary
    assert split_sentences("A. person walked. B. ran.") == [
        "A. person walked.", "B. ran."]


def test_split_parenthetical_quoted() -> None:
    assert split_sentences("I said 'go' and went.") == ["I said 'go' and went."]


# --------------------------------------------------------------------------
# FK grade / ease — hand-computed
# --------------------------------------------------------------------------

def test_fk_grade_hand_computed() -> None:
    # 2 sentences, 10 words; syllables: the(1) cat(1) sat(1) on(1) the(1)
    # mat(1) the(1) cat(1) sat(1) down(1) = 10
    text = "The cat sat on the mat. The cat sat down."
    words = word_tokens(text)
    assert len(words) == 10
    assert len(split_sentences(text)) == 2
    asl = 5.0
    assert total_syllables(words) == 10
    asp = 1.0
    assert flesch_kincaid_grade(text) == 0.39 * asl + 11.8 * asp - 15.59
    assert flesch_reading_ease(text) == 206.835 - 1.015 * asl - 84.6 * asp


def test_fk_grade_empty() -> None:
    assert flesch_kincaid_grade("") == 0.0
    assert flesch_reading_ease("") == 0.0
    assert mean_sentence_length("") == 0.0


def test_mean_sentence_length_hand_computed() -> None:
    text = "One two three. Four five."
    assert mean_sentence_length(text) == 2.5


def test_sentence_length_cov() -> None:
    # lengths 3 and 9: mean 6, var = ((3-6)^2 + (9-6)^2)/2 = 9, sqrt=3, cov=0.5
    text = "a b c. d e f g h i j k l."
    assert len(split_sentences(text)) == 2
    assert math.isclose(sentence_length_cov(text), 0.5)
    assert sentence_length_cov("only one") == 0.0


def test_type_token_ratio_hand_computed() -> None:
    # 4 tokens, 3 types
    assert math.isclose(type_token_ratio("the cat the dog"), 0.75)
    assert type_token_ratio("") == 0.0


# --------------------------------------------------------------------------
# contractions
# --------------------------------------------------------------------------

def test_contraction_rate_hand_computed() -> None:
    # 4 words, 1 contraction ("don't") -> 25.0 /100 words
    assert contraction_rate("I don't think so.") == 25.0
    assert contraction_rate("I do not think so.") == 0.0
    assert contraction_rate("") == 0.0


# --------------------------------------------------------------------------
# optional cross-check against textstat (skip when unavailable)
# --------------------------------------------------------------------------

def _textstat_text():
    return ("The cat sat on the mat all day. It was a quiet and calm day "
            "for everyone in the house, and the cat felt fine about it.")


@pytest.fixture(scope="module")
def textstat_stats():
    try:
        import textstat
    except ImportError:
        pytest.skip("textstat not installed")
    text = _textstat_text()
    try:
        fk = textstat.flesch_kincaid_grade(text)
        ease = textstat.flesch_reading_ease(text)
    except Exception:  # needs cmudict data which may be un-downloadable
        pytest.skip("textstat could not compute (missing word data)")
    return text, fk, ease


def test_textstat_cross_check(textstat_stats) -> None:
    text, fk, ease = textstat_stats
    # our syllable heuristic differs from the cmudict lookup, so allow
    # generous slack — this only checks we're in the same ball-park
    assert abs(flesch_kincaid_grade(text) - fk) < 4.0
    assert abs(flesch_reading_ease(text) - ease) < 25.0
