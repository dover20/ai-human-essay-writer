"""Shared test fixtures for the humanwrite test suite.

Unit tests never touch the network: the pipeline test uses a deterministic
StubClient. Live e2e tests are opt-in with ``pytest --live``.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from humanwrite.llm import LLMClient  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False,
                     help="run live e2e tests against the configured LLM endpoint")


def pytest_configure(config):
    config.addinivalue_line("markers",
                            "live: run only with --live (hits the real endpoint)")


SKIP_LIVE = pytest.mark.skip(reason="enable with --live")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    for item in items:
        if "live" in item.keywords:
            item.add_marker(SKIP_LIVE)


HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
EXAMPLES = os.path.join(PKG, "examples")


@pytest.fixture(scope="session")
def rough_text() -> str:
    with open(os.path.join(EXAMPLES, "rough_sample.md"), encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="session")
def requirements_text() -> str:
    with open(os.path.join(EXAMPLES, "requirements_sample.md"),
              encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="session")
def notes_text() -> str:
    with open(os.path.join(EXAMPLES, "notes_sample.md"), encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="session")
def guidelines_text() -> str:
    with open(os.path.join(EXAMPLES, "guidelines_sample.md"),
              encoding="utf-8") as fh:
        return fh.read()


def make_final_from_rough(rough: str) -> str:
    """The prepared 'final draft' used by pipeline tests.

    It is the rough draft with the four repeated 'That' sentence openings
    de-duplicated and 'dont' fixed to "don't" — enough to clear the tell
    scan and anchor gate while staying in-band on every metric.
    """
    t = rough.replace("dont", "don't")
    t = t.replace("That was the moment I stopped",
                  "It was the moment I stopped")
    t = t.replace("That was the worst week of my life",
                  "It was the worst week of my life")
    return t


class StubClient:
    """Deterministic stand-in for LLMClient with the same chat() signature.

    Each call pops the next prepared response; when the list is exhausted it
    keeps returning the last one. Calls are recorded in ``.calls``.
    """

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self._i = 0

    def chat(self, messages, temperature=None, top_p=None,
             presence_penalty=None, max_tokens=None) -> str:
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
        })
        if self._i < len(self.responses):
            out = self.responses[self._i]
        else:
            out = self.responses[-1] if self.responses else ""
        self._i += 1
        return out

    def ping(self) -> bool:
        return True


class FailClient(StubClient):
    """Always fails — for the retry/LLMError path."""

    def chat(self, messages, temperature=None, top_p=None,
             presence_penalty=None, max_tokens=None) -> str:
        raise RuntimeError("endpoint down")
