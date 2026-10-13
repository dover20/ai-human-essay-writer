"""Live e2e: the real endpoint, one full pipeline run on the examples.

Skipped unless pytest is invoked with ``--live``. The endpoint is slow and
shares one concurrency slot, so the run is deliberately small: one
candidate, short timeout, generous cap.
"""
from __future__ import annotations

import pytest

from conftest import EXAMPLES
from humanwrite.config import load_config
from humanwrite.llm import LLMClient, client_from_config
from humanwrite.pipeline import run_pipeline


def _cfg_live():
    cfg = load_config()
    cfg["sampling"]["n_candidates"] = 1
    cfg["gate"]["max_repair_iterations"] = 1
    return cfg


@pytest.mark.live
def test_endpoint_reachable():
    cfg = _cfg_live()
    client = client_from_config(cfg, log=lambda m: None)
    assert client.ping(), (
        f"endpoint {cfg['endpoint']} not reachable; run --live only when it is up")


@pytest.mark.live
def test_live_full_pipeline(tmp_path, rough_text, requirements_text,
                            notes_text, guidelines_text):
    cfg = _cfg_live()
    client = LLMClient(endpoint=cfg["endpoint"], model=cfg["model"],
                       api_style=cfg.get("api_style", "auto"),
                       timeout=float(cfg.get("timeout_seconds", 600)),
                       max_retries=1, log=lambda m: None)
    result = run_pipeline(
        rough_path=f"{EXAMPLES}/rough_sample.md",
        requirements_path=f"{EXAMPLES}/requirements_sample.md",
        notes_path=f"{EXAMPLES}/notes_sample.md",
        guidelines_path=f"{EXAMPLES}/guidelines_sample.md",
        client=client, cfg=cfg,
        out_path=str(tmp_path / "out" / "final.md"),
        log=lambda m: None,
    )
    assert (tmp_path / "out" / "final.md").exists()
    assert (tmp_path / "out" / "report.md").exists()
    # a live run may legitimately fail the gate (model output varies); the
    # requirement is that the pipeline completes and the report is written
    assert result["gate"] is not None
    for name, check in result["gate"]["checks"].items():
        assert check["ok"] is not None, name
