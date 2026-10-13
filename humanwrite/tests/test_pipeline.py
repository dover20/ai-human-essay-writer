"""Pipeline: the full ingest -> humanizer -> constrain -> gate loop against
the stub LLM client, plus the gate and CLI pieces. No network.
"""
from __future__ import annotations

import copy
import io
import json
import os

import pytest

from humanwrite import gate, ingest, pipeline
from humanwrite.config import load_config
from humanwrite.llm import LLMClient, LLMError
from humanwrite.metrics import compute_stats
from conftest import StubClient, make_final_from_rough


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _inputs(tmp_path):
    rough = tmp_path / "rough.md"
    req = tmp_path / "req.md"
    notes = tmp_path / "notes.md"
    guides = tmp_path / "guides.md"
    return rough, req, notes, guides


# --------------------------------------------------------------------------
# gate math — each failing check produces exactly its own failure line
# --------------------------------------------------------------------------

def test_gate_clean_pass(rough_text, cfg):
    final = make_final_from_rough(rough_text)
    result = gate.evaluate(rough_text, final, cfg)
    assert result["passed"]
    assert result["failures"] == []
    for name in ("fk_grade", "mean_sentence_length", "ttr", "contraction_rate",
                 "tell_scan", "anchor_preservation", "length"):
        assert result["checks"][name]["ok"], name


def test_gate_length_failure(rough_text, cfg):
    short = "\n\n".join(rough_text.strip().split("\n\n")[:2])
    result = gate.evaluate(rough_text, short, cfg)
    failing = {n for n, c in result["checks"].items() if not c["ok"]}
    assert "length" in failing
    assert any("length" in f.lower() for f in result["failures"])


def test_gate_tell_failure(rough_text, cfg):
    final = make_final_from_rough(rough_text)
    result = gate.evaluate(rough_text, final + " This is a pivotal moment.", cfg)
    assert "tell_scan" in {n for n, c in result["checks"].items() if not c["ok"]}
    assert not result["passed"]


def test_gate_anchor_failure(rough_text, cfg):
    final = make_final_from_rough(rough_text)
    result = gate.evaluate(rough_text, final, cfg)
    assert result["checks"]["anchor_preservation"]["ok"]
    # total rewrite fails the anchor check
    other = "Cats sit quietly in the warm sun all day long and sleep. " \
            "They are calm and gentle pets for any quiet home to keep." * 20
    r2 = gate.evaluate(rough_text, other, cfg)
    assert not r2["checks"]["anchor_preservation"]["ok"]


def test_gate_contraction_low_threshold_absolute_mode():
    # rough rate < 1.0/100 words -> absolute mode (limit 1.0)
    small_r = "I do not like it much at all today."
    small_f = "I don't can't won't isn't haven't couldn't do that thing."
    cfg = load_config()
    r = gate.evaluate(small_r, small_f, cfg)
    c = r["checks"]["contraction_rate"]
    assert c["mode"] == "absolute"
    assert not c["ok"]


def test_gate_uses_config_bands(rough_text, cfg):
    # same draft with one sentence lengthened so the FK grade moves slightly
    final = make_final_from_rough(rough_text).replace(
        "I think that is backwards.",
        "I think that is a backwards way to do it for a long time now.")
    assert compute_stats(rough_text)["fk_grade"] != compute_stats(final)["fk_grade"]
    cfg2 = copy.deepcopy(cfg)
    cfg2["gate"]["fk_tolerance"] = 0.0
    result = gate.evaluate(rough_text, final, cfg2)
    assert not result["checks"]["fk_grade"]["ok"]


def test_gate_stats_shapes(rough_text, cfg):
    r = gate.evaluate(rough_text, make_final_from_rough(rough_text), cfg)
    for key in ("word_count", "sentence_count", "fk_grade", "mean_sentence_length",
                "ttr", "contraction_rate", "sentence_length_cov"):
        assert key in r["stats"]["rough"]
        assert key in r["stats"]["final"]
    assert r["anchors"]["total"] > 0
    assert "max_deviation" in r["proportions"]


# --------------------------------------------------------------------------
# candidate scoring
# --------------------------------------------------------------------------

def test_score_candidate_prefers_rough_like(rough_text, cfg):
    good = make_final_from_rough(rough_text)
    bad = "This is a pivotal and seamless landscape. We must leverage the " \
          "robust realm and delve into the intricate tapestry of failure. " \
          "It is worth noting that the kids, the students, and the family " \
          "were all changed, symbolizing a holistic shift in how we think."
    s_good = pipeline.score_candidate(rough_text, good, cfg)
    s_bad = pipeline.score_candidate(rough_text, bad, cfg)
    assert s_good["score"] < s_bad["score"]
    assert s_good["anchors"] > s_bad["anchors"]
    assert s_bad["tells"] > 0


# --------------------------------------------------------------------------
# pipeline with stub client — the full loop, files, and the repair loop
# --------------------------------------------------------------------------

def test_run_pipeline_passes_and_writes_files(tmp_path, rough_text,
                                              requirements_text, notes_text,
                                              guidelines_text, cfg):
    rough, req, notes, guides = _inputs(tmp_path)
    rough.write_text(rough_text, encoding="utf-8")
    req.write_text(requirements_text, encoding="utf-8")
    notes.write_text(notes_text, encoding="utf-8")
    guides.write_text(guidelines_text, encoding="utf-8")
    final_text = make_final_from_rough(rough_text)
    client = StubClient([final_text])  # every step returns the final text

    out = tmp_path / "out" / "final.md"
    result = pipeline.run_pipeline(
        rough_path=str(rough), requirements_path=str(req),
        notes_path=str(notes), guidelines_path=str(guides),
        client=client, cfg=cfg, out_path=str(out), log=lambda m: None,
    )
    assert result["passed"]
    assert out.exists()
    assert out.read_text(encoding="utf-8").strip() == final_text.strip()
    assert (out.parent / "report.md").exists()
    assert (out.parent / "author_profile.json").exists()
    assert (out.parent / "candidate.md").exists()
    assert (out.parent / "humanized.md").exists()
    assert (out.parent / "constrained.md").exists()
    assert (out.parent / "candidates" / "candidate_1.md").exists()
    assert (out.parent / "candidates" / "candidate_3.md").exists()
    # 3 candidates + humanizer + constrain + (gate passes on iteration 0)
    assert len(client.calls) == 5
    assert result["gate"]["passed"]
    report = (out.parent / "report.md").read_text(encoding="utf-8")
    assert "PASS" in report and "candidate" in report.lower()
    prof = json.loads((out.parent / "author_profile.json")
                      .read_text(encoding="utf-8"))
    assert prof["stats"]["word_count"] == compute_stats(rough_text)["word_count"]
    assert result["candidate_scores"][0]["index"] == 1


def test_run_pipeline_repair_loop_repairs_on_iteration_one(tmp_path,
                                                           rough_text, cfg):
    rough = tmp_path / "rough.md"
    rough.write_text(rough_text, encoding="utf-8")
    broken = make_final_from_rough(rough_text) + " This is a pivotal moment."
    fixed = make_final_from_rough(rough_text)
    # calls 1-3: candidates (broken); 4: humanizer (broken); 5: constrain
    # (broken) -> gate fails on iteration 0; call 6: repair returns fixed text
    # -> gate iteration 1 passes.
    client = StubClient([broken] * 5 + [fixed])
    out = tmp_path / "out" / "final.md"
    result = pipeline.run_pipeline(
        rough_path=str(rough), requirements_path=None, notes_path=None,
        guidelines_path=None, client=client, cfg=cfg, out_path=str(out),
        log=lambda m: None,
    )
    assert result["passed"]
    assert result["gate"]["passed"]
    assert len(client.calls) == 6
    assert out.read_text(encoding="utf-8").strip() == fixed.strip()


def test_run_pipeline_repair_loop_gives_up_after_max(tmp_path, rough_text, cfg):
    rough = tmp_path / "rough.md"
    rough.write_text(rough_text, encoding="utf-8")
    broken = make_final_from_rough(rough_text) + " This is a pivotal moment."
    client = StubClient([broken])  # repair never fixes it
    out = tmp_path / "out" / "final.md"
    result = pipeline.run_pipeline(
        rough_path=str(rough), requirements_path=None, notes_path=None,
        guidelines_path=None, client=client, cfg=cfg, out_path=str(out),
        log=lambda m: None,
    )
    assert not result["passed"]
    # 3 candidates + humanizer + constrain + 3 repairs (max_repair_iterations)
    assert len(client.calls) == 8
    assert result["gate"]["failures"]


def test_run_pipeline_rejects_oversized_notes(tmp_path, rough_text, cfg):
    cfg2 = copy.deepcopy(cfg)
    cfg2["gate"]["max_notes_words"] = 20000
    rough = tmp_path / "rough.md"
    rough.write_text(rough_text, encoding="utf-8")
    big = tmp_path / "notes.md"
    big.write_text("word " * 20001, encoding="utf-8")
    with pytest.raises(ValueError, match="limit"):
        pipeline.run_pipeline(
            rough_path=str(rough), requirements_path=None,
            notes_path=str(big), guidelines_path=None, client=StubClient([]),
            cfg=cfg2, out_path=str(tmp_path / "final.md"), log=lambda m: None)


def test_run_pipeline_no_notes_cap_by_default(tmp_path, rough_text, cfg):
    # default cap is 0 (unlimited): oversized notes pass the cap check and the
    # pipeline runs on to completion (gate result, files written)
    rough = tmp_path / "rough.md"
    rough.write_text(rough_text, encoding="utf-8")
    big = tmp_path / "notes.md"
    big.write_text("word " * 20001, encoding="utf-8")
    result = pipeline.run_pipeline(
        rough_path=str(rough), requirements_path=None,
        notes_path=str(big), guidelines_path=None, client=StubClient([]),
        cfg=cfg, out_path=str(tmp_path / "final.md"), log=lambda m: None)
    assert "passed" in result
    assert (tmp_path / "final.md").exists()


def test_run_pipeline_respects_notes_word_cap_config(tmp_path, rough_text, cfg):
    cfg2 = copy.deepcopy(cfg)
    cfg2["gate"]["max_notes_words"] = 100
    rough = tmp_path / "rough.md"
    rough.write_text(rough_text, encoding="utf-8")
    big = tmp_path / "notes.md"
    big.write_text("word " * 200, encoding="utf-8")
    with pytest.raises(ValueError, match="limit"):
        pipeline.run_pipeline(
            rough_path=str(rough), requirements_path=None,
            notes_path=str(big), guidelines_path=None, client=StubClient([]),
            cfg=cfg2, out_path=str(tmp_path / "final.md"), log=lambda m: None)


def test_strip_wrapper() -> None:
    assert pipeline.strip_wrapper("```markdown\ntext here\n```") == "text here"
    assert pipeline.strip_wrapper("Here is the draft: hi") == "hi"
    assert pipeline.strip_wrapper("  plain text  ") == "plain text"


# --------------------------------------------------------------------------
# gate loop directly
# --------------------------------------------------------------------------

def test_gate_loop_bounds(tmp_path, rough_text, cfg):
    broken = make_final_from_rough(rough_text) + " This is a pivotal moment."
    client = StubClient([broken])
    final, result, iters = pipeline.run_gate_loop(
        broken, rough_text, client, cfg, log=lambda m: None)
    assert not result["passed"]
    max_iter = cfg["gate"]["max_repair_iterations"]
    assert len(iters) == max_iter + 1  # 0..max, all failing


# --------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------

def test_ingest_cli_argument_parsing_error(tmp_path):
    # missing required --rough -> argparse exits 2
    with pytest.raises(SystemExit) as exc:
        ingest.main(["--out", str(tmp_path / "final.md")])
    assert exc.value.code == 2


def test_ingest_cli_passes_with_stub(monkeypatch, tmp_path, rough_text, cfg):
    rough = tmp_path / "rough.md"
    rough.write_text(rough_text, encoding="utf-8")
    final_text = make_final_from_rough(rough_text)
    monkeypatch.setattr(
        ingest, "client_from_config",
        lambda c, log=None: StubClient([final_text]))

    class FakeResult(dict):
        pass

    captured = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        return {"passed": True, "out": kwargs["out_path"], "report": "r"}

    monkeypatch.setattr(ingest, "run_pipeline", fake_run_pipeline)
    rc = ingest.main([
        "--rough", str(rough), "--out", str(tmp_path / "final.md"),
    ])
    assert rc == 0
    assert captured["rough_path"] == str(rough)


def test_ingest_cli_gate_failure_exit_code(monkeypatch, tmp_path, rough_text):
    rough = tmp_path / "rough.md"
    rough.write_text(rough_text, encoding="utf-8")
    monkeypatch.setattr(
        ingest, "client_from_config", lambda c, log=None: StubClient(["x"]))

    def fake_run_pipeline(**kwargs):
        return {"passed": False, "out": "x", "report": "r"}

    monkeypatch.setattr(ingest, "run_pipeline", fake_run_pipeline)
    assert ingest.main(["--rough", str(rough),
                        "--out", str(tmp_path / "final.md")]) == 3


def test_ingest_cli_llm_error_exit_code(monkeypatch, tmp_path, rough_text):
    rough = tmp_path / "rough.md"
    rough.write_text(rough_text, encoding="utf-8")

    def boom(**kwargs):
        raise LLMError("no endpoint")

    monkeypatch.setattr(ingest, "run_pipeline", boom)
    monkeypatch.setattr(
        ingest, "client_from_config", lambda c, log=None: StubClient([]))
    assert ingest.main(["--rough", str(rough),
                        "--out", str(tmp_path / "final.md")]) == 2


# --------------------------------------------------------------------------
# llm client unit pieces (no network: fake urlopen)
# --------------------------------------------------------------------------

def test_llm_client_chat_openai_shape(monkeypatch):
    def fake_urlopen(req, timeout=None):
        class R:
            status = 200
            def read(self):
                return json.dumps({"choices": [
                    {"message": {"content": "hello from the model"}}
                ]}).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        return R()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = LLMClient("http://x.example/v1", "m")
    assert client.chat([{"role": "user", "content": "hi"}]) == \
        "hello from the model"


def test_llm_client_chat_llama_shape(monkeypatch):
    def fake_urlopen(req, timeout=None):
        class R:
            status = 200
            def read(self):
                return json.dumps(
                    {"message": {"content": "llama says hi"}}).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        return R()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = LLMClient("http://x.example", "m", api_style="llama")
    assert client.chat([{"role": "user", "content": "hi"}]) == "llama says hi"


def test_llm_client_retries_429_with_long_backoff(monkeypatch):
    import time as time_mod
    import urllib.error

    sleeps: list[float] = []
    monkeypatch.setattr(time_mod, "sleep", lambda s: sleeps.append(s))
    attempts = {"n": 0}

    def fake_urlopen(req, timeout=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise urllib.error.HTTPError(
                req.full_url, 429, "Too Many Requests", {},
                io.BytesIO(b'{"error": {"code": "concurrency_limit"}}'))
        class R:
            status = 200
            def read(self):
                return json.dumps({"choices": [
                    {"message": {"content": "ok now"}}]}).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        return R()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = LLMClient("http://x.example/v1", "m", max_retries=3)
    assert client.chat([{"role": "user", "content": "hi"}]) == "ok now"
    assert attempts["n"] == 2
    assert sleeps and sleeps[0] == 30.0  # 429 waits longer than normal backoff


def test_llm_client_extra_body_merged_into_request(monkeypatch):
    import urllib.error

    seen_bodies: list[dict] = []

    def fake_urlopen(req, timeout=None):
        seen_bodies.append(json.loads(req.data.decode("utf-8")))

        class R:
            status = 200
            def read(self):
                return json.dumps({"choices": [
                    {"message": {"content": "merged"}}]}).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        return R()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = LLMClient(
        "http://x.example/v1", "m",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    assert client.chat([{"role": "user", "content": "hi"}]) == "merged"
    assert len(seen_bodies) == 1
    # extra_body keys are merged verbatim into the request body
    assert seen_bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    # ...without clobbering the standard keys
    assert seen_bodies[0]["model"] == "m"
    assert seen_bodies[0]["max_tokens"] == 4096  # client default, untouched

    # empty extra_body adds nothing
    seen_bodies.clear()
    client = LLMClient("http://x.example/v1", "m")
    client.chat([{"role": "user", "content": "hi"}])
    assert "chat_template_kwargs" not in seen_bodies[0]
