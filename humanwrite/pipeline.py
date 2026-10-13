"""The four-step pipeline:

  1. profile the rough draft, generate n candidates, pick the best
  2. humanizer pass (remove AI tells via the local model)
  3. constrain pass (pull back toward the rough draft via the local model)
  4. quantitative gate; on failure, one targeted repair pass per iteration,
     up to max_repair_iterations

All prose generation goes through the injected client — this module never
writes prose itself.

Entry point: humanwrite.ingest (CLI) or run_pipeline() (library).
"""

from __future__ import annotations

import json
import os
from typing import Any

from humanwrite import gate, humanizer, metrics, profile, tell_scan
from humanwrite.config import get
from humanwrite.constrain import build_messages as constrain_messages


def _read(path: str | None) -> str:
    if not path:
        return ""
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip()


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


# --------------------------------------------------------------------------
# step 1: candidates
# --------------------------------------------------------------------------

def build_generation_messages(
    rough: str,
    requirements: str,
    notes: str,
    guidelines: str,
    prof: dict[str, Any],
    cfg: dict[str, Any],
) -> list[dict[str, str]]:
    g = cfg.get("gate", {})
    exemplars = "\n\n".join(
        f"[exemplar {i + 1} — {ex['reason']}]\n{ex['text']}"
        for i, ex in enumerate(prof["voice_exemplars"])
    )
    struct = prof["stats"]["structure"]
    if struct["granularity"] == "section":
        structure_map = "\n".join(
            f"- {s['title']}: {s['words']} words, {prop:.0%} of the draft, {s['paragraphs']} paragraphs"
            for s, prop in zip(struct["sections"], struct["section_proportions"])
        )
    else:
        structure_map = (
            f"{struct['n_paragraphs']} paragraphs, proportions: "
            + ", ".join(f"{p:.2f}" for p in struct["paragraph_proportions"])
        )
    system = (
        "You are the student who wrote the attached rough draft. Your voice, "
        "vocabulary level, and habits are exactly those in it. You are writing "
        "the final draft as your own polished version of your own rough work — "
        "you revise your own draft, you do not write a new essay. You keep your "
        "own opinions, your own examples, and the same section structure, in "
        "the same order. Your prose stays uneven the way your rough draft was: "
        "the section that was over-developed stays over-developed, and the thin "
        "one stays thin. Write only the final draft, no commentary, no preamble, "
        "no headings you were not given."
    )
    user = (
        "Below are your rough draft, the assignment, your notes, and the "
        "guidelines.\n\n"
        "--- ASSIGNMENT REQUIREMENTS ---\n"
        f"{requirements or '(none provided)'}\n\n"
        "--- YOUR NOTES (paraphrase these; use short quotes with page numbers; "
        "never copy the source wording beyond short quoted phrases) ---\n"
        f"{notes or '(none provided)'}\n\n"
        "--- WRITING GUIDELINES ---\n"
        f"{guidelines or '(none provided)'}\n\n"
        "--- VOICE EXEMPLARS (paragraphs from your rough draft — sound like this) ---\n"
        f"{exemplars}\n\n"
        "--- YOUR ROUGH DRAFT (the whole thing) ---\n"
        f"{rough}\n\n"
        "--- STRUCTURE MAP (keep this shape) ---\n"
        f"{structure_map}\n\n"
        "Requirements for the final draft:\n"
        "1. Keep every section from the rough draft, in the same order, with "
        "roughly the same length proportions.\n"
        "2. Keep the facts, quotes, and page numbers from your notes. "
        "Paraphrase the notes; quote sparingly.\n"
        "3. Keep the rough draft's voice: same vocabulary level, same rhythm, "
        "including its short abrupt sentences.\n"
        "4. Polish it the way a student would: fix the worst grammar, tighten "
        "the weakest parts, but do not rewrite your own sentences wholesale. A "
        "good third of the sentences should be exactly or nearly exactly yours.\n"
        "5. Do not use words you have never used before: no fancy vocabulary, "
        "no big-word filler.\n"
        "6. No 'In conclusion', no 'It is worth noting', no 'In today's world'.\n"
        f"7. Keep the length between {g.get('length_ratio_low', 0.9)}x and "
        f"{g.get('length_ratio_high', 1.5)}x the rough draft's word count.\n\n"
        "Write the final draft now."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def score_candidate(rough: str, candidate: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Lower is better. Combines tell hits, metric deltas vs the rough draft,
    and anchor preservation (the candidate should already look like the
    student's own revision)."""
    r = metrics.compute_stats(rough)
    c = metrics.compute_stats(candidate)
    tells = len(tell_scan.scan_text(candidate, cfg))
    anchors = gate.anchor_preservation(rough, candidate)["ratio"]
    msl_ratio = c["mean_sentence_length"] / r["mean_sentence_length"] if r["mean_sentence_length"] else 1.0
    ttr_ratio = c["ttr"] / r["ttr"] if r["ttr"] else 1.0
    len_ratio = c["word_count"] / r["word_count"] if r["word_count"] else 1.0
    g = cfg.get("gate", {})

    def band_penalty(value: float, lo: float, hi: float) -> float:
        if value < lo:
            return lo - value
        if value > hi:
            return value - hi
        return 0.0

    score = (
        tells * 2.0
        + abs(c["fk_grade"] - r["fk_grade"]) * 0.5
        + band_penalty(msl_ratio, g.get("msl_ratio_low", 0.80), g.get("msl_ratio_high", 1.25)) * 1.0
        + band_penalty(ttr_ratio, g.get("ttr_ratio_low", 0.85), g.get("ttr_ratio_high", 1.15)) * 1.0
        + abs(c["contraction_rate"] - r["contraction_rate"]) * 0.3
        + (1.0 - anchors) * 3.0
        + band_penalty(len_ratio, g.get("length_ratio_low", 0.9), g.get("length_ratio_high", 1.5)) * 2.0
    )
    return {
        "score": round(score, 3),
        "tells": tells,
        "anchors": anchors,
        "fk_delta": round(c["fk_grade"] - r["fk_grade"], 2),
        "msl_ratio": round(msl_ratio, 3),
        "ttr_ratio": round(ttr_ratio, 3),
        "length_ratio": round(len_ratio, 3),
        "contraction_delta": round(c["contraction_rate"] - r["contraction_rate"], 3),
    }


def generate_candidates(rough: str, requirements: str, notes: str,
                        guidelines: str, prof: dict[str, Any], client,
                        cfg: dict[str, Any], out_dir: str, log=print) -> tuple[str, list[dict[str, Any]]]:
    n = int(get(cfg, "sampling.n_candidates", 3))
    s = cfg.get("sampling", {})
    messages = build_generation_messages(rough, requirements, notes, guidelines, prof, cfg)
    candidates: list[str] = []
    scores: list[dict[str, Any]] = []
    for i in range(n):
        cand = client.chat(
            messages,
            temperature=s.get("temperature"),
            top_p=s.get("top_p"),
            presence_penalty=s.get("presence_penalty"),
            max_tokens=s.get("max_tokens"),
        )
        cand = strip_wrapper(cand)
        candidates.append(cand)
        sc = score_candidate(rough, cand, cfg)
        sc["index"] = i + 1
        scores.append(sc)
        log(f"candidate {i + 1}: score {sc['score']} (tells={sc['tells']}, "
            f"anchors={sc['anchors']:.0%}, FK delta={sc['fk_delta']:+.2f}, "
            f"len {sc['length_ratio']}x)")
        _write(os.path.join(out_dir, "candidates", f"candidate_{i + 1}.md"), cand)
    best = min(scores, key=lambda s: s["score"])
    log(f"selected candidate {best['index']} (score {best['score']})")
    return candidates[best["index"] - 1], scores


def strip_wrapper(text: str) -> str:
    """Drop common model wrappers (```markdown fences, 'Here is the draft:')."""
    t = text.strip()
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl > 0:
            t = t[first_nl + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    for prefix in ("Here is the final draft:", "Here is the draft:",
                   "Here is the revised draft:", "Final draft:"):
        if t.lower().startswith(prefix.lower()):
            t = t[len(prefix):].lstrip(": \n")
    return t.strip()


# --------------------------------------------------------------------------
# step 4: gate + repair
# --------------------------------------------------------------------------

def build_repair_messages(draft: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    system = (
        "You are editing a draft to fix specific measured problems. Change "
        "only what the list says to fix. Keep every fact, claim, quote, and "
        "page number. Keep the same sections and the same structure. Output "
        "only the revised draft."
    )
    user = (
        "--- DRAFT ---\n"
        f"{draft}\n\n"
        "The draft was measured and the following numbers are out of range. "
        "Fix ONLY these, with the smallest edits possible:\n"
        + "\n".join(f"- {f}" for f in result["failures"])
        + "\n\n"
        "Do not change anything else: no rewording that is not on this list, "
        "no added or removed sentences beyond what the list requires, no "
        "changes to structure. Output only the revised draft."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def run_gate_loop(draft: str, rough: str, client, cfg: dict[str, Any],
                  log=print) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Gate; on failure run targeted repairs, up to max_repair_iterations.

    Returns (final_text, last gate result, iteration log).
    """
    max_iter = int(get(cfg, "gate.max_repair_iterations", 3))
    iters: list[dict[str, Any]] = []
    for iteration in range(max_iter + 1):
        result = gate.evaluate(rough, draft, cfg)
        iters.append({
            "iteration": iteration,
            "passed": result["passed"],
            "failures": result["failures"],
            "checks": {k: v for k, v in result["checks"].items()},
        })
        if result["passed"]:
            log(f"gate: PASS after {iteration} repair iteration(s)")
            return draft, result, iters
        if iteration == max_iter:
            log(f"gate: still failing after {max_iter} repairs; stopping")
            return draft, result, iters
        log(f"gate: {len(result['failures'])} check(s) failing -> repair pass "
            f"{iteration + 1}/{max_iter}")
        s = cfg.get("sampling", {})
        messages = build_repair_messages(draft, result)
        draft = strip_wrapper(client.chat(
            messages,
            temperature=s.get("temperature"),
            top_p=s.get("top_p"),
            presence_penalty=s.get("presence_penalty"),
            max_tokens=s.get("max_tokens"),
        ))
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _fmt_check(name: str, c: dict) -> str:
    ok = "PASS" if c["ok"] else "**FAIL**"
    if name == "fk_grade":
        return (f"| Flesch-Kincaid grade | {c['rough']} | {c['final']} | "
                f"±{c['tolerance']} | {ok} |")
    if name == "mean_sentence_length":
        return (f"| Mean sentence length | {c['rough']} | {c['final']} "
                f"(ratio {c['ratio']}) | {c['band'][0]}–{c['band'][1]} | {ok} |")
    if name == "ttr":
        return (f"| TTR | {c['rough']} | {c['final']} (ratio {c['ratio']}) | "
                f"{c['band'][0]}–{c['band'][1]} | {ok} |")
    if name == "contraction_rate":
        return (f"| Contraction rate /100w | {c['rough']} | {c['final']} "
                f"({'diff ' + str(c['diff']) + ' ' + c['mode']}) | "
                f"{'≤ ' + str(c['limit']) + ' ' + c['mode']} | {ok} |")
    if name == "tell_scan":
        return f"| AI-tell scan hits | 0 required | {c['hits']} | 0 | {ok} |"
    if name == "anchor_preservation":
        return (f"| Anchor preservation | — | {c['ratio']:.0%} "
                f"({c['anchored']}/{c['total']}) | ≥ {c['minimum']:.0%} | {ok} |")
    if name == "length":
        return (f"| Word count | {c['rough']} | {c['final']} "
                f"(ratio {c['ratio']}) | {c['band'][0]}x–{c['band'][1]}x | {ok} |")
    return f"| {name} | | | | {ok} |"


def write_report(report_path: str, rough: str, final: str, prof: dict[str, Any],
                 result: dict[str, Any], iters: list[dict[str, Any]],
                 candidate_scores: list[dict[str, Any]],
                 repair_log: list[str]) -> None:
    rs, fs = result["stats"]["rough"], result["stats"]["final"]
    ng = result["non_gating"]
    lines: list[str] = []
    lines.append("# humanwrite report\n")
    lines.append(f"**Result: {'PASS' if result['passed'] else 'FAIL'}** "
                 f"({len(iters)} gate iteration(s))\n")

    lines.append("## Gate checks\n")
    lines.append("| Metric | Rough draft | Final draft | Allowed | Status |")
    lines.append("|---|---|---|---|---|")
    for name in ("fk_grade", "mean_sentence_length", "ttr", "contraction_rate",
                 "tell_scan", "anchor_preservation", "length"):
        lines.append(_fmt_check(name, result["checks"][name]))
    lines.append("")
    if result["failures"]:
        lines.append("Failing metrics:")
        lines.extend(f"- {f}" for f in result["failures"])
        lines.append("")

    lines.append("## Non-gating (informational)\n")
    lines.append("| Metric | Rough | Final |")
    lines.append("|---|---|---|")
    lines.append(f"| Sentence-length CoV | {ng['sentence_length_cov']['rough']} | "
                 f"{ng['sentence_length_cov']['final']} |")
    lines.append(f"| Passive-voice rate | {ng['passive_rate']['rough']} | "
                 f"{ng['passive_rate']['final']} |")
    lines.append(f"| Flesch Reading Ease | {ng['flesch_ease']['rough']} | "
                 f"{ng['flesch_ease']['final']} |")
    lines.append(f"| Section proportion max deviation | — | "
                 f"{result['proportions']['max_deviation']} |")
    lines.append("")

    lines.append("## Side-by-side profiles\n")
    lines.append("| | Rough draft | Final draft |")
    lines.append("|---|---|---|")
    for key, label in (("word_count", "Words"), ("sentence_count", "Sentences"),
                       ("fk_grade", "FK grade"), ("flesch_ease", "Flesch ease"),
                       ("mean_sentence_length", "Mean sentence length"),
                       ("ttr", "TTR"), ("contraction_rate", "Contractions/100w"),
                       ("passive_rate", "Passive rate")):
        lines.append(f"| {label} | {rs[key]} | {fs[key]} |")
    lines.append("")

    lines.append("## Anchor preservation\n")
    a = result["anchors"]
    lines.append(f"{a['anchored']}/{a['total']} final sentences are identical to or "
                 f"near-identical (token Jaccard ≥ {a['threshold']}) to a rough-draft "
                 f"sentence. Ratio: {a['ratio']:.0%} (required ≥ "
                 f"{result['checks']['anchor_preservation']['minimum']:.0%}).")
    lines.append("")
    if a["sample_matches"]:
        lines.append("Sample matches:")
        for m in a["sample_matches"][:10]:
            lines.append(f"- (J={m['jaccard']}) {m['final']}")
        lines.append("")

    lines.append("## Tell scan (final draft)\n")
    if result["tells"]:
        lines.extend(f"- line {v['line']}: [{v['pattern']} {v['name']}] {v['text']!r}"
                     for v in result["tells"])
    else:
        lines.append("0 violations.")
    lines.append("")

    lines.append("## Step-1 candidate scores\n")
    lines.append("| Candidate | Score | Tells | Anchors | FK Δ | MSL ratio | TTR ratio | Length |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cs in candidate_scores:
        lines.append(f"| {cs['index']} | {cs['score']} | {cs['tells']} | "
                     f"{cs['anchors']:.0%} | {cs['fk_delta']:+.2f} | "
                     f"{cs['msl_ratio']} | {cs['ttr_ratio']} | {cs['length_ratio']}x |")
    lines.append("")

    lines.append("## Iteration log\n")
    for it in iters:
        status = "PASS" if it["passed"] else f"{len(it['failures'])} failing"
        lines.append(f"- iteration {it['iteration']}: {status}")
        for f in it["failures"]:
            lines.append(f"    - {f}")
    lines.extend(f"- {msg}" for msg in repair_log)
    lines.append("")

    _write(report_path, "\n".join(lines))


# --------------------------------------------------------------------------
# top-level
# --------------------------------------------------------------------------

def run_pipeline(
    rough_path: str,
    requirements_path: str | None,
    notes_path: str | None,
    guidelines_path: str | None,
    client,
    cfg: dict[str, Any],
    out_path: str,
    log=print,
) -> dict[str, Any]:
    rough = _read(rough_path)
    requirements = _read(requirements_path)
    notes = _read(notes_path)
    guidelines = _read(guidelines_path)
    out_dir = os.path.dirname(os.path.abspath(out_path))

    # optional cap: reject --notes that are "the book, not notes". 0 = no cap
    # (fine once the endpoint's context window can hold the source material).
    max_notes = int(get(cfg, "gate.max_notes_words", 0))
    notes_words = len(metrics.word_tokens(notes))
    if max_notes > 0 and notes_words > max_notes:
        raise ValueError(
            f"--notes is {notes_words} words; the limit is {max_notes}. Reduce the "
            "material to notes and quotes (or raise gate.max_notes_words) before "
            "running the pipeline."
        )

    log("step 1/4: building author profile from rough draft")
    prof = profile.build_profile(rough)
    profile_path = os.path.join(out_dir, "author_profile.json")
    profile.write_profile(prof, profile_path)
    profile.print_summary(prof)

    log("step 1/4: generating candidates")
    winner, scores = generate_candidates(
        rough, requirements, notes, guidelines, prof, client, cfg, out_dir, log)
    _write(os.path.join(out_dir, "candidate.md"), winner)

    log("step 2/4: humanizer pass")
    humanized = humanizer.run(winner, client, cfg, log)
    humanized = strip_wrapper(humanized)
    _write(os.path.join(out_dir, "humanized.md"), humanized)

    log("step 3/4: constrain pass")
    constrained = strip_wrapper(_constrain_run(humanized, rough, prof, client, cfg, log))
    _write(os.path.join(out_dir, "constrained.md"), constrained)

    log("step 4/4: gate")
    final, result, iters = run_gate_loop(constrained, rough, client, cfg, log)
    _write(out_path, final)

    report_path = os.path.join(out_dir, "report.md")
    write_report(report_path, rough, final, prof, result, iters, scores, [])
    log(f"wrote {out_path} and {report_path}")
    return {
        "passed": result["passed"],
        "out": out_path,
        "report": report_path,
        "profile": profile_path,
        "candidate_scores": scores,
        "gate": result,
    }


def _constrain_run(draft: str, rough: str, prof: dict[str, Any], client,
                   cfg: dict[str, Any], log) -> str:
    from humanwrite.constrain import run
    return run(draft, rough, prof, client, cfg, log)
