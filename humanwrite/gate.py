"""The quantitative gate: compare final draft against the rough draft.

PASS only if every gating check holds (tolerances from config.yaml):

  fk_grade within +-fk_tolerance of the rough draft
  mean sentence length ratio in [msl_ratio_low, msl_ratio_high]
  TTR ratio in [ttr_ratio_low, ttr_ratio_high]
  contraction rate relative diff <= contraction_rel_diff_max
      (if the rough rate is below contraction_low_threshold, use absolute
       difference <= contraction_abs_diff_max instead)
  tell_scan hits == 0 on the final draft
  anchor preservation >= anchor_min  (token Jaccard >= anchor_jaccard_min)
  final word count in [length_ratio_low, length_ratio_high] x rough

Non-gating (reported): sentence-length CoV, passive rate, Flesch ease,
error-budget placement, section proportion deviations.
"""

from __future__ import annotations

import re
from typing import Any

from humanwrite import metrics, tell_scan

WORD_TOKEN_RE = re.compile(r"[a-z0-9']+")


def sentence_tokens(sentence: str) -> set[str]:
    return set(WORD_TOKEN_RE.findall(sentence.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def anchor_preservation(rough_text: str, final_text: str,
                        jaccard_min: float = 0.85) -> dict[str, Any]:
    """Fraction of final sentences that are identical to, or near-identical
    to (token Jaccard >= jaccard_min), some sentence of the rough draft.
    """
    rough_sents = metrics.split_sentences(rough_text)
    rough_tokens = [sentence_tokens(s) for s in rough_sents]
    rough_tokens = [t for t in rough_tokens if t]
    final_sents = metrics.split_sentences(final_text)
    total = 0
    anchored = 0
    matches: list[dict[str, Any]] = []
    for fs in final_sents:
        ft = sentence_tokens(fs)
        if not ft:
            continue
        total += 1
        best = 0.0
        best_s = ""
        for rt in rough_tokens:
            j = jaccard(ft, rt)
            if j > best:
                best = j
                best_s = fs
        if best >= jaccard_min:
            anchored += 1
            matches.append({"final": fs[:100], "jaccard": round(best, 3)})
    ratio = anchored / total if total else 0.0
    return {
        "anchored": anchored,
        "total": total,
        "ratio": round(ratio, 4),
        "threshold": jaccard_min,
        "sample_matches": matches[:20],
    }


def section_proportion_deviation(rough_text: str, final_text: str) -> dict[str, Any]:
    """Max absolute deviation of section (or paragraph) proportions."""
    r = metrics.structure_map(rough_text)
    f = metrics.structure_map(final_text)
    rp = r.section_proportions if r.granularity == "section" else r.paragraph_proportions
    fp = f.section_proportions if f.granularity == "section" else f.paragraph_proportions
    if not rp or not fp:
        return {"max_deviation": None, "rough_proportions": rp, "final_proportions": fp}
    # align by position; if lengths differ, compare what we can
    n = min(len(rp), len(fp))
    dev = max(abs(a - b) for a, b in zip(rp[:n], fp[:n])) if n else 1.0
    return {
        "max_deviation": round(dev, 4),
        "rough_proportions": [round(x, 3) for x in rp[:16]],
        "final_proportions": [round(x, 3) for x in fp[:16]],
    }


def _contraction_check(rough_rate: float, final_rate: float, g: dict) -> dict[str, Any]:
    if rough_rate < g.get("contraction_low_threshold", 1.0):
        ok = abs(final_rate - rough_rate) <= g.get("contraction_abs_diff_max", 1.0)
        mode = "absolute"
        limit = g.get("contraction_abs_diff_max", 1.0)
        diff = abs(final_rate - rough_rate)
    else:
        ref = max(rough_rate, 0.01)
        rel = abs(final_rate - rough_rate) / ref
        ok = rel <= g.get("contraction_rel_diff_max", 0.5)
        mode = "relative"
        limit = g.get("contraction_rel_diff_max", 0.5)
        diff = round(rel, 3)
    return {"ok": ok, "mode": mode, "rough": rough_rate, "final": final_rate,
            "diff": diff, "limit": limit}


def evaluate(rough_text: str, final_text: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run all checks. Returns:

    {
      "passed": bool,
      "checks": {name: {"ok": bool, ...values}},
      "failures": [human-readable failure lines],
      "stats": {"rough": {...}, "final": {...}},
      "anchors": {...}, "tells": [violations], "proportions": {...},
    }
    """
    g = cfg.get("gate", {})
    rs = metrics.compute_stats(rough_text)
    fs = metrics.compute_stats(final_text)
    rs["contraction_rate"] = rs.get("contraction_rate", 0.0)
    checks: dict[str, Any] = {}

    # FK grade
    fk_ok = abs(fs["fk_grade"] - rs["fk_grade"]) <= g.get("fk_tolerance", 2.0)
    checks["fk_grade"] = {
        "ok": fk_ok, "rough": rs["fk_grade"], "final": fs["fk_grade"],
        "tolerance": g.get("fk_tolerance", 2.0),
    }

    # mean sentence length ratio
    msl_ratio = (fs["mean_sentence_length"] / rs["mean_sentence_length"]) \
        if rs["mean_sentence_length"] else 1.0
    msl_ok = g.get("msl_ratio_low", 0.80) <= msl_ratio <= g.get("msl_ratio_high", 1.25)
    checks["mean_sentence_length"] = {
        "ok": msl_ok, "rough": rs["mean_sentence_length"], "final": fs["mean_sentence_length"],
        "ratio": round(msl_ratio, 3),
        "band": [g.get("msl_ratio_low", 0.80), g.get("msl_ratio_high", 1.25)],
    }

    # TTR ratio
    ttr_ratio = (fs["ttr"] / rs["ttr"]) if rs["ttr"] else 1.0
    ttr_ok = g.get("ttr_ratio_low", 0.85) <= ttr_ratio <= g.get("ttr_ratio_high", 1.15)
    checks["ttr"] = {
        "ok": ttr_ok, "rough": rs["ttr"], "final": fs["ttr"],
        "ratio": round(ttr_ratio, 3),
        "band": [g.get("ttr_ratio_low", 0.85), g.get("ttr_ratio_high", 1.15)],
    }

    # contractions
    checks["contraction_rate"] = _contraction_check(
        rs["contraction_rate"], fs["contraction_rate"], g)

    # tell scan
    tells = tell_scan.scan_text(final_text, cfg)
    checks["tell_scan"] = {"ok": len(tells) == 0, "hits": len(tells)}

    # anchor preservation
    anchors = anchor_preservation(rough_text, final_text,
                                  g.get("anchor_jaccard_min", 0.85))
    anchor_ok = anchors["ratio"] >= g.get("anchor_min", 0.30)
    checks["anchor_preservation"] = {
        "ok": anchor_ok, "ratio": anchors["ratio"],
        "minimum": g.get("anchor_min", 0.30),
        "anchored": anchors["anchored"], "total": anchors["total"],
    }

    # length
    len_ratio = (fs["word_count"] / rs["word_count"]) if rs["word_count"] else 1.0
    len_ok = g.get("length_ratio_low", 0.9) <= len_ratio <= g.get("length_ratio_high", 1.5)
    checks["length"] = {
        "ok": len_ok, "rough": rs["word_count"], "final": fs["word_count"],
        "ratio": round(len_ratio, 3),
        "band": [g.get("length_ratio_low", 0.9), g.get("length_ratio_high", 1.5)],
    }

    failures: list[str] = []
    for name, c in checks.items():
        if not c["ok"]:
            failures.append(_failure_line(name, c, rs, fs, tells))

    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "stats": {"rough": rs, "final": fs},
        "anchors": anchors,
        "tells": tells,
        "proportions": section_proportion_deviation(rough_text, final_text),
        "non_gating": {
            "sentence_length_cov": {"rough": rs["sentence_length_cov"],
                                    "final": fs["sentence_length_cov"]},
            "passive_rate": {"rough": rs["passive_rate"], "final": fs["passive_rate"]},
            "flesch_ease": {"rough": rs["flesch_ease"], "final": fs["flesch_ease"]},
        },
    }


def _failure_line(name: str, c: dict, rs: dict, fs: dict, tells: list) -> str:
    if name == "fk_grade":
        return (f"Flesch-Kincaid grade is {c['final']} but must be within "
                f"{c['rough'] - c['tolerance']:.1f} to {c['rough'] + c['tolerance']:.1f} "
                f"(rough draft is {c['rough']}). Adjust sentence length or word length "
                f"to move it into band.")
    if name == "mean_sentence_length":
        lo, hi = c["band"]
        return (f"Mean sentence length ratio is {c['ratio']} (rough {c['rough']}, "
                f"final {c['final']}) but must be between {lo} and {hi}. "
                f"{'Split some long sentences.' if c['ratio'] > hi else 'Merge or lengthen some sentences.'}")
    if name == "ttr":
        lo, hi = c["band"]
        return (f"TTR ratio is {c['ratio']} (rough {c['rough']}, final {c['final']}) but "
                f"must be between {lo} and {hi}. "
                f"{'Vary word choice: replace repeated words with synonyms.' if c['ratio'] < lo else 'Reduce unusual vocabulary.'}")
    if name == "contraction_rate":
        mode = c["mode"]
        limit = c["limit"]
        if mode == "relative":
            return (f"Contraction rate is {c['final']}/100 words, relative difference from "
                    f"the rough draft's {c['rough']}/100 is {c['diff']} but must be <= {limit:.0%}. "
                    f"Convert plain forms to contractions (do not -> don't, cannot -> can't, "
                    f"is not -> isn't) or split contractions back, whichever moves it into band.")
        return (f"Contraction rate is {c['final']}/100 words, absolute difference from the "
                f"rough draft's {c['rough']}/100 is {c['diff']} but must be <= {limit}.")
    if name == "tell_scan":
        return (f"Tell scan found {c['hits']} AI-tell pattern(s) on the final draft; it must "
                f"be 0. Remove the listed words and constructions (see tell scan output).")
    if name == "anchor_preservation":
        return (f"Anchor preservation is {c['ratio']:.0%} of sentences but must be at least "
                f"{c['minimum']:.0%}. Copy more sentences from the rough draft verbatim "
                f"(or nearly verbatim) into the final draft.")
    if name == "length":
        lo, hi = c["band"]
        return (f"Final length is {c['ratio']}x the rough draft ({c['final']} words vs "
                f"{c['rough']}) but must be between {lo}x and {hi}x. "
                f"{'Expand sections' if c['ratio'] < lo else 'Trim the draft'} to get into band.")
    return f"{name} failed"
