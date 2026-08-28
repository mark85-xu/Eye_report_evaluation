"""P1 clinical-consistency metric built on top of the frozen report_parser.

This module is NOT an official ranking metric.  It only answers the analysis
question: *"is a low BLEU caused by a real medical-status error, or only by
wording?"*  It reuses ``src/data/report_parser.parse_report`` unchanged so the
P0 concepts always follow the frozen v0.1.1 definitions.

Unknown handling: ``unknown`` from the parser means "insufficient/unresolved".
It is NEVER treated as 0.  Samples where GT or prediction is unknown for a
concept are excluded from that concept's F1 and are reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from src.data.report_parser import P0_CONCEPTS, UNKNOWN, parse_report

BINARY_CONCEPTS = [
    "roughness",
    "imt_thickening",
    "left_plaque",
    "right_plaque",
    "stenosis",
    "flow_normal",
]
MAIN_STATUS_CLASSES = ["normal", "wall_abnormal_no_plaque", "plaque"]
_MAIN_STATUS_KEY = "main_status"


def _parse_many(reports: Sequence[Any]) -> list[dict[str, str]]:
    return [parse_report(r).labels for r in reports]


def _binary_prf(
    gt_labels: list[str], pred_labels: list[str]
) -> dict[str, float | int | None]:
    """Precision/recall/F1 treating '1' as positive, excluding unknown."""
    tp = fp = fn = tn = unknown = 0
    for g, p in zip(gt_labels, pred_labels):
        if g == UNKNOWN or p == UNKNOWN:
            unknown += 1
            continue
        if g == "1":
            if p == "1":
                tp += 1
            else:
                fn += 1
        else:
            if p == "1":
                fp += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = 2 * precision * recall / (precision + recall) if (precision and recall) else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "unknown": unknown,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _main_status_stats(
    gt_labels: list[str], pred_labels: list[str]
) -> dict[str, Any]:
    acc_n = acc_match = 0
    per_class = {}
    micro_tp = micro_fp = micro_fn = 0
    for cls in MAIN_STATUS_CLASSES:
        tp = fp = fn = unknown = 0
        for g, p in zip(gt_labels, pred_labels):
            if g == UNKNOWN or p == UNKNOWN:
                unknown += 1
                continue
            g_bin = int(g == cls)
            p_bin = int(p == cls)
            if g_bin == 1 and p_bin == 1:
                tp += 1
            elif g_bin == 0 and p_bin == 1:
                fp += 1
            elif g_bin == 1 and p_bin == 0:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = 2 * precision * recall / (precision + recall) if (precision and recall) else None
        per_class[cls] = {"tp": tp, "fp": fp, "fn": fn, "unknown": unknown,
                          "precision": precision, "recall": recall, "f1": f1}
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
    for g, p in zip(gt_labels, pred_labels):
        if g == UNKNOWN or p == UNKNOWN:
            continue
        acc_n += 1
        if g == p:
            acc_match += 1
    macro_f1s = [per_class[c]["f1"] for c in MAIN_STATUS_CLASSES if per_class[c]["f1"] is not None]
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else None
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else None
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p and micro_r) else None
    return {
        "accuracy": acc_match / acc_n if acc_n else None,
        "n_scored": acc_n,
        "macro_f1": sum(macro_f1s) / len(macro_f1s) if macro_f1s else None,
        "micro_f1": micro_f1,
        "per_class": per_class,
    }


def _sample_errors(
    gt: dict[str, str], pred: dict[str, str], bleu4: float | None, bleu_word_only_threshold: float
) -> list[str]:
    tags: list[str] = []

    def flip(c: str) -> bool:
        return (
            gt.get(c) in {"0", "1"}
            and pred.get(c) in {"0", "1"}
            and gt.get(c) != pred.get(c)
        )

    left_gt, right_gt = gt.get("left_plaque"), gt.get("right_plaque")
    left_pr, right_pr = pred.get("left_plaque"), pred.get("right_plaque")
    side_swapped = (
        left_gt in {"0", "1"}
        and right_gt in {"0", "1"}
        and left_gt != right_gt
        and left_pr == right_gt
        and right_pr == left_gt
    )
    if side_swapped:
        tags.append("SIDE_ERROR")

    for side, g, p in [
        ("left_plaque", left_gt, left_pr),
        ("right_plaque", right_gt, right_pr),
    ]:
        if g == "1" and p in {"0", UNKNOWN}:
            tags.append("PLAQUE_FALSE_NEGATIVE")
        elif g in {"0", UNKNOWN} and p == "1":
            tags.append("PLAQUE_FALSE_POSITIVE")

    # The parser encodes side-level plaque as "0" only for definite negatives;
    # a report that simply states ``可见斑块形成`` (no side) leaves side labels
    # unknown while main_status becomes "plaque".  Compare main_status too so
    # those predictions are still flagged as plaque FP / FN.
    gt_ms, pr_ms = gt.get("main_status"), pred.get("main_status")
    if pr_ms == "plaque" and gt_ms != "plaque":
        tags.append("PLAQUE_FALSE_POSITIVE")
    if gt_ms == "plaque" and pr_ms != "plaque":
        tags.append("PLAQUE_FALSE_NEGATIVE")

    if flip("roughness"):
        tags.append("ROUGHNESS_ERROR")
    if flip("imt_thickening"):
        tags.append("IMT_ERROR")
    if flip("stenosis"):
        tags.append("STENOSIS_ERROR")
    if flip("flow_normal"):
        tags.append("FLOW_ERROR")

    neg_flags = {
        "roughness", "imt_thickening", "left_plaque", "right_plaque",
        "stenosis", "flow_normal",
    }
    plaque_flipped = (gt_ms == "plaque") != (pr_ms == "plaque")
    if any(flip(c) for c in neg_flags) or plaque_flipped:
        tags.append("NEGATION_ERROR")

    all_match = all(
        gt.get(c) == pred.get(c) for c in P0_CONCEPTS
    )
    if all_match and bleu4 is not None and bleu4 < bleu_word_only_threshold:
        tags.append("WORDING_ONLY")
    return tags


@dataclass
class ClinicalResult:
    summary: dict[str, Any] = field(default_factory=dict)
    per_sample: pd.DataFrame = field(default_factory=pd.DataFrame)


def evaluate_clinical(
    gt_reports: Sequence[Any],
    pred_reports: Sequence[Any],
    ids: Sequence[Any] | None = None,
    gt_reports_tokenized: Sequence[str] | None = None,
    pred_reports_tokenized: Sequence[str] | None = None,
    bleu4_per_sample: Mapping[Any, float] | None = None,
    bleu_word_only_threshold: float = 0.5,
) -> ClinicalResult:
    """Compare GT vs prediction clinical concepts via the frozen parser.

    ``bleu4_per_sample`` (optional, keyed by id) enables the WORDING_ONLY
    tag; without it that tag is simply not produced.
    """
    gt_parsed = _parse_many(gt_reports)
    pred_parsed = _parse_many(pred_reports)
    if len(gt_parsed) != len(pred_parsed):
        raise ValueError("gt and prediction must have the same length")

    if ids is None:
        ids = list(range(len(gt_parsed)))

    summary: dict[str, Any] = {"parser_version": "v0.1.1 (frozen)"}

    for concept in BINARY_CONCEPTS:
        gt_c = [d[concept] for d in gt_parsed]
        pr_c = [d[concept] for d in pred_parsed]
        summary[concept] = _binary_prf(gt_c, pr_c)

    summary[_MAIN_STATUS_KEY] = _main_status_stats(
        [d[_MAIN_STATUS_KEY] for d in gt_parsed],
        [d[_MAIN_STATUS_KEY] for d in pred_parsed],
    )

    macro_f1s = [
        summary[c]["f1"]
        for c in BINARY_CONCEPTS
        if summary[c]["f1"] is not None
    ]
    ms_macro = summary[_MAIN_STATUS_KEY]["macro_f1"]
    if ms_macro is not None:
        macro_f1s.append(ms_macro)
    summary["clinical_macro_f1"] = (
        sum(macro_f1s) / len(macro_f1s) if macro_f1s else None
    )

    micro_tp = micro_fp = micro_fn = 0
    for c in BINARY_CONCEPTS:
        micro_tp += summary[c]["tp"]
        micro_fp += summary[c]["fp"]
        micro_fn += summary[c]["fn"]
    ms = summary[_MAIN_STATUS_KEY]
    for cls in MAIN_STATUS_CLASSES:
        micro_tp += ms["per_class"][cls]["tp"]
        micro_fp += ms["per_class"][cls]["fp"]
        micro_fn += ms["per_class"][cls]["fn"]
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else None
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else None
    summary["clinical_micro_f1"] = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p and micro_r) else None
    )

    n_exact_all = sum(
        1
        for g, p in zip(gt_parsed, pred_parsed)
        if all(g.get(c) == p.get(c) for c in P0_CONCEPTS)
    )
    summary["clinical_exact_match"] = n_exact_all / len(gt_parsed) if gt_parsed else None

    rows = []
    for i, (g, p) in enumerate(zip(gt_parsed, pred_parsed)):
        row = {"id": ids[i]}
        for c in P0_CONCEPTS:
            row[f"gt_{c}"] = g[c]
            row[f"pred_{c}"] = p[c]
        bleu4 = None
        if bleu4_per_sample is not None:
            bleu4 = bleu4_per_sample.get(ids[i])
        row["error_tags"] = "|".join(
            _sample_errors(g, p, bleu4, bleu_word_only_threshold)
        ) or "MATCH"
        rows.append(row)
    per_sample = pd.DataFrame(rows)
    return ClinicalResult(summary=summary, per_sample=per_sample)
