#!/usr/bin/env python
"""Rule-based parser for carotid ultrasound report_2 supervision labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PARSER_VERSION = "v0.1.1"
UNKNOWN = "unknown"
P0_CONCEPTS = [
    "main_status",
    "roughness",
    "imt_thickening",
    "left_plaque",
    "right_plaque",
    "stenosis",
    "flow_normal",
]
LABEL_COLUMNS = [
    "id",
    "official_id",
    "split",
    "parser_version",
    "parser_status",
    "conflict_flag",
    "main_status",
    "roughness",
    "imt_thickening",
    "left_plaque",
    "right_plaque",
    "stenosis",
    "flow_normal",
]
CONCEPT_AUDIT_COLUMNS = [
    "concept",
    "parser_version",
    "split",
    "total",
    "positive_count",
    "negative_count",
    "unknown_count",
    "unknown_ratio",
    "positive_ratio",
    "normal_count",
    "wall_abnormal_no_plaque_count",
    "plaque_count",
    "conflict_count",
    "conflict_ratio",
    "candidate_mention_count",
    "recommended_for_v0",
    "notes",
]
AUDIT_SAMPLE_COLUMNS = [
    "id",
    "official_id",
    "split",
    "report_2",
    "concept",
    "predicted_label",
    "matched_clause",
    "matched_rule",
    "review_status",
    "review_note",
]
PARSER_ERROR_COLUMNS = [
    "id",
    "official_id",
    "split",
    "error_type",
    "concept",
    "detail",
    "report_2",
]

NEGATION_TERMS = [
    "未见明显",
    "未见",
    "未探及明显",
    "未探及",
    "未发现",
    "无明显",
    "无",
]
LEFT_TERMS = ["左侧", "左侧颈", "左颈", "左"]
RIGHT_TERMS = ["右侧", "右侧颈", "右颈", "右"]
BILATERAL_TERMS = ["双侧", "两侧"]
ROUGHNESS_POSITIVE_TERMS = ["毛糙", "稍毛糙", "表面毛糙", "不光滑"]
ROUGHNESS_NEGATIVE_TERMS = ["内膜光滑", "内-中膜光滑", "内中膜光滑", "管壁光滑", "内壁光滑"]
IMT_POSITIVE_PATTERNS = [
    "内-中膜增厚",
    "内中膜增厚",
    "内膜增厚",
    "不均增厚",
    "局部增厚",
    "轻度增厚",
    "稍增厚",
    "较厚",
    "内-中膜厚",
    "内中膜厚",
    "增厚伴",
]
IMT_NEGATIVE_PATTERNS = [
    "未见明显增厚",
    "无增厚",
    "光滑无增厚",
    "未见增厚",
    "未见明显内-中膜增厚",
    "未见明显内中膜增厚",
    "内-中膜不厚",
    "内中膜不厚",
    "内-中膜正常",
    "内中膜正常",
]
PLAQUE_TERMS = ["斑块"]
PLAQUE_POSITIVE_HINTS = [
    "可见",
    "见一",
    "见数",
    "探及",
    "伴斑块",
    "斑块形成",
    "斑块回声",
    "斑块大小",
    "斑块较",
    "较小斑块",
    "中等斑块",
    "较大斑块",
    "强回声斑块",
    "低回声斑块",
    "混合回声斑块",
    "多个斑块",
    "数个斑块",
]
PLAQUE_NEGATIVE_PATTERNS = [
    "未见明显异常斑块",
    "未见明显斑块",
    "未见斑块",
    "未探及明显斑块",
    "未探及斑块",
    "未发现斑块",
    "无明显斑块",
    "无斑块",
    "未见明显异常斑块回声",
    "未见明显斑块回声",
    "未见明显斑块形成",
]
STENOSIS_NEGATIVE_PATTERNS = [
    "未见明显狭窄",
    "未见狭窄",
    "无明显狭窄",
    "无狭窄",
    "管腔无狭窄",
    "无狭窄征象",
    "管腔未见明显狭窄",
    "管腔未见狭窄",
]
STENOSIS_POSITIVE_PATTERNS = [
    "可见狭窄",
    "轻度狭窄",
    "局部狭窄",
    "狭窄率",
    "狭窄约",
    "狭窄达",
]
FLOW_NORMAL_PATTERNS = [
    "血流通畅",
    "血流畅通",
    "血流充填良好",
    "血流充盈良好",
    "血流充盈佳",
    "显示血流充填良好",
    "示血流充填良好",
    "示血流充盈良好",
    "示血流充盈佳",
]
P1_RULES = {
    "left_echo": ["低回声", "等回声", "强回声", "混合回声", "混合性"],
    "right_echo": ["低回声", "等回声", "强回声", "混合回声", "混合性"],
    "left_size": ["较小", "中等", "较大"],
    "right_size": ["较小", "中等", "较大"],
    "left_site": ["分叉处", "颈总动脉分叉", "颈内动脉起始处", "窦部"],
    "right_site": ["分叉处", "颈总动脉分叉", "颈内动脉起始处", "窦部"],
    "plaque_multiple": ["多个斑块", "数个斑块", "多发斑块"],
}


@dataclass
class Evidence:
    value: str
    clause: str
    rule: str


@dataclass
class ParseResult:
    labels: dict[str, str]
    evidence: dict[str, Evidence] = field(default_factory=dict)
    conflicts: dict[str, list[str]] = field(default_factory=dict)
    parser_status: str = "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse report_2 into P0 labels.")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--output", default="data/labels.csv")
    parser.add_argument("--audit-output", default="outputs/concept_audit.csv")
    parser.add_argument("--audit-samples", default="outputs/parser_audit_samples.csv")
    parser.add_argument("--errors-output", default="outputs/parser_errors.csv")
    parser.add_argument("--version-output", default="parser_version")
    parser.add_argument("--audit-samples-per-concept", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if not isinstance(value, str):
        return ""
    return unicodedata.normalize("NFKC", value).strip()


def split_clauses(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    clauses = [c.strip() for c in re.split(r"[，,。；;：:\n]+", text) if c.strip()]
    return clauses


def has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def first_match(text: str, terms: list[str]) -> str:
    for term in terms:
        if term in text:
            return term
    return ""


def detect_negation(clause: str, entity: str) -> bool:
    pos = clause.find(entity)
    if pos < 0:
        return False
    prefix = clause[max(0, pos - 12) : pos]
    return has_any(prefix, NEGATION_TERMS)


def detect_laterality(clause: str) -> set[str]:
    sides: set[str] = set()
    if has_any(clause, BILATERAL_TERMS):
        sides.update({"left", "right"})
    if has_any(clause, LEFT_TERMS):
        sides.add("left")
    if has_any(clause, RIGHT_TERMS):
        sides.add("right")
    return sides


def is_explicit_bilateral_plaque(clause: str) -> bool:
    if not has_any(clause, BILATERAL_TERMS) or "斑块" not in clause:
        return False
    explicit_patterns = [
        "双侧可见",
        "双侧颈动脉可见",
        "双侧颈动脉内壁可见",
        "双侧动脉内壁上可见",
        "双侧动脉内壁可见",
        "双侧颈动脉壁上可见",
    ]
    return has_any(clause, explicit_patterns)


def set_evidence(
    evidence: dict[str, Evidence],
    concept: str,
    value: str,
    clause: str,
    rule: str,
) -> None:
    if concept not in evidence:
        evidence[concept] = Evidence(value=value, clause=clause, rule=rule)


def parse_roughness(clauses: list[str]) -> tuple[str, Evidence | None]:
    positive: Evidence | None = None
    negative: Evidence | None = None
    for clause in clauses:
        if has_any(clause, ROUGHNESS_POSITIVE_TERMS) and not detect_negation(
            clause, "毛糙"
        ):
            positive = positive or Evidence("1", clause, "positive_roughness")
        elif has_any(clause, ROUGHNESS_NEGATIVE_TERMS):
            negative = negative or Evidence("0", clause, "negative_smooth_intima")
    if positive:
        return "1", positive
    if negative:
        return "0", negative
    return UNKNOWN, None


def parse_imt(clauses: list[str]) -> tuple[str, Evidence | None]:
    positive: Evidence | None = None
    negative: Evidence | None = None
    for clause in clauses:
        if has_any(clause, IMT_NEGATIVE_PATTERNS):
            negative = negative or Evidence("0", clause, "negative_imt_not_thick")
        if has_any(clause, IMT_POSITIVE_PATTERNS):
            if not has_any(clause, IMT_NEGATIVE_PATTERNS) and not detect_negation(
                clause, "增厚"
            ):
                positive = positive or Evidence("1", clause, "positive_imt_thickening")
    if positive:
        return "1", positive
    if negative:
        return "0", negative
    return UNKNOWN, None


def plaque_clause_polarity(clause: str) -> str:
    if not has_any(clause, PLAQUE_TERMS):
        return "none"
    if has_any(clause, PLAQUE_NEGATIVE_PATTERNS) or detect_negation(clause, "斑块"):
        if has_any(clause, PLAQUE_POSITIVE_HINTS) and not has_any(
            clause, PLAQUE_NEGATIVE_PATTERNS
        ):
            return "conflict"
        return "negative"
    if has_any(clause, PLAQUE_POSITIVE_HINTS):
        return "positive"
    return UNKNOWN


def parse_plaque(
    clauses: list[str],
) -> tuple[dict[str, str], dict[str, Evidence], dict[str, list[str]], bool]:
    side_votes = {"left": set(), "right": set()}
    evidence: dict[str, Evidence] = {}
    conflicts: dict[str, list[str]] = {}
    general_positive = False
    context_sides: set[str] = set()

    for clause in clauses:
        polarity = plaque_clause_polarity(clause)
        explicit_sides = detect_laterality(clause)
        if explicit_sides:
            context_sides = set(explicit_sides)
        if polarity == "none":
            continue

        sides: set[str] = set()
        if polarity == "positive":
            if explicit_sides == {"left", "right"}:
                sides = explicit_sides if is_explicit_bilateral_plaque(clause) else set()
            elif explicit_sides:
                sides = explicit_sides
            elif len(context_sides) == 1:
                sides = set(context_sides)
        elif polarity == "negative":
            if explicit_sides:
                sides = explicit_sides
            elif len(context_sides) == 1:
                sides = set(context_sides)
            else:
                sides = {"left", "right"}
        elif polarity == UNKNOWN and explicit_sides:
            sides = explicit_sides

        if not sides and polarity == "positive":
            general_positive = True
            set_evidence(
                evidence,
                "main_status",
                "plaque",
                clause,
                "general_positive_plaque_without_side",
            )
            continue
        if polarity == UNKNOWN and not sides:
            continue
        for side in sides:
            concept = f"{side}_plaque"
            if polarity == "positive":
                side_votes[side].add("1")
                set_evidence(evidence, concept, "1", clause, f"positive_{side}_plaque")
                general_positive = True
            elif polarity == "negative":
                side_votes[side].add("0")
                set_evidence(evidence, concept, "0", clause, f"negative_{side}_plaque")
            else:
                side_votes[side].add(UNKNOWN)
                set_evidence(evidence, concept, UNKNOWN, clause, "ambiguous_plaque")

    labels: dict[str, str] = {}
    for side in ["left", "right"]:
        concept = f"{side}_plaque"
        votes = side_votes[side]
        # P0 plaque is side-level presence: any reliable positive site/region on
        # that side makes the side positive. Local negative clauses do not
        # override another local positive clause; detailed site resolution is P1.
        if "1" in votes:
            labels[concept] = "1"
        elif "0" in votes:
            labels[concept] = "0"
        else:
            labels[concept] = UNKNOWN
    return labels, evidence, conflicts, general_positive


def parse_stenosis(
    clauses: list[str],
) -> tuple[str, Evidence | None, dict[str, list[str]]]:
    positive: Evidence | None = None
    negative: Evidence | None = None
    for clause in clauses:
        if "狭窄" not in clause:
            continue
        if has_any(clause, STENOSIS_NEGATIVE_PATTERNS) or detect_negation(clause, "狭窄"):
            negative = negative or Evidence("0", clause, "negative_stenosis")
        if has_any(clause, STENOSIS_POSITIVE_PATTERNS):
            positive = positive or Evidence("1", clause, "positive_stenosis")
    if positive and negative:
        return UNKNOWN, Evidence(UNKNOWN, positive.clause, "conflict_stenosis"), {
            "stenosis": [positive.clause, negative.clause]
        }
    if positive:
        return "1", positive, {}
    if negative:
        return "0", negative, {}
    return UNKNOWN, None, {}


def parse_flow(clauses: list[str]) -> tuple[str, Evidence | None]:
    for clause in clauses:
        term = first_match(clause, FLOW_NORMAL_PATTERNS)
        if term:
            return "1", Evidence("1", clause, f"positive_flow_normal:{term}")
    return UNKNOWN, None


def derive_main_status(
    labels: dict[str, str],
    plaque_any_positive: bool,
    clauses: list[str],
) -> tuple[str, Evidence | None]:
    if plaque_any_positive or labels["left_plaque"] == "1" or labels["right_plaque"] == "1":
        return "plaque", Evidence("plaque", "", "derived_from_plaque_positive")

    plaque_explicit_negative = labels["left_plaque"] == "0" and labels["right_plaque"] == "0"
    wall_positive = labels["roughness"] == "1" or labels["imt_thickening"] == "1"
    wall_negative = labels["roughness"] == "0" and labels["imt_thickening"] == "0"

    if wall_positive:
        return "wall_abnormal_no_plaque", Evidence(
            "wall_abnormal_no_plaque",
            "",
            "derived_from_wall_abnormal_without_plaque_positive",
        )
    if wall_negative and plaque_explicit_negative:
        return "normal", Evidence("normal", "", "derived_from_normal_wall_and_no_plaque")

    has_normal_lumen = any(("内径正常" in c or "管径正常" in c or "管径处于正常范围" in c) for c in clauses)
    if has_normal_lumen and plaque_explicit_negative and labels["stenosis"] == "0":
        return "normal", Evidence("normal", "", "derived_from_lumen_no_plaque_no_stenosis")
    return UNKNOWN, None


def parse_report(report: Any) -> ParseResult:
    text = normalize_text(report)
    if not text:
        labels = {concept: UNKNOWN for concept in P0_CONCEPTS}
        return ParseResult(labels=labels, parser_status="empty_report")

    clauses = split_clauses(text)
    labels: dict[str, str] = {}
    evidence: dict[str, Evidence] = {}
    conflicts: dict[str, list[str]] = {}

    labels["roughness"], rough_ev = parse_roughness(clauses)
    if rough_ev:
        evidence["roughness"] = rough_ev

    labels["imt_thickening"], imt_ev = parse_imt(clauses)
    if imt_ev:
        evidence["imt_thickening"] = imt_ev

    plaque_labels, plaque_evidence, plaque_conflicts, plaque_any_positive = parse_plaque(
        clauses
    )
    labels.update(plaque_labels)
    evidence.update(plaque_evidence)
    conflicts.update(plaque_conflicts)

    labels["stenosis"], stenosis_ev, stenosis_conflicts = parse_stenosis(clauses)
    if stenosis_ev:
        evidence["stenosis"] = stenosis_ev
    conflicts.update(stenosis_conflicts)

    labels["flow_normal"], flow_ev = parse_flow(clauses)
    if flow_ev:
        evidence["flow_normal"] = flow_ev

    labels["main_status"], main_ev = derive_main_status(
        labels, plaque_any_positive, clauses
    )
    if main_ev:
        evidence["main_status"] = main_ev

    consistency_conflicts = validate_parsed_result(labels)
    for concept, items in consistency_conflicts.items():
        conflicts.setdefault(concept, []).extend(items)
    return ParseResult(labels=labels, evidence=evidence, conflicts=conflicts)


def validate_parsed_result(labels: dict[str, str]) -> dict[str, list[str]]:
    conflicts: dict[str, list[str]] = {}
    if labels["main_status"] == "normal":
        for concept in ["roughness", "imt_thickening", "left_plaque", "right_plaque", "stenosis"]:
            if labels[concept] == "1":
                conflicts.setdefault("main_status", []).append(
                    f"normal_conflicts_with_{concept}=1"
                )
    if labels["main_status"] == "wall_abnormal_no_plaque":
        if labels["left_plaque"] == "1" or labels["right_plaque"] == "1":
            conflicts.setdefault("main_status", []).append(
                "wall_abnormal_status_conflicts_with_plaque_positive"
            )
    return conflicts


def load_metadata(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"metadata not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    required = {"id", "official_id", "split", "report_2"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"metadata missing required columns: {sorted(missing)}")
    ids = [row["id"] for row in rows]
    duplicate_ids = sorted([k for k, v in Counter(ids).items() if v > 1])
    if duplicate_ids:
        raise ValueError(f"metadata.id must be unique; duplicates: {duplicate_ids[:10]}")
    invalid_splits = sorted({row["split"] for row in rows} - {"train", "val"})
    if invalid_splits:
        raise ValueError(f"metadata split values must be train/val: {invalid_splits}")
    return rows


def build_labels(metadata_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, ParseResult]]:
    label_rows: list[dict[str, str]] = []
    error_rows: list[dict[str, str]] = []
    parsed_by_id: dict[str, ParseResult] = {}
    for row in metadata_rows:
        result = parse_report(row.get("report_2"))
        parsed_by_id[row["id"]] = result
        conflict_flag = "1" if result.conflicts else "0"
        label_row = {
            "id": row["id"],
            "official_id": row.get("official_id", ""),
            "split": row["split"],
            "parser_version": PARSER_VERSION,
            "parser_status": result.parser_status,
            "conflict_flag": conflict_flag,
        }
        for concept in P0_CONCEPTS:
            label_row[concept] = result.labels[concept]
        label_rows.append(label_row)
        for concept, details in result.conflicts.items():
            error_rows.append(
                {
                    "id": row["id"],
                    "official_id": row.get("official_id", ""),
                    "split": row["split"],
                    "error_type": "parse_conflict",
                    "concept": concept,
                    "detail": " | ".join(details),
                    "report_2": row.get("report_2", ""),
                }
            )
        if result.parser_status != "ok":
            error_rows.append(
                {
                    "id": row["id"],
                    "official_id": row.get("official_id", ""),
                    "split": row["split"],
                    "error_type": result.parser_status,
                    "concept": "",
                    "detail": result.parser_status,
                    "report_2": row.get("report_2", ""),
                }
            )
    return label_rows, error_rows, parsed_by_id


def p1_candidate_count(concept: str, rows: list[dict[str, str]]) -> int:
    terms = P1_RULES[concept]
    if concept.startswith("left_"):
        return sum("左侧" in row["report_2"] and has_any(row["report_2"], terms) for row in rows)
    if concept.startswith("right_"):
        return sum("右侧" in row["report_2"] and has_any(row["report_2"], terms) for row in rows)
    return sum(has_any(row["report_2"], terms) for row in rows)


def build_concept_audit(
    metadata_rows: list[dict[str, str]],
    label_rows: list[dict[str, str]],
    parsed_by_id: dict[str, ParseResult],
) -> list[dict[str, str]]:
    meta_by_id = {row["id"]: row for row in metadata_rows}
    groups = {
        "Overall": label_rows,
        "train": [row for row in label_rows if row["split"] == "train"],
        "val": [row for row in label_rows if row["split"] == "val"],
    }
    audit_rows: list[dict[str, str]] = []
    for split_name, rows in groups.items():
        total = len(rows)
        for concept in P0_CONCEPTS:
            values = Counter(row[concept] for row in rows)
            conflict_count = sum(1 for row in rows if concept in parsed_by_id[row["id"]].conflicts)
            positive = values.get("1", 0)
            negative = values.get("0", 0)
            unknown = values.get(UNKNOWN, 0)
            normal = values.get("normal", 0)
            wall = values.get("wall_abnormal_no_plaque", 0)
            plaque = values.get("plaque", 0)
            if concept == "main_status":
                unknown = values.get(UNKNOWN, 0)
                positive = plaque + wall
                negative = normal
            audit_rows.append(
                {
                    "concept": concept,
                    "parser_version": PARSER_VERSION,
                    "split": split_name,
                    "total": str(total),
                    "positive_count": str(positive),
                    "negative_count": str(negative),
                    "unknown_count": str(unknown),
                    "unknown_ratio": f"{unknown / total:.6f}" if total else "0",
                    "positive_ratio": f"{positive / total:.6f}" if total else "0",
                    "normal_count": str(normal),
                    "wall_abnormal_no_plaque_count": str(wall),
                    "plaque_count": str(plaque),
                    "conflict_count": str(conflict_count),
                    "conflict_ratio": f"{conflict_count / total:.6f}" if total else "0",
                    "candidate_mention_count": "",
                    "recommended_for_v0": "yes" if concept in P0_CONCEPTS else "no",
                    "notes": "P0 label generated by report_parser.py",
                }
            )

        split_meta = [meta_by_id[row["id"]] for row in rows]
        for concept in P1_RULES:
            candidate_mentions = p1_candidate_count(concept, split_meta)
            audit_rows.append(
                {
                    "concept": concept,
                    "parser_version": PARSER_VERSION,
                    "split": split_name,
                    "total": str(total),
                    "positive_count": "",
                    "negative_count": "",
                    "unknown_count": str(total),
                    "unknown_ratio": "1.000000" if total else "0",
                    "positive_ratio": "",
                    "normal_count": "",
                    "wall_abnormal_no_plaque_count": "",
                    "plaque_count": "",
                    "conflict_count": "",
                    "conflict_ratio": "",
                    "candidate_mention_count": str(candidate_mentions),
                    "recommended_for_v0": "no",
                    "notes": "P1 candidate mentions audited only; side binding requires manual review before training.",
                }
            )
    return audit_rows


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_parser_audit_samples(
    metadata_rows: list[dict[str, str]],
    label_rows: list[dict[str, str]],
    parsed_by_id: dict[str, ParseResult],
    per_concept: int,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    meta_by_id = {row["id"]: row for row in metadata_rows}
    samples: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for concept in P0_CONCEPTS:
        candidates = []
        for row in label_rows:
            meta = meta_by_id[row["id"]]
            report = meta["report_2"]
            high_risk_score = 0
            if row["conflict_flag"] == "1":
                high_risk_score += 5
            if row[concept] == UNKNOWN:
                high_risk_score += 3
            if has_any(report, NEGATION_TERMS):
                high_risk_score += 2
            if "左侧" in report and "右侧" in report:
                high_risk_score += 2
            if "斑块" in report and concept in {"left_plaque", "right_plaque", "main_status"}:
                high_risk_score += 2
            if "狭窄" in report and concept == "stenosis":
                high_risk_score += 2
            if "血流" not in report and concept == "flow_normal":
                high_risk_score += 2
            candidates.append((high_risk_score, rng.random(), row))
        candidates.sort(key=lambda x: (-x[0], x[1]))
        chosen = []
        value_buckets = defaultdict(list)
        for _, _, row in candidates:
            value_buckets[row[concept]].append(row)
        for value in ["1", "0", UNKNOWN, "plaque", "wall_abnormal_no_plaque", "normal"]:
            if value_buckets.get(value):
                chosen.append(value_buckets[value][0])
        for _, _, row in candidates:
            if len(chosen) >= per_concept:
                break
            if row not in chosen:
                chosen.append(row)
        for row in chosen[:per_concept]:
            key = (row["id"], concept)
            if key in seen:
                continue
            seen.add(key)
            result = parsed_by_id[row["id"]]
            ev = result.evidence.get(concept)
            samples.append(
                {
                    "id": row["id"],
                    "official_id": row.get("official_id", ""),
                    "split": row["split"],
                    "report_2": meta_by_id[row["id"]]["report_2"],
                    "concept": concept,
                    "predicted_label": row[concept],
                    "matched_clause": ev.clause if ev else "",
                    "matched_rule": ev.rule if ev else "",
                    "review_status": "pending",
                    "review_note": "",
                }
            )
    return samples


def write_parser_version(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True) if path.parent != Path(".") else None
    path.write_text(PARSER_VERSION + "\n", encoding="utf-8")


def print_corpus_audit(metadata_rows: list[dict[str, str]]) -> None:
    reports = [row["report_2"] for row in metadata_rows]
    train_reports = [row["report_2"] for row in metadata_rows if row["split"] == "train"]
    val_reports = [row["report_2"] for row in metadata_rows if row["split"] == "val"]
    print("[Corpus Audit]")
    print(f"metadata rows: {len(metadata_rows)}")
    print(f"columns: {list(metadata_rows[0].keys()) if metadata_rows else []}")
    print(f"id unique: {len({row['id'] for row in metadata_rows}) == len(metadata_rows)}")
    print(f"official_id exists: {'official_id' in metadata_rows[0] if metadata_rows else False}")
    print(f"split counts: {dict(Counter(row['split'] for row in metadata_rows))}")
    print(f"missing report_1: {sum(1 for row in metadata_rows if not row.get('report_1'))}")
    print(f"missing report_2: {sum(1 for row in metadata_rows if not row.get('report_2'))}")
    print(f"unique report_2 overall/train/val: {len(set(reports))}/{len(set(train_reports))}/{len(set(val_reports))}")
    print(
        "common punctuation: "
        + json.dumps(dict(Counter(ch for text in reports for ch in text if ch in '，。；：、,.').most_common(12)), ensure_ascii=False)
    )
    print(
        "common negations: "
        + json.dumps({term: sum(term in text for text in reports) for term in NEGATION_TERMS}, ensure_ascii=False)
    )
    print(
        "common laterality: "
        + json.dumps(
            {
                "双侧": sum("双侧" in text for text in reports),
                "两侧": sum("两侧" in text for text in reports),
                "左侧": sum("左侧" in text for text in reports),
                "右侧": sum("右侧" in text for text in reports),
                "left_and_right": sum("左侧" in text and "右侧" in text for text in reports),
            },
            ensure_ascii=False,
        )
    )
    print(
        "common terms: "
        + json.dumps(
            {
                "毛糙": sum("毛糙" in text for text in reports),
                "增厚": sum("增厚" in text for text in reports),
                "斑块": sum("斑块" in text for text in reports),
                "狭窄": sum("狭窄" in text for text in reports),
                "血流通畅": sum("血流通畅" in text for text in reports),
                "血流充填良好": sum("血流充填良好" in text for text in reports),
                "血流充盈良好": sum("血流充盈良好" in text for text in reports),
            },
            ensure_ascii=False,
        )
    )
    print()


def print_label_summary(label_rows: list[dict[str, str]]) -> None:
    print(f"parser_version: {PARSER_VERSION}")
    for concept in P0_CONCEPTS:
        print(f"{concept}: {dict(Counter(row[concept] for row in label_rows))}")
    print(f"conflicts: {sum(row['conflict_flag'] == '1' for row in label_rows)}")


def main() -> int:
    args = parse_args()
    metadata_rows = load_metadata(Path(args.metadata))
    print_corpus_audit(metadata_rows)

    label_rows, error_rows, parsed_by_id = build_labels(metadata_rows)
    concept_audit_rows = build_concept_audit(metadata_rows, label_rows, parsed_by_id)
    audit_sample_rows = build_parser_audit_samples(
        metadata_rows,
        label_rows,
        parsed_by_id,
        args.audit_samples_per_concept,
        args.seed,
    )

    write_csv(Path(args.output), label_rows, LABEL_COLUMNS)
    write_csv(Path(args.audit_output), concept_audit_rows, CONCEPT_AUDIT_COLUMNS)
    write_csv(Path(args.audit_samples), audit_sample_rows, AUDIT_SAMPLE_COLUMNS)
    write_csv(Path(args.errors_output), error_rows, PARSER_ERROR_COLUMNS)
    write_parser_version(Path(args.version_output))

    print_label_summary(label_rows)
    print(f"labels: {args.output}")
    print(f"concept_audit: {args.audit_output}")
    print(f"audit_samples: {args.audit_samples}")
    print(f"parser_errors: {args.errors_output}")
    print(f"parser_version_file: {args.version_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
