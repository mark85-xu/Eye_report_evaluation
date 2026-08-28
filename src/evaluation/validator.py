"""P0 submission validator for ``res.csv``.

Catches every common fatal error before a submission is uploaded:
missing / extra / duplicate / empty ids, empty or untokenized reports,
wrong columns, broken GBK encoding, and GBK round-trip integrity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from src.evaluation.formatter import SUBMISSION_COLUMNS

_TOKENIZATION_BREAK_MARKER = " "


def _read_gbk(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="gbk", dtype=str)


def _check_ids(series: pd.Series) -> list[str]:
    """Return a list of id-level problems."""
    problems: list[str] = []
    if series.isna().any():
        problems.append(f"image_id contains {int(series.isna().sum())} NaN value(s)")
    empty = series.fillna("").astype(str).str.strip().eq("")
    if empty.any():
        problems.append(f"image_id contains {int(empty.sum())} empty string(s)")
    dups = series[series.astype(str).duplicated()]
    if not dups.empty:
        problems.append(
            f"image_id contains {len(dups)} duplicate value(s), "
            f"e.g. {dups.tolist()[:5]}"
        )
    return problems


def _check_reports(series: pd.Series) -> list[str]:
    problems: list[str] = []
    if series.isna().any():
        problems.append(f"predicted_report contains {int(series.isna().sum())} NaN value(s)")
    filled = series.fillna("").astype(str).str.strip()
    empty = filled.eq("")
    if empty.any():
        problems.append(f"predicted_report contains {int(empty.sum())} empty string(s)")
    return problems


def _check_tokenization(series: pd.Series, sample_size: int = 50) -> list[str]:
    """Sanity check only: a tokenized report is space-joined and multi-token.

    Deliberately lenient so short normal reports are not falsely rejected.
    """
    problems: list[str] = []
    filled = series.fillna("").astype(str).str.strip()
    n = min(sample_size, len(filled))
    for value in filled.head(n):
        if not value:
            continue
        if _TOKENIZATION_BREAK_MARKER not in value:
            problems.append(
                f"predicted_report may be untokenized (no spaces): {value[:40]}..."
            )
            break
    return problems


def validate_submission(
    res_csv: str | Path,
    expected_ids: Iterable[str] | None = None,
    verbose: bool = True,
) -> dict:
    """Validate a res.csv and return a detailed check report.

    Returns a dict with ``valid`` plus per-check booleans and details.
    """
    path = Path(res_csv)
    report: dict = {
        "file_exists": False,
        "gbk_readable": False,
        "columns": False,
        "id_valid": False,
        "id_set_match": False,
        "report_non_empty": False,
        "tokenization": False,
        "gbk_roundtrip": False,
        "missing_ids": [],
        "extra_ids": [],
        "duplicate_ids": [],
        "empty_reports": [],
        "problems": [],
        "valid": False,
        "expected_count": None,
        "actual_count": None,
    }

    if not path.exists():
        report["problems"].append(f"file not found: {path}")
        _print_report(report) if verbose else None
        return report
    report["file_exists"] = True

    try:
        df = _read_gbk(path)
    except Exception as exc:  # noqa: BLE001 - any read failure is a validator failure
        report["problems"].append(f"failed to read as GBK: {exc}")
        _print_report(report) if verbose else None
        return report
    report["gbk_readable"] = True

    if list(df.columns) == SUBMISSION_COLUMNS:
        report["columns"] = True
    else:
        report["problems"].append(
            f"columns must be exactly {SUBMISSION_COLUMNS}; got {list(df.columns)}"
        )

    # Normalise ids to plain strings: pandas may keep an empty cell as a real
    # float NaN even with dtype=str (StringDtype), which must not leak into
    # set comparison / sorting.
    report["actual_count"] = len(df)
    has_id = "image_id" in df.columns
    has_report = "predicted_report" in df.columns

    if has_id:
        ids_str = df["image_id"].fillna("").astype(str)
        id_problems = _check_ids(df["image_id"])
        report["duplicate_ids"] = sorted(
            ids_str[ids_str.duplicated()].tolist()
        )
        if not id_problems:
            report["id_valid"] = True
        else:
            report["problems"].extend(id_problems)

        if expected_ids is not None:
            expected = list(expected_ids)
            report["expected_count"] = len(expected)
            actual_set = set(ids_str)
            expected_set = set(expected)
            report["missing_ids"] = sorted(expected_set - actual_set)
            report["extra_ids"] = sorted(actual_set - expected_set)
            if not report["missing_ids"] and not report["extra_ids"]:
                report["id_set_match"] = True
            if report["missing_ids"]:
                report["problems"].append(
                    f"missing ids: {len(report['missing_ids'])} "
                    f"(first 10): {report['missing_ids'][:10]}"
                )
            if report["extra_ids"]:
                report["problems"].append(
                    f"extra ids: {len(report['extra_ids'])} (first 10): {report['extra_ids'][:10]}"
                )

    if has_report:
        report_problems = _check_reports(df["predicted_report"])
        empty_mask = df["predicted_report"].fillna("").astype(str).str.strip().eq("")
        report["empty_reports"] = df.loc[empty_mask, "image_id"].tolist() if has_id else empty_mask.index.tolist()
        if not report_problems:
            report["report_non_empty"] = True
        else:
            report["problems"].extend(report_problems)

        tok_problems = _check_tokenization(df["predicted_report"])
        if not tok_problems:
            report["tokenization"] = True
        else:
            report["problems"].extend(tok_problems)

    try:
        _read_gbk(path)
        report["gbk_roundtrip"] = True
    except Exception as exc:  # noqa: BLE001
        report["problems"].append(f"GBK round-trip failed: {exc}")

    required = [
        "file_exists",
        "gbk_readable",
        "columns",
        "id_valid",
        "id_set_match",
        "report_non_empty",
        "gbk_roundtrip",
    ]
    report["valid"] = all(report[k] for k in required) and not report["problems"]

    if verbose:
        _print_report(report)
    return report


def _print_report(report: dict) -> None:
    n_exp = report.get("expected_count")
    n_act = report.get("actual_count")
    exp_str = str(n_exp) if n_exp is not None else "N/A"
    act_str = str(n_act) if n_act is not None else "N/A"
    print("========== Submission Validation ==========")
    print(f"Expected samples : {exp_str}")
    print(f"Actual samples   : {act_str}")
    print(f"Missing IDs      : {len(report['missing_ids'])}")
    print(f"Extra IDs        : {len(report['extra_ids'])}")
    print(f"Duplicate IDs    : {len(report['duplicate_ids'])}")
    print(f"Empty reports    : {len(report['empty_reports'])}")
    for check in [
        "file_exists",
        "gbk_readable",
        "columns",
        "id_valid",
        "id_set_match",
        "report_non_empty",
        "tokenization",
        "gbk_roundtrip",
    ]:
        label = {
            "file_exists": "File exists",
            "gbk_readable": "GBK encoding",
            "columns": "Columns",
            "id_valid": "IDs",
            "id_set_match": "ID set match",
            "report_non_empty": "Reports non-empty",
            "tokenization": "Tokenization",
            "gbk_roundtrip": "GBK round-trip",
        }[check]
        status = "PASS" if report.get(check) else "FAIL"
        print(f"{label:<20}: {status}")
    if report["valid"]:
        print("\nSubmission       : VALID")
    else:
        print("\nSubmission       : FAIL")
        for problem in report["problems"]:
            print(f"  - {problem}")
    print("============================================")
