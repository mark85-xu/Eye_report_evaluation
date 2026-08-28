"""Build the official-format ``res.csv`` submission from raw model predictions.

Mapping rule (matches the frozen metadata schema):

    metadata.id        -> internal join key  (always used inside the project)
    metadata.official_id -> final submission ``image_id`` (official JSON ``id``)

The formatter refuses to guess: if the internal->official mapping cannot be
resolved uniquely, it raises instead of silently writing wrong ids.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from src.evaluation.tokenizer import tokenize_report

SUBMISSION_COLUMNS = ["image_id", "predicted_report"]


def load_predictions(
    path: str | Path,
    id_column: str = "id",
    report_column: str = "predicted_report",
) -> pd.DataFrame:
    """Read a raw prediction file, keeping only id + report columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"prediction file not found: {path}")
    df = pd.read_csv(path)
    if id_column not in df.columns or report_column not in df.columns:
        raise ValueError(
            f"prediction file must have columns {id_column!r} and {report_column!r}; "
            f"got {list(df.columns)}"
        )
    df = df[[id_column, report_column]].copy()
    df = df.rename(columns={id_column: "id", report_column: "predicted_report"})
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(str).str.strip()
    return df


def build_id_mapping(metadata_path: str | Path) -> dict[str, str]:
    """Map internal ``metadata.id`` -> official ``metadata.official_id``.

    Raises if the mapping is not deterministic (duplicate source ids).
    """
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"metadata not found: {path}")
    meta = pd.read_csv(path)
    for col in ("id", "official_id"):
        if col not in meta.columns:
            raise ValueError(f"metadata must have column {col!r}; got {list(meta.columns)}")
    if meta["id"].duplicated().any():
        dups = meta.loc[meta["id"].duplicated(), "id"].tolist()[:5]
        raise ValueError(f"metadata.id must be unique for mapping; duplicates: {dups}")
    mapping = dict(zip(meta["id"].astype(str), meta["official_id"].astype(str)))
    return mapping


def resolve_official_ids(
    internal_ids: Sequence[str], mapping: dict[str, str]
) -> list[str]:
    missing = sorted(set(internal_ids) - set(mapping))
    if missing:
        raise ValueError(
            f"{len(missing)} internal ids have no official_id mapping "
            f"(first 10): {missing[:10]}"
        )
    return [mapping[i] for i in internal_ids]


def make_submission(
    predictions: pd.DataFrame,
    metadata: str | Path,
    medical_dict: str | Path | None = None,
    already_tokenized: bool = False,
) -> pd.DataFrame:
    """Produce the final res.csv DataFrame.

    ``predictions`` must contain columns ``id`` (internal) and
    ``predicted_report`` (raw report text).
    """
    from src.evaluation.tokenizer import load_medical_dict

    if medical_dict is not None:
        load_medical_dict(medical_dict)
    else:
        load_medical_dict()

    mapping = build_id_mapping(metadata)
    official_ids = resolve_official_ids(predictions["id"].tolist(), mapping)

    reports = [
        tokenize_report(r, already_tokenized=already_tokenized)
        for r in predictions["predicted_report"].tolist()
    ]

    return pd.DataFrame({"image_id": official_ids, "predicted_report": reports})


def save_submission(df: pd.DataFrame, output: str | Path) -> Path:
    """Write res.csv in the official GBK format (header, index=False)."""
    if list(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(
            f"submission must have exactly columns {SUBMISSION_COLUMNS}; got {list(df.columns)}"
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, header=True, encoding="gbk")
    return output
