#!/usr/bin/env python
"""Build the official-format ``res.csv`` from raw model predictions.

Input predictions use the internal ``metadata.id`` as their key; the output
``res.csv`` carries the official ``image_id`` (= metadata.official_id) and a
jieba-tokenized ``predicted_report``, saved as GBK exactly like the official
submit path:

    df_res.to_csv("res.csv", index=False, header=True, encoding="gbk")

Usage:
    python scripts/make_submission.py \
        --pred outputs/evaluation/val_preds_raw.csv \
        --metadata data/metadata.csv \
        --medical-dict 赛题资料及数据集/医学字典/medical_dict_final.txt \
        --output outputs/evaluation/res_val.csv \
        --validate --expected-split val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.evaluation.formatter import (
    SUBMISSION_COLUMNS,
    load_predictions,
    make_submission,
    save_submission,
)
from src.evaluation.validator import validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build res.csv from raw predictions.")
    parser.add_argument("--pred", required=True, help="CSV with internal id + raw report")
    parser.add_argument(
        "--pred-id-col", default="id", help="prediction id column (internal metadata.id)"
    )
    parser.add_argument(
        "--pred-report-col", default="predicted_report",
        help="prediction report column (raw text by default)",
    )
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument(
        "--medical-dict", default="赛题资料及数据集/医学字典/medical_dict_final.txt"
    )
    parser.add_argument("--output", required=True, help="output res.csv path")
    parser.add_argument(
        "--already-tokenized", action="store_true",
        help="predicted_report is already space-tokenized; do not run jieba again",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="run the P0 submission validator after writing",
    )
    parser.add_argument(
        "--expected-split", default="test",
        help="metadata split used for expected ids when validating (default test; "
             "falls back to val when no test rows exist)",
    )
    return parser.parse_args()


def expected_ids_from_metadata(metadata: str, split: str) -> list[str]:
    meta = pd.read_csv(metadata)
    if split in meta["split"].values:
        ids = meta.loc[meta["split"] == split, "official_id"].astype(str).tolist()
    else:
        ids = meta.loc[meta["split"] == "val", "official_id"].astype(str).tolist()
    return ids


def main() -> int:
    args = parse_args()

    pred = load_predictions(args.pred, id_column=args.pred_id_col,
                            report_column=args.pred_report_col)
    sub = make_submission(
        pred,
        metadata=args.metadata,
        medical_dict=args.medical_dict,
        already_tokenized=args.already_tokenized,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_submission(sub, out)
    print(f"wrote {out} ({len(sub)} rows)")

    if args.validate:
        expected = expected_ids_from_metadata(args.metadata, args.expected_split)
        validate_submission(out, expected_ids=expected)

    return 0


if __name__ == "__main__":
    sys.exit(main())
