#!/usr/bin/env python
"""Proxy evaluation of model predictions against Val report_2 ground truth.

Only report_2 is used as GT (report_1 never enters text scoring).

Usage:
    python scripts/evaluate_reports.py \
        --pred outputs/evaluation/val_preds_raw.csv \
        --metadata data/metadata.csv \
        --split val \
        --medical-dict 赛题资料及数据集/医学字典/medical_dict_final.txt \
        --output-dir outputs/evaluation/val_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.evaluation.clinical_metric import evaluate_clinical
from src.evaluation.tokenizer import load_medical_dict, tokenize_report
from src.evaluation.text_metrics import evaluate_text_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proxy evaluate Val predictions against report_2 GT (NOT official)."
    )
    parser.add_argument(
        "--pred", required=True, help="CSV with internal id + raw predicted_report"
    )
    parser.add_argument(
        "--pred-id-col", default="id", help="prediction id column (internal metadata.id)"
    )
    parser.add_argument(
        "--pred-report-col", default="predicted_report",
        help="prediction report column (raw text by default)",
    )
    parser.add_argument(
        "--metadata", default="data/metadata.csv",
        help="metadata.csv; used to source report_2 GT and restrict split",
    )
    parser.add_argument(
        "--gt-csv", default=None,
        help="optional explicit GT csv with columns [id, report_2]; overrides --metadata GT",
    )
    parser.add_argument(
        "--split", default="val", help="metadata split to evaluate (default val)"
    )
    parser.add_argument(
        "--medical-dict", default="赛题资料及数据集/医学字典/medical_dict_final.txt"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--pred-already-tokenized", action="store_true",
        help="predicted_report is already space-tokenized; do not run jieba again",
    )
    parser.add_argument(
        "--no-clinical", action="store_true", help="skip the P1 clinical metric"
    )
    parser.add_argument(
        "--seed", type=int, default=20260828, help="seeds none; kept for reproducibility"
    )
    return parser.parse_args()


def load_gt(metadata_path: str, split: str, gt_csv: str | None) -> pd.DataFrame:
    if gt_csv is not None:
        gt = pd.read_csv(gt_csv)
        for col in ("id", "report_2"):
            if col not in gt.columns:
                raise ValueError(f"gt_csv must have column {col!r}; got {list(gt.columns)}")
        return gt[["id", "report_2"]].copy()
    meta = pd.read_csv(metadata_path)
    for col in ("id", "split", "report_2"):
        if col not in meta.columns:
            raise ValueError(f"metadata must have column {col!r}; got {list(meta.columns)}")
    gt = meta.loc[meta["split"] == split, ["id", "report_2"]].copy()
    if gt.empty:
        raise ValueError(f"no rows with split={split!r} in {metadata_path}")
    return gt


def main() -> int:
    args = parse_args()

    load_medical_dict(args.medical_dict)

    gt = load_gt(args.metadata, args.split, args.gt_csv)
    pred = pd.read_csv(args.pred)
    for col in (args.pred_id_col, args.pred_report_col):
        if col not in pred.columns:
            raise ValueError(
                f"pred file must have column {col!r}; got {list(pred.columns)}"
            )
    pred = pred[[args.pred_id_col, args.pred_report_col]].copy()
    pred = pred.rename(
        columns={args.pred_id_col: "id", args.pred_report_col: "predicted_report"}
    )

    joined = gt.merge(pred, on="id", how="inner")
    missing_in_pred = set(gt["id"]) - set(pred["id"])
    extra_in_pred = set(pred["id"]) - set(gt["id"])
    if len(joined) != len(gt):
        print(
            f"[warn] join mismatch: GT={len(gt)}, pred={len(pred)}, "
            f"joined={len(joined)}; missing_in_pred={len(missing_in_pred)}, "
            f"extra_in_pred={len(extra_in_pred)}"
        )

    ids = joined["id"].tolist()
    raw_gt = joined["report_2"].tolist()
    raw_pred = joined["predicted_report"].tolist()

    gt_tok = [tokenize_report(r) for r in raw_gt]
    pred_tok = [
        tokenize_report(r, already_tokenized=args.pred_already_tokenized)
        for r in raw_pred
    ]

    metrics = evaluate_text_metrics(
        pred_tok,
        [[r] for r in gt_tok],
        ids=ids,
        tokenizer_label="jieba + medical_dict_final.txt",
        parser_version="v0.1.1",
        n_samples=len(ids),
    )

    per_sample = pd.DataFrame(metrics["per_sample"])
    per_sample = per_sample.merge(
        pd.DataFrame({
            "id": ids,
            "gt_raw_report": raw_gt,
            "pred_raw_report": raw_pred,
            "gt_tokenized_report": gt_tok,
            "pred_tokenized_report": pred_tok,
        }),
        on="id",
        how="left",
    )
    per_sample_cols = [
        "id",
        "gt_raw_report",
        "pred_raw_report",
        "gt_tokenized_report",
        "pred_tokenized_report",
        "BLEU_1",
        "BLEU_2",
        "BLEU_3",
        "BLEU_4",
        "ROUGE_L",
        "METEOR",
    ]
    per_sample = per_sample[per_sample_cols]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_summary.json").write_text(
        json.dumps(metrics["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    per_sample.to_csv(
        out_dir / "per_sample_metrics.csv", index=False, encoding="utf-8"
    )

    print("========== Proxy Evaluation ==========")
    print(f"N              : {metrics['summary']['n_samples']}")
    for key in ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4", "ROUGE_L", "METEOR"]:
        marker = " (PRIMARY)" if key == "BLEU_4" else ""
        val = metrics['summary'].get(key)
        if isinstance(val, float):
            val = f"{val:.6f}"
        print(f"{key:<15}: {val}{marker}")
    print(f"METEOR status  : {metrics['summary'].get('meteor_status')}")
    print("NOTE: PROXY evaluation, NOT official testOffLine score.")
    print("=======================================")

    if args.no_clinical:
        return 0

    clinical = evaluate_clinical(
        raw_gt,
        raw_pred,
        ids=ids,
        gt_reports_tokenized=gt_tok,
        pred_reports_tokenized=pred_tok,
        bleu4_per_sample=dict(zip(ids, per_sample["BLEU_4"])),
    )
    (out_dir / "clinical_summary.json").write_text(
        json.dumps(clinical.summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    clinical.per_sample.to_csv(
        out_dir / "clinical_per_sample.csv", index=False, encoding="utf-8"
    )
    print("Clinical metric (P1, NOT official) written to clinical_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
