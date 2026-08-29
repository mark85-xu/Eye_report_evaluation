"""pycocoevalcap-based BLEU-1/2/3/4, ROUGE_L and (optional) METEOR scoring.

The official scorer uses pycocoevalcap; BLEU-4 is the PRIMARY metric for the
competition.  This module feeds pycocoevalcap the *space-tokenized* strings
produced by :mod:`src.evaluation.tokenizer` (never raw Chinese, and never
double-tokenized input).

Interface notes (verified against the installed pycocoevalcap 1.2 source):

* ``gts`` / ``res`` are dicts of ``id -> list[str]`` (res[id] has length 1).
* ``Bleu(n).compute_score`` returns ``(corpus_bleus[4], per_sentence)`` where
  ``per_sentence[k]`` is the list of per-sentence BLEU-(k+1).  The corpus
  BLEU is NOT the mean of the per-sentence scores (different aggregation),
  so per-sample BLEU is used only for error analysis.
* ``Rouge().compute_score`` returns ``(corpus_mean, np.array per_sentence)``.
* ``Meteor().compute_score`` returns ``(corpus_mean, list per_sentence)``.

METEOR is optional: it shells out to a Java process (the pycocoevalcap
meteor jar + paraphrase data are bundled; only a ``java`` binary on PATH is
required).  When Java is unavailable the module reports ``meteor_status:
skipped`` instead of failing the whole BLEU + ROUGE pipeline.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.evaluation.tokenizer import tokenize_report

PRIMARY_METRIC = "BLEU_4"
_CORPUS_KEYS = ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]


def _build_coco_dicts(
    ids: Sequence[Any],
    preds: Sequence[str],
    refs: Sequence[Sequence[str]],
) -> tuple[dict, dict]:
    gts: dict = {}
    res: dict = {}
    for i, pred, refs_i in zip(ids, preds, refs):
        gts[i] = list(refs_i)
        res[i] = [pred]
    return gts, res


def _extract_bleu(score, per_sentence, ids: Sequence[Any]) -> dict:
    """Normalise Bleu.compute_score output into corpus + per-sample dicts.

    ``score`` is the corpus [Bleu_1..Bleu_4]; ``per_sentence`` is a list of
    four lists (one per n), each aligned to the gts-key iteration order.
    """
    corpus = list(np.asarray(score, dtype=float).reshape(-1))
    if len(corpus) < 4:
        corpus = [None] * 4
    corpus_map = dict(zip(_CORPUS_KEYS, corpus[:4]))

    per_sample = {}
    n_sent = len(ids)
    for j, i in enumerate(ids):
        values = []
        for k in range(4):
            arr = per_sentence[k]
            values.append(float(arr[j]) if j < len(arr) else None)
        per_sample[i] = dict(zip(_CORPUS_KEYS, values))
    return {"corpus": corpus_map, "per_sample": per_sample}


def _extract_rouge(score, per_sentence, ids: Sequence[Any]) -> dict:
    """Normalise Rouge.compute_score output: (corpus float, ndarray)."""
    corpus = float(np.asarray(score, dtype=float))
    arr = np.asarray(per_sentence, dtype=float).reshape(-1)
    per_sample = {}
    for j, i in enumerate(ids):
        per_sample[i] = float(arr[j]) if j < len(arr) else None
    return {"corpus": {"ROUGE_L": corpus}, "per_sample": per_sample}


def evaluate_text_metrics(
    preds: Sequence[str],
    refs: Sequence[Sequence[str]],
    ids: Sequence[Any] | None = None,
    tokenizer_label: str = "jieba + medical_dict_final.txt",
    parser_version: str = "v0.1.1",
    n_samples: int | None = None,
    enable_meteor: bool = True,
    already_tokenized: bool = True,
) -> dict:
    """Score already-tokenized predictions/references via pycocoevalcap.

    ``preds`` / ``refs`` are space-joined token strings (one per sample;
    refs is a list of lists so multiple references are supported).  Pass
    ``already_tokenized=False`` only if the caller supplies raw reports that
    must first go through the project tokenizer.
    """
    if ids is None:
        ids = list(range(len(preds)))
    if already_tokenized is False:
        preds = [tokenize_report(p) for p in preds]
        refs = [[tokenize_report(r) for r in refs_i] for refs_i in refs]
    if len(preds) != len(refs) or len(preds) != len(ids):
        raise ValueError("preds, refs and ids must have the same length")

    gts, res = _build_coco_dicts(ids, preds, refs)

    # ---- BLEU ----
    from pycocoevalcap.bleu.bleu import Bleu

    bleu_scorer = Bleu(4)
    bleu_score, bleu_sent = bleu_scorer.compute_score(gts, res, verbose=0)
    bleu = _extract_bleu(bleu_score, bleu_sent, ids)

    # ---- ROUGE_L ----
    from pycocoevalcap.rouge.rouge import Rouge

    rouge_scorer = Rouge()
    rouge_score, rouge_sent = rouge_scorer.compute_score(gts, res)
    rouge = _extract_rouge(rouge_score, rouge_sent, ids)

    # ---- METEOR (optional) ----
    meteor_corpus = None
    meteor_per_sample: dict = {}
    meteor_status = "skipped"
    meteor_reason = None
    if enable_meteor:
        try:
            # When Java is missing, Meteor() fails mid-construction and its
            # __del__ then raises a cosmetic AttributeError at GC time.  Patch
            # it locally so skipped-METEOR runs stay clean (package untouched).
            from pycocoevalcap.meteor.meteor import Meteor

            if getattr(Meteor, "__del__", None):
                _orig_del = Meteor.__del__

                def _safe_del(self, _orig=_orig_del):
                    try:
                        _orig(self)
                    except Exception:  # noqa: BLE001
                        pass

                Meteor.__del__ = _safe_del

            meteor_scorer = Meteor()
            meteor_corpus, meteor_sent = meteor_scorer.compute_score(gts, res)
            meteor_corpus = float(meteor_corpus)
            for j, i in enumerate(ids):
                meteor_per_sample[i] = (
                    float(meteor_sent[j]) if j < len(meteor_sent) else None
                )
            meteor_status = "ok"
        except Exception as exc:  # noqa: BLE001 - never let METEOR block the pipeline
            meteor_status = "skipped"
            meteor_reason = f"{type(exc).__name__}: {exc}"

    per_sample_rows = []
    for i in ids:
        row = {"id": i}
        row.update(bleu["per_sample"][i])
        row["ROUGE_L"] = rouge["per_sample"].get(i)
        row["METEOR"] = meteor_per_sample.get(i)
        per_sample_rows.append(row)

    summary: dict[str, Any] = {
        "evaluation_type": "proxy",
        "n_samples": n_samples if n_samples is not None else len(ids),
        "primary_metric": PRIMARY_METRIC,
        **bleu["corpus"],
        "ROUGE_L": rouge["corpus"]["ROUGE_L"],
        "METEOR": meteor_corpus,
        "meteor_status": meteor_status,
        "meteor_reason": meteor_reason,
        "tokenizer": tokenizer_label,
        "parser_version": parser_version,
        "note": "NOT OFFICIAL TESTOFFLINE SCORE",
    }

    return {"summary": summary, "per_sample": per_sample_rows}


def per_sample_frame(metrics: dict) -> "pd.DataFrame":
    """Return the per-sample metrics list as a DataFrame (order preserved)."""
    import pandas as pd

    return pd.DataFrame(metrics["per_sample"])
