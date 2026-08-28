"""Unified report tokenizer for the proxy evaluation pipeline.

The only official-compliant tokenization entry used across the project:

    raw Chinese report
        -> normalize whitespace / handle NaN / None / empty
        -> jieba with medical_dict_final.txt loaded as user dictionary
        -> tokens joined by " "

Both GT (report_2) and model predictions MUST go through this same
tokenizer before being scored by pycocoevalcap.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import jieba

DEFAULT_MEDICAL_DICT = (
    Path(__file__).resolve().parents[2]
    / "赛题资料及数据集"
    / "医学字典"
    / "medical_dict_final.txt"
)

# Observed on the official gts.csv sample: the ASCII token ``CDFI`` is dropped
# while the following full-width ``：`` is kept (``CDFI：显示血流充填良好`` ->
# ``： 显示 血流 充填 良好``).  ~65% of report_2 contain CDFI, so keeping it
# would systematically lower BLEU against the official GT.  This is a proxy
# decision to be re-verified against testOffLine.py.
CDFI_RE = re.compile(r"CDFI", re.IGNORECASE)

_loaded_path: str | None = None


def clean_report_text(value: Any) -> str:
    """Return a clean, plain-whitespace report string (no segmentation).

    Handles None / NaN / non-str / leading-trailing spaces / consecutive
    spaces / newlines / CRLF.  Never alters medical content and does NOT
    normalise full-width punctuation: the official gts keeps ``，`` ``：``
    ``。`` as tokens, so those must survive tokenization unchanged.
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if not isinstance(value, str):
        return ""
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_medical_dict(path: str | Path | None = None) -> str:
    """Load medical_dict_final.txt into jieba as the user dictionary.

    The official dict mixes two separators:
      * most lines:  ``word<tab>freq``        e.g. ``精神\t606946``
      * a few lines: ``word freq tag``        e.g. ``囊性 99999 n``

    ``jieba.load_userdict`` splits on a single space and would therefore
    swallow the tab lines, so the file is parsed here and applied with
    ``jieba.add_word`` (freq kept when present).  Idempotent: loading twice
    re-applies the same words.
    """
    global _loaded_path
    if path is None:
        path = DEFAULT_MEDICAL_DICT
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"medical dict not found: {path}")

    if _loaded_path == str(path):
        return str(path)

    n_words = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if not parts:
            continue
        word = parts[0]
        freq = None
        if len(parts) >= 2:
            try:
                freq = int(parts[1])
            except ValueError:
                freq = None
        jieba.add_word(word, freq)
        n_words += 1

    jieba.initialize()
    _loaded_path = str(path)
    return str(path)


def tokenize_report(report: Any, already_tokenized: bool = False) -> str:
    """Tokenize one report and return a space-joined string.

    ``already_tokenized=True`` skips jieba entirely: the input is only
    whitespace-normalized (so ``"双侧 颈动脉"`` never gets re-segmented).
    Never apply jieba twice to the same text.
    """
    if already_tokenized:
        return clean_report_text(report)

    text = clean_report_text(report)
    if not text:
        return ""
    text = CDFI_RE.sub("", text)
    # Chinese reports contain no meaningful spaces; any whitespace is just
    # line-break formatting.  Remove it so jieba never binds a space to a
    # punctuation token (e.g. ``， ``), which would corrupt n-grams.
    text = re.sub(r"\s+", "", text)
    load_medical_dict()  # idempotent after first call
    tokens = jieba.lcut(text)
    return " ".join(tokens)


def tokenize_reports(
    reports: Iterable[Any], already_tokenized: bool = False
) -> list[str]:
    """Tokenize a batch of reports (same rules as :func:`tokenize_report`)."""
    return [tokenize_report(r, already_tokenized=already_tokenized) for r in reports]
