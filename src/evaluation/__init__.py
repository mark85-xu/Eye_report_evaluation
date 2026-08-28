"""Proxy Evaluation V0.1 for 赛题3 (carotid ultrasound report generation).

This is a PROXY evaluator used for relative Val comparison and for producing
a valid ``res.csv`` submission.  It is NOT the official ``testOffLine.py``;
alignment with the official offline scorer happens once the official file is
released.  See ``src/evaluation/README.md``.
"""

from .tokenizer import (
    clean_report_text,
    load_medical_dict,
    tokenize_report,
    tokenize_reports,
)

__version__ = "0.1.0"
