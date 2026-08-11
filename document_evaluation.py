"""Metrics for comparing extracted document text with a reference transcript."""

import re
from collections import Counter


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def extraction_accuracy(reference: str, extracted: str) -> dict[str, float]:
    """Return token precision, recall, and F1 against a labelled transcript."""
    expected = Counter(_tokens(reference))
    actual = Counter(_tokens(extracted))
    matches = sum((expected & actual).values())
    precision = matches / sum(actual.values()) if actual else 0.0
    recall = matches / sum(expected.values()) if expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "reference_tokens": float(sum(expected.values())),
        "extracted_tokens": float(sum(actual.values())),
    }


def compare_pdf_extraction(
    reference: str, normal_pdf_text: str, ocr_text: str
) -> dict[str, dict[str, float]]:
    """Compare normal PDF extraction and OCR against the same ground truth."""
    return {
        "normal_pdf": extraction_accuracy(reference, normal_pdf_text),
        "ocr": extraction_accuracy(reference, ocr_text),
    }
