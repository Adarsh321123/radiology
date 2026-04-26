"""Unit checks for score-aligned label masking and loss reduction.

Run from repo root:

    uv run python tests/test_score_aligned_objective.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset import _labels_to_array
from train import masked_bce_with_logits


def assert_close(a: float, b: float, tol: float = 1e-6) -> None:
    if abs(a - b) > tol:
        raise AssertionError(f"{a} != {b}")


def test_legacy_label_mapping() -> None:
    df = pd.DataFrame({
        "A": [1.0, 0.0, -1.0, np.nan],
    })
    y = _labels_to_array(df, ["A"], mode="u_ones")
    expected = np.array([[1.0], [1.0], [0.0], [0.0]], dtype=np.float32)
    if not np.array_equal(y, expected):
        raise AssertionError(f"legacy mapping changed: {y} vs {expected}")


def test_score_aligned_label_mapping() -> None:
    df = pd.DataFrame({
        "A": [1.0, 0.0, -1.0, np.nan],
        "B": [np.nan, -1.0, 0.0, 1.0],
    })
    y_train = _labels_to_array(
        df, ["A", "B"],
        mode="per_label",
        default_uncertain_strategy="ignore",
        blank_strategy="ignore",
    )
    expected_train = np.array([
        [1.0, np.nan],
        [np.nan, 0.0],
        [0.0, np.nan],
        [np.nan, 1.0],
    ], dtype=np.float32)
    if not np.array_equal(np.isnan(y_train), np.isnan(expected_train)):
        raise AssertionError(f"NaN mask mismatch: {y_train} vs {expected_train}")
    if not np.array_equal(np.nan_to_num(y_train), np.nan_to_num(expected_train)):
        raise AssertionError(f"value mismatch: {y_train} vs {expected_train}")

    y_val = _labels_to_array(
        df, ["A", "B"],
        mode="val",
        blank_strategy="ignore",
    )
    if not np.array_equal(np.isnan(y_val), np.isnan(expected_train)):
        raise AssertionError(f"val NaN mask mismatch: {y_val} vs {expected_train}")


def test_masked_bce_micro_and_macro() -> None:
    logits = torch.tensor([
        [0.0, 10.0],
        [0.0, -10.0],
    ])
    targets = torch.tensor([
        [1.0, float("nan")],
        [0.0, 1.0],
    ])

    elem = F.binary_cross_entropy_with_logits(
        logits, targets.nan_to_num(0.0), reduction="none",
    )
    valid = ~torch.isnan(targets)
    expected_micro = (elem * valid).sum() / valid.sum()

    loss_label0 = elem[:, 0].mean()
    loss_label1 = elem[1, 1]
    expected_macro = (loss_label0 + loss_label1) / 2.0

    got_micro = masked_bce_with_logits(logits, targets, reduction="micro")
    got_macro = masked_bce_with_logits(logits, targets, reduction="macro")

    assert_close(float(got_micro), float(expected_micro))
    assert_close(float(got_macro), float(expected_macro))
    if math.isclose(float(got_micro), float(got_macro)):
        raise AssertionError("micro and macro reductions should differ on this fixture")


def main() -> None:
    test_legacy_label_mapping()
    test_score_aligned_label_mapping()
    test_masked_bce_micro_and_macro()
    print("score-aligned objective tests OK")


if __name__ == "__main__":
    main()
