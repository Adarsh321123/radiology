"""Pre-submission sanity checks. Run on every CSV before uploading.

Usage:
    uv run python sanity_check.py submission.csv
    uv run python sanity_check.py submission.csv --reference old_good_submission.csv

Checks:
  1. Per-label mean prediction vs training prevalence (catches v5 disaster)
  2. Per-label prediction distribution (catches TTA collapse toward 0.5)
  3. Single-label blowup detection (catches 04-05 and v5 disasters)
  4. Comparison against a reference CSV (catches regressions)
  5. Basic format validation (row count, column names, value ranges)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_NAMES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Pneumonia",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

TRAIN_PREVALENCE = {
    "No Finding": 0.106,
    "Enlarged Cardiomediastinum": 0.042,
    "Cardiomegaly": 0.113,
    "Lung Opacity": 0.473,
    "Pneumonia": 0.015,
    "Pleural Effusion": 0.385,
    "Pleural Other": 0.017,
    "Fracture": 0.039,
    "Support Devices": 0.527,
}

def check_format(df: pd.DataFrame, test_ids_csv: Path | None = None) -> list[str]:
    errors = []
    if "Id" not in df.columns:
        errors.append("FAIL: missing 'Id' column")
        return errors
    for label in LABEL_NAMES:
        if label not in df.columns:
            errors.append(f"FAIL: missing column '{label}'")
    if len(df) == 0:
        errors.append("FAIL: CSV has 0 rows")
    for label in LABEL_NAMES:
        if label not in df.columns:
            continue
        vals = df[label]
        if vals.isna().any():
            errors.append(f"FAIL: {label} has {vals.isna().sum()} NaN values")
        if (vals < 0).any() or (vals > 1).any():
            errors.append(f"FAIL: {label} has values outside [0, 1]")
    # ID checks
    if df["Id"].duplicated().any():
        n_dup = df["Id"].duplicated().sum()
        errors.append(f"FAIL: {n_dup} duplicate Id(s)")
    if test_ids_csv is not None and test_ids_csv.exists():
        ref_ids = set(pd.read_csv(test_ids_csv)["Id"].tolist())
        sub_ids = set(df["Id"].tolist())
        missing = ref_ids - sub_ids
        extra = sub_ids - ref_ids
        if missing:
            errors.append(f"FAIL: {len(missing)} IDs in test_ids.csv but missing from submission")
        if extra:
            errors.append(f"FAIL: {len(extra)} IDs in submission but not in test_ids.csv")
        if not missing and not extra and len(df) != len(ref_ids):
            errors.append(f"FAIL: row count {len(df)} != expected {len(ref_ids)}")
    return errors


def check_prevalence(df: pd.DataFrame) -> list[str]:
    """Flag if mean prediction is wildly different from training prevalence."""
    warnings = []
    for label in LABEL_NAMES:
        if label not in df.columns:
            continue
        mean_pred = df[label].mean()
        prev = TRAIN_PREVALENCE[label]
        ratio = mean_pred / prev if prev > 0 else float("inf")

        if ratio > 3.0:
            warnings.append(
                f"FAIL: {label} mean prediction {mean_pred:.3f} is {ratio:.1f}x "
                f"training prevalence {prev:.3f} — likely inflated"
            )
        elif ratio > 2.0:
            warnings.append(
                f"WARN: {label} mean prediction {mean_pred:.3f} is {ratio:.1f}x "
                f"training prevalence {prev:.3f}"
            )
        elif ratio < 0.3:
            warnings.append(
                f"WARN: {label} mean prediction {mean_pred:.3f} is only {ratio:.1f}x "
                f"training prevalence {prev:.3f} — possibly suppressed"
            )
    return warnings


def check_distribution(df: pd.DataFrame) -> list[str]:
    """Flag if predictions are collapsed (TTA problem) or degenerate."""
    warnings = []
    for label in LABEL_NAMES:
        if label not in df.columns:
            continue
        vals = df[label]
        std = vals.std()
        mean = vals.mean()

        if std < 0.01:
            warnings.append(
                f"FAIL: {label} predictions nearly constant "
                f"(std={std:.4f}, mean={mean:.3f})"
            )
        elif std < 0.05:
            warnings.append(
                f"WARN: {label} predictions have very low variance "
                f"(std={std:.4f}, mean={mean:.3f}) — possible TTA collapse"
            )

        near_half = ((vals > 0.4) & (vals < 0.6)).mean()
        if near_half > 0.8:
            warnings.append(
                f"WARN: {label} has {near_half:.0%} of predictions in [0.4, 0.6] "
                f"— model may be unconfident or collapsed"
            )
    return warnings


def check_no_blowup(df: pd.DataFrame) -> list[str]:
    """Estimate which labels would produce NMSE > 2.0 based on prediction mean."""
    warnings = []
    for label in LABEL_NAMES:
        if label not in df.columns:
            continue
        mean_pred = df[label].mean()
        prev = TRAIN_PREVALENCE[label]
        var = prev * (1 - prev)
        if var <= 0:
            continue
        mse_estimate = (mean_pred - prev) ** 2
        nmse_estimate = mse_estimate / var
        if nmse_estimate > 1.5:
            warnings.append(
                f"FAIL: {label} estimated NMSE from mean alone is {nmse_estimate:.2f} "
                f"(pred_mean={mean_pred:.3f}, prev={prev:.3f}) — will blow up"
            )
        elif nmse_estimate > 0.5:
            warnings.append(
                f"WARN: {label} estimated NMSE from mean alone is {nmse_estimate:.2f} "
                f"(pred_mean={mean_pred:.3f}, prev={prev:.3f})"
            )
    return warnings


def check_reference(df: pd.DataFrame, ref: pd.DataFrame) -> list[str]:
    """Compare against a known-good reference CSV, aligned by Id."""
    warnings = []
    if "Id" not in df.columns or "Id" not in ref.columns:
        warnings.append("WARN: cannot compare reference — missing Id column")
        return warnings
    merged = df.set_index("Id").join(ref.set_index("Id"), lsuffix="_new", rsuffix="_ref", how="inner")
    if len(merged) == 0:
        warnings.append("FAIL: no matching IDs between submission and reference")
        return warnings
    if len(merged) < len(df) * 0.9:
        warnings.append(
            f"WARN: only {len(merged)}/{len(df)} IDs matched reference"
        )
    for label in LABEL_NAMES:
        col_new = f"{label}_new"
        col_ref = f"{label}_ref"
        if col_new not in merged.columns or col_ref not in merged.columns:
            continue
        vals_new = merged[col_new].values
        vals_ref = merged[col_ref].values
        mean_new = np.mean(vals_new)
        mean_ref = np.mean(vals_ref)
        diff = abs(mean_new - mean_ref)
        if diff > 0.1:
            warnings.append(
                f"FAIL: {label} mean shifted by {diff:.3f} from reference "
                f"({mean_ref:.3f} -> {mean_new:.3f})"
            )
        elif diff > 0.05:
            warnings.append(
                f"WARN: {label} mean shifted by {diff:.3f} from reference "
                f"({mean_ref:.3f} -> {mean_new:.3f})"
            )
        corr = np.corrcoef(vals_new, vals_ref)[0, 1]
        if np.isnan(corr):
            warnings.append(
                f"FAIL: {label} correlation with reference is NaN — "
                f"predictions may be constant"
            )
        elif corr < 0.8:
            warnings.append(
                f"FAIL: {label} correlation with reference is {corr:.3f} — "
                f"predictions are very different"
            )
        elif corr < 0.95:
            warnings.append(
                f"WARN: {label} correlation with reference is {corr:.3f}"
            )
    return warnings


def main():
    ap = argparse.ArgumentParser(description="Pre-submission sanity checks")
    ap.add_argument("csv", type=Path, help="submission CSV to check")
    ap.add_argument("--reference", type=Path, default=None,
                    help="known-good CSV to compare against")
    ap.add_argument("--test-ids", type=Path, default=None,
                    help="test_ids.csv to validate ID set against")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Checking {args.csv} ({len(df)} rows)\n")

    all_issues: list[str] = []

    print("=== Format ===")
    issues = check_format(df, test_ids_csv=args.test_ids)
    all_issues.extend(issues)
    print("\n".join(issues) if issues else "OK")

    print("\n=== Prevalence ===")
    issues = check_prevalence(df)
    all_issues.extend(issues)
    print("\n".join(issues) if issues else "OK")

    print("\n=== Distribution ===")
    issues = check_distribution(df)
    all_issues.extend(issues)
    print("\n".join(issues) if issues else "OK")

    print("\n=== Blowup detection ===")
    issues = check_no_blowup(df)
    all_issues.extend(issues)
    print("\n".join(issues) if issues else "OK")

    if args.reference:
        ref = pd.read_csv(args.reference)
        print(f"\n=== Reference comparison ({args.reference}) ===")
        issues = check_reference(df, ref)
        all_issues.extend(issues)
        print("\n".join(issues) if issues else "OK")

    print("\n=== Per-label summary ===")
    print(f"{'Label':30s}  {'Mean':>8s}  {'Std':>8s}  {'Prevalence':>10s}  {'Ratio':>6s}")
    for label in LABEL_NAMES:
        if label not in df.columns:
            continue
        m = df[label].mean()
        s = df[label].std()
        p = TRAIN_PREVALENCE[label]
        r = m / p if p > 0 else float("inf")
        print(f"  {label:28s}  {m:8.4f}  {s:8.4f}  {p:10.3f}  {r:6.1f}x")

    fails = [i for i in all_issues if i.startswith("FAIL")]
    warns = [i for i in all_issues if i.startswith("WARN")]
    print(f"\n{'='*60}")
    print(f"Result: {len(fails)} FAIL, {len(warns)} WARN")
    if fails:
        print("\nDO NOT SUBMIT — hard failures detected:")
        for f in fails:
            print(f"  {f}")
        sys.exit(1)
    elif warns:
        print("\nSubmit with caution — review warnings above")
        sys.exit(0)
    else:
        print("\nAll checks passed — safe to submit")
        sys.exit(0)


if __name__ == "__main__":
    main()
