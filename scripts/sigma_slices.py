# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2025 Gabriele Tiboni
# MSc Thesis — Computer Engineering, University of Padua (UniPD)
#
# Rights and licensing:
# - This source file is intended for academic/research dissemination as part of a
#   Master’s thesis project.
# - If this repository includes a LICENSE file, this file is distributed under
#   those terms. Otherwise, all rights are reserved by the author.
#
# Notes on originality:
# - This script performs a standard post-processing analysis over CSV outputs
#   produced by the project evaluation pipeline (pandas quantiles + slicing +
#   aggregation). This is an idiomatic research data-analysis pattern.
#
# Purpose and scope:
# - Compute "sigma-imp slices" (low / mid / high implied volatility regimes) for
#   ALL models produced by the ablation suite (8 base + 8 PFTD variants), starting
#   from batch_inference_eval outputs that contain FD comparison columns:
#       price_err, delta_err, vega_err
#
# Output:
# - One CSV per input eval CSV:
#       <eval_file>_sigma_slices.csv
# - A global summary table:
#       results/summary_sigma_slices_all_models.csv
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REQUIRED_COLS = ["sigma_imp", "price_err", "delta_err", "vega_err"]


def parse_exp_and_u(filename: str) -> tuple[str | None, float | None]:
    """
    Parse exp_name and barrier offset u from filenames like:
        afno_phys_lam01_pftd_test_eval_u015.csv
        fno_plain_test_eval_u010.csv

    Returns
    -------
    (exp_name, u)
    """
    exp_name = None
    u_val = None

    if "_test_eval_" in filename:
        exp_name = filename.split("_test_eval_")[0]

    m = re.search(r"u(\d{3})", filename)
    if m:
        u_val = int(m.group(1)) / 100.0

    return exp_name, u_val


def compute_sigma_slices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 3 volatility slices and compute mean/max absolute errors per slice.

    Slices:
      - low_sigma<=q10: sigma_imp <= 10th percentile
      - mid_sigma~q50 : sigma_imp within +/- 10% of median
      - high_sigma>=q90: sigma_imp >= 90th percentile
    """
    q10 = df["sigma_imp"].quantile(0.10)
    q50 = df["sigma_imp"].quantile(0.50)
    q90 = df["sigma_imp"].quantile(0.90)

    bins = [
        ("low_sigma<=q10", df[df["sigma_imp"] <= q10]),
        ("mid_sigma~q50", df[(df["sigma_imp"] >= q50 * 0.9) & (df["sigma_imp"] <= q50 * 1.1)]),
        ("high_sigma>=q90", df[df["sigma_imp"] >= q90]),
    ]

    rows = []
    for name, d in bins:
        # Convert to numeric and drop NaNs (FD failures propagate NaNs)
        price = pd.to_numeric(d["price_err"], errors="coerce").dropna()
        delta = pd.to_numeric(d["delta_err"], errors="coerce").dropna()
        vega = pd.to_numeric(d["vega_err"], errors="coerce").dropna()

        rows.append(
            {
                "slice": name,
                "n_rows": int(len(d)),
                "n_price_err": int(price.shape[0]),
                "n_delta_err": int(delta.shape[0]),
                "n_vega_err": int(vega.shape[0]),
                "price_mae": float(price.mean()) if not price.empty else float("nan"),
                "delta_mae": float(delta.mean()) if not delta.empty else float("nan"),
                "vega_mae": float(vega.mean()) if not vega.empty else float("nan"),
                "price_max": float(price.max()) if not price.empty else float("nan"),
                "delta_max": float(delta.max()) if not delta.empty else float("nan"),
                "vega_max": float(vega.max()) if not vega.empty else float("nan"),
                "q10_sigma": float(q10),
                "q50_sigma": float(q50),
                "q90_sigma": float(q90),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute implied-volatility (sigma_imp) slices for all ablation eval CSVs."
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results",
        help="Root directory containing evaluation CSVs (recursive scan).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*test_eval_u*.csv",
        help="Glob pattern (relative to results_root) used to find eval CSVs.",
    )
    parser.add_argument(
        "--only_ablation",
        action="store_true",
        help="If set, only process files whose exp_name starts with 'ablation_' "
             "or whose path contains 'ablation'. (Useful if results/ contains older runs.)",
    )
    parser.add_argument(
        "--output_summary_csv",
        type=str,
        default="results/summary_sigma_slices_all_models.csv",
        help="Path to write the global summary CSV.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.is_dir():
        raise FileNotFoundError(f"results_root not found: {results_root}")

    paths = sorted(results_root.glob(args.pattern))

    all_rows = []
    processed = 0
    skipped = 0

    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            skipped += 1
            continue

        if not all(c in df.columns for c in REQUIRED_COLS):
            skipped += 1
            continue

        exp_name, u = parse_exp_and_u(p.name)

        if args.only_ablation:
            # Heuristic: if you saved ablation evals under a dedicated folder, this catches it.
            # Or if filenames contain ablation_* it catches it too.
            s = str(p).lower()
            if (exp_name is None) or (("ablation" not in s) and ("ablation_" not in (exp_name or ""))):
                skipped += 1
                continue

        out_slices = compute_sigma_slices(df)

        # Add identifiers
        out_slices.insert(0, "file", p.name)
        out_slices.insert(1, "path", str(p.as_posix()))
        out_slices.insert(2, "exp_name", exp_name)
        out_slices.insert(3, "u", u)

        # Write per-file slice summary next to the eval CSV
        out_path = p.with_name(p.stem + "_sigma_slices.csv")
        out_slices.to_csv(out_path, index=False)

        all_rows.append(out_slices)
        processed += 1
        print(f"Saved: {out_path}")

    if processed == 0:
        print("No valid eval CSVs found (do they include sigma_imp and *_err columns?)")
        print(f"Scanned: {len(paths)} files, skipped: {skipped}")
        return

    summary = pd.concat(all_rows, ignore_index=True)

    # Sort for readability
    sort_cols = [c for c in ["u", "exp_name", "slice", "file"] if c in summary.columns]
    summary = summary.sort_values(sort_cols)

    out_summary = Path(args.output_summary_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)

    print(f"\nSaved global summary: {out_summary}")
    print(f"Scanned: {len(paths)} | processed: {processed} | skipped: {skipped}")


if __name__ == "__main__":
    main()