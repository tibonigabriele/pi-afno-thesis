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
# - The CSV aggregation performed by this script (glob + pandas summary stats)
#   follows standard data-analysis patterns commonly used in research codebases.
#
# Purpose and scope:
# - Aggregate per-file evaluation outputs produced by scripts.batch_inference_eval
#   (with --compare_fd enabled), computing mean and max absolute errors for:
#       price_err, delta_err, vega_err
# - The script scans all matching CSVs under the results/ directory (recursive),
#   so it can summarize outputs for multiple experiment suites and u settings.
#
# Output:
# - results/summary_mean_max_errors.csv
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REQUIRED_ERR_COLS = ["price_err", "delta_err", "vega_err"]


def parse_exp_and_u(filename: str) -> tuple[str | None, float | None]:
    """
    Attempt to parse:
      - exp_name: prefix before '_test_eval_'
      - u:        token like 'u015' or 'u010' or 'u020' (-> 0.15, 0.10, 0.20)

    Examples:
      'afno_phys_lam01_pftd_test_eval_u015.csv' -> ('afno_phys_lam01_pftd', 0.15)
      'fno_plain_test_eval_u010.csv'           -> ('fno_plain', 0.10)
    """
    exp_name = None
    u_val = None

    if "_test_eval_" in filename:
        exp_name = filename.split("_test_eval_")[0]

    m = re.search(r"u(\d{3})", filename)
    if m:
        u_val = int(m.group(1)) / 100.0

    return exp_name, u_val


def summarize_one_csv(path: Path) -> dict | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if not all(c in df.columns for c in REQUIRED_ERR_COLS):
        return None

    # Drop NaNs (FD can fail per-row; batch_inference_eval writes NaNs in that case)
    price = pd.to_numeric(df["price_err"], errors="coerce").dropna()
    delta = pd.to_numeric(df["delta_err"], errors="coerce").dropna()
    vega = pd.to_numeric(df["vega_err"], errors="coerce").dropna()

    exp_name, u_val = parse_exp_and_u(path.name)

    return {
        "file": path.name,
        "path": str(path.as_posix()),
        "exp_name": exp_name,
        "u": u_val,
        "n_rows": int(len(df)),
        "n_price_err": int(price.shape[0]),
        "n_delta_err": int(delta.shape[0]),
        "n_vega_err": int(vega.shape[0]),
        "price_mae": float(price.mean()) if not price.empty else float("nan"),
        "delta_mae": float(delta.mean()) if not delta.empty else float("nan"),
        "vega_mae": float(vega.mean()) if not vega.empty else float("nan"),
        "price_max": float(price.max()) if not price.empty else float("nan"),
        "delta_max": float(delta.max()) if not delta.empty else float("nan"),
        "vega_max": float(vega.max()) if not vega.empty else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize mean/max absolute errors from batch_inference_eval CSV outputs."
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
        "--output_csv",
        type=str,
        default="results/summary_mean_max_errors.csv",
        help="Output CSV path for the summary.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.is_dir():
        raise FileNotFoundError(f"results_root not found or not a directory: {results_root}")

    paths = sorted(results_root.glob(args.pattern))

    rows: list[dict] = []
    skipped = 0

    for p in paths:
        r = summarize_one_csv(p)
        if r is None:
            skipped += 1
            continue
        rows.append(r)

    out = pd.DataFrame(rows)

    if out.empty:
        print("No valid eval CSVs found (missing required *_err columns?).")
        print(f"Scanned: {len(paths)} files, skipped: {skipped}")
        return

    # Useful ordering: by u then exp_name then file
    sort_cols = [c for c in ["u", "exp_name", "file"] if c in out.columns]
    out = out.sort_values(sort_cols)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print(out)
    print(f"\nSaved: {output_csv}")
    print(f"Scanned: {len(paths)} files | summarized: {len(out)} | skipped: {skipped}")


if __name__ == "__main__":
    main()