# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2025 Gabriele Tiboni
# MSc Thesis — Computer Engineering, University of Padua (UniPD)
#
# Rights and licensing:
# - This source file is released for academic/research dissemination as part of a
#   Master’s thesis project.
# - If this repository provides a LICENSE file, this file is distributed under
#   those terms. Otherwise, all rights are reserved by the author.
#
# Notes on originality and references:
# - The code implements a lightweight synthetic data generator (parameter sampling
#   + CSV export) following standard scientific Python conventions (NumPy/Pandas).
# - No direct reuse of third-party repository code is indicated by this file; the
#   design mirrors common patterns for producing toy/synthetic datasets used to
#   validate pipelines before ingesting real market data.
# -----------------------------------------------------------------------------

import argparse
import math
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.utils.misc import ensure_dir


def generate_synthetic_spx_spy(
    n_rows: int,
    seed: int = 42,
):
    """
    Generate a synthetic SPX/SPY options DataFrame compatible with
    scripts/build_barrier_dataset.py and aligned with the assumptions described
    in the thesis (Chapter 2).

    Output columns:
        - underlying : 'SPX' or 'SPY' (informational; not consumed by build_barrier_dataset)
        - date       : synthetic trade date
        - expiry     : synthetic expiry date
        - S          : underlying spot
        - K          : strike
        - sigma_imp  : implied volatility in decimal form (e.g., 0.2 = 20%)
        - r          : risk-free rate in decimal form
        - T          : time-to-maturity in years (days_to_expiry / 365)
        - type       : 'C' (call) or 'P' (put)

    Constraints enforced:
        - S > 0, K > 0, sigma_imp > 0, T > 0
        - log-moneyness log(S/K) sampled in a plausible range [-0.3, 0.3]
        - maturities spanning ~1 week to 2 years
        - sigma_imp sampled in [10%, 50%]
        - r sampled in [1%, 5%]
    """
    np.random.seed(seed)

    # Synthetic base date (adjust if a different reference is preferred).
    base_date = datetime(2024, 1, 2)

    # Typical tenors (days) used for time-to-maturity sampling.
    tenor_days_choices = np.array([7, 14, 30, 60, 90, 180, 365, 730], dtype=int)

    # Record container.
    records = []

    for i in range(n_rows):
        # --- Underlying selection: SPX vs. SPY ---
        if np.random.rand() < 0.5:
            underlying = "SPX"
            base_spot = 4500.0
        else:
            underlying = "SPY"
            base_spot = 450.0

        # Spot S: sampled around base_spot with a mild perturbation.
        spot_noise = np.random.normal(loc=0.0, scale=0.02)  # ~2% standard deviation
        S = base_spot * (1.0 + spot_noise)
        S = max(S, 1.0)  # numerical safeguard

        # --- Log-moneyness: cover ITM/ATM/OTM within a controlled range ---
        # x = log(S/K) ~ U[-0.3, 0.3] (approximately ±30% in moneyness).
        x_logm = np.random.uniform(-0.3, 0.3)
        K = S / math.exp(x_logm)
        K = max(K, 0.5)  # numerical safeguard

        # --- Implied volatility (10–50%) ---
        sigma_imp = np.random.uniform(0.10, 0.50)

        # --- Risk-free rate (1–5%) ---
        r = np.random.uniform(0.01, 0.05)

        # --- Time to maturity ---
        tenor_days = int(np.random.choice(tenor_days_choices))
        T = tenor_days / 365.0

        # Synthetic trade/expiry dates.
        # A small offset around base_date is used to introduce variation.
        day_offset = int(np.random.uniform(-10, 10))
        trade_date = base_date + timedelta(days=day_offset)
        expiry_date = trade_date + timedelta(days=tenor_days)

        # --- Option type: Call / Put ---
        opt_type = "C" if np.random.rand() < 0.5 else "P"

        record = {
            "underlying": underlying,
            "date": trade_date.strftime("%Y-%m-%d"),
            "expiry": expiry_date.strftime("%Y-%m-%d"),
            "S": float(S),
            "K": float(K),
            "sigma_imp": float(sigma_imp),
            "r": float(r),
            "T": float(T),
            "type": opt_type,
        }
        records.append(record)

    df = pd.DataFrame.from_records(records)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic SPX/SPY options CSV compatible with build_barrier_dataset.py"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/spx_spy_options.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=20000,
        help="Number of synthetic option records to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    df = generate_synthetic_spx_spy(
        n_rows=args.n_rows,
        seed=args.seed,
    )

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        ensure_dir(out_dir)

    df.to_csv(args.output_path, index=False)
    print(f"Synthetic SPX/SPY options CSV saved to: {args.output_path}")
    print(f"Number of rows: {len(df)}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
