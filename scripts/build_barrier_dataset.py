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
# - The dataset-building workflow (CSV ingestion → feature construction →
#   numerical-label generation → normalization → torch.save) follows standard
#   research engineering practice and is not uniquely attributable to any single
#   public repository.
# - The numerical labeling uses a classical PDE-based finite-difference baseline
#   (Crank–Nicolson + central finite differences for Greeks), implemented in this
#   project under src/numerics/fd_barrier.py.
# - The feature design (log-moneyness, implied volatility, rate, maturity, barrier
#   ratio, call/put indicator) reflects common quantitative finance practice for
#   option modeling; no direct reuse of third-party code is implied by this file.
# -----------------------------------------------------------------------------

import argparse
import math
import os

import numpy as np
import pandas as pd
import torch

from src.numerics.fd_barrier import price_barrier_fd, delta_fd, vega_fd
from src.utils.misc import ensure_dir


def generate_barrier_ratios(n_barriers=3, low=0.10, high=0.20, deterministic=True):
    """
    Generate n_barriers values for the barrier ratio (B/S - 1) within [low, high].
    If deterministic=True, use evenly spaced values; otherwise sample uniformly at random.
    """
    if deterministic:
        if n_barriers == 1:
            return [0.5 * (low + high)]
        return list(np.linspace(low, high, n_barriers))
    else:
        return list(np.random.uniform(low, high, size=n_barriers))


def build_barrier_dataset(
    csv_path: str,
    output_path: str,
    n_barriers_per_option: int = 3,
    barrier_low: float = 0.10,
    barrier_high: float = 0.20,
    max_rows: int | None = None,
    grid_size_spatial: int = 96,
    grid_size_time: int = 96,
    seed: int = 42,
):
    """
    Build barrier_dataset.pt starting from a CSV of vanilla option-chain quotes.

    Expected CSV schema (one row = one vanilla SPX/SPY option quote):
        - S          : spot price
        - K          : strike
        - sigma_imp  : implied volatility (decimal)
        - r          : risk-free rate (decimal)
        - T          : time-to-maturity in years
        - type       : 'C' (call) or 'P' (put)

    For each vanilla option, generate n_barriers_per_option up-and-out barrier options:
        B = S * (1 + u), with u ∈ [barrier_low, barrier_high].

    The output .pt file contains:
        - inputs:  [N, 6, 1, 1]   (log_moneyness, sigma_imp, r, T, B/S, is_call)
        - targets: [N, 3, 1, 1]   (price, delta, vega)
        - x_min:   [6]            (per-feature min, used for normalization)
        - x_max:   [6]            (per-feature max)
        - meta:    dictionary with additional dataset descriptors
    """
    np.random.seed(seed)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["S", "K", "sigma_imp", "r", "T", "type"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Filter out invalid rows (e.g., non-positive maturities/volatilities).
    df = df.copy()
    df = df[(df["T"] > 0) & (df["sigma_imp"] > 0) & (df["S"] > 0) & (df["K"] > 0)]
    df = df.reset_index(drop=True)

    if max_rows is not None:
        df = df.head(max_rows)

    n_options = len(df)
    if n_options == 0:
        raise ValueError("No valid rows found in the CSV after filtering.")

    print(f"Input CSV parsed successfully: using {n_options} vanilla option quotes.")

    n_samples = n_options * n_barriers_per_option
    C_in = 6
    C_out = 3

    # Pre-allocate tensors for efficiency.
    inputs = torch.zeros((n_samples, C_in, 1, 1), dtype=torch.float32)
    targets = torch.zeros((n_samples, C_out, 1, 1), dtype=torch.float32)

    # Collect basic statistics for meta-information.
    S_values = []
    K_values = []
    T_values = []

    idx = 0
    barrier_ratios = generate_barrier_ratios(
        n_barriers=n_barriers_per_option,
        low=barrier_low,
        high=barrier_high,
        deterministic=True,
    )

    for row_idx, row in df.iterrows():
        S0 = float(row["S"])
        K = float(row["K"])
        sigma_imp = float(row["sigma_imp"])
        r = float(row["r"])
        T = float(row["T"])
        opt_type = str(row["type"]).strip().upper()

        if opt_type not in ("C", "P"):
            # Skip rows with unrecognized option type.
            continue

        is_call = 1.0 if opt_type == "C" else 0.0

        S_values.append(S0)
        K_values.append(K)
        T_values.append(T)

        for u in barrier_ratios:
            if idx >= n_samples:
                break

            B = S0 * (1.0 + u)

            # Numerical labels via FD baseline (Crank–Nicolson + finite differences).
            try:
                price = price_barrier_fd(
                    S0,
                    K,
                    B,
                    r,
                    sigma_imp,
                    T,
                    M=grid_size_time,
                    N=grid_size_spatial,
                    is_call=(opt_type == "C"),
                    is_up_and_out=True,
                )
                delta = delta_fd(
                    S0,
                    K,
                    B,
                    r,
                    sigma_imp,
                    T,
                    M=grid_size_time,
                    N=grid_size_spatial,
                    is_call=(opt_type == "C"),
                    is_up_and_out=True,
                )
                vega = vega_fd(
                    S0,
                    K,
                    B,
                    r,
                    sigma_imp,
                    T,
                    M=grid_size_time,
                    N=grid_size_spatial,
                    is_call=(opt_type == "C"),
                    is_up_and_out=True,
                )
            except Exception as e:
                print(
                    f"[WARNING] FD baseline failed for row {row_idx} "
                    f"(barrier ratio u={u:.3f}): {e}"
                )
                continue

            # Input features.
            log_moneyness = math.log(S0 / K)
            B_over_S = B / S0

            x_vec = torch.tensor(
                [
                    log_moneyness,   # 0
                    sigma_imp,       # 1
                    r,               # 2
                    T,               # 3
                    B_over_S,        # 4
                    is_call,         # 5
                ],
                dtype=torch.float32,
            )

            y_vec = torch.tensor(
                [
                    price,           # 0
                    delta,           # 1
                    vega,            # 2
                ],
                dtype=torch.float32,
            )

            inputs[idx, :, 0, 0] = x_vec
            targets[idx, :, 0, 0] = y_vec
            idx += 1

    if idx == 0:
        raise RuntimeError(
            "No valid samples were generated (e.g., the FD baseline failed for all rows)."
        )

    # Shrink tensors if some combinations were skipped.
    inputs = inputs[:idx]
    targets = targets[:idx]

    print(f"Barrier-option dataset generation completed: {idx} samples created.")

    # Input normalization: per-feature min-max across the dataset.
    x_flat = inputs.view(idx, C_in)  # [N, C_in]
    x_min, _ = x_flat.min(dim=0)
    x_max, _ = x_flat.max(dim=0)
    x_range = (x_max - x_min).clamp_min(1e-8)

    inputs_norm = (inputs - x_min.view(1, C_in, 1, 1)) / x_range.view(1, C_in, 1, 1)

    # Meta-information.
    S_values = np.array(S_values, dtype=float)
    K_values = np.array(K_values, dtype=float)
    T_values = np.array(T_values, dtype=float)

    S_min, S_max = float(S_values.min()), float(S_values.max())
    K_min, K_max = float(K_values.min()), float(K_values.max())
    T_min, T_max = float(T_values.min()), float(T_values.max())

    # Note: S_grid and t_grid are global placeholders.
    # The FD solver internally adapts grids to each (S0, K, B, T) instance;
    # these vectors are stored to support potential future extensions (e.g., physics losses).
    S_grid_global = torch.linspace(0.0, S_max * 4.0, grid_size_spatial)
    t_grid_global = torch.linspace(0.0, T_max, grid_size_time)

    meta = {
        "description": (
            "Barrier option dataset: inputs are normalized features "
            "[log(S/K), sigma_imp, r, T, B/S, is_call], "
            "targets are [price, delta, vega] from a Crank–Nicolson FD baseline (96x96)."
        ),
        "C_in": C_in,
        "C_out": C_out,
        "normalization": {
            "x_min": x_min,
            "x_max": x_max,
        },
        "S_grid": S_grid_global,
        "t_grid": t_grid_global,
        "S_min": S_min,
        "S_max": S_max,
        "K_min": K_min,
        "K_max": K_max,
        "T_min": T_min,
        "T_max": T_max,
        "n_samples": idx,
        "n_barriers_per_option": n_barriers_per_option,
        "barrier_range": [barrier_low, barrier_high],
        "grid_size_spatial": grid_size_spatial,
        "grid_size_time": grid_size_time,
    }

    # Persist to disk.
    ensure_dir(os.path.dirname(output_path) or ".")
    torch.save(
        {
            "inputs": inputs_norm,
            "targets": targets,
            "x_min": x_min,
            "x_max": x_max,
            "S_grid": S_grid_global,
            "t_grid": t_grid_global,
            "meta": meta,
        },
        output_path,
    )

    print(f"Dataset artifact written to: {output_path}")
    print(f"inputs shape:  {tuple(inputs_norm.shape)}")
    print(f"targets shape: {tuple(targets.shape)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a barrier-option dataset from an SPX/SPY option chain CSV."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the CSV file containing SPX/SPY option chain data.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/barrier_dataset.pt",
        help="Path to the output .pt artifact.",
    )
    parser.add_argument(
        "--n_barriers_per_option",
        type=int,
        default=3,
        help="Number of barrier levels generated per vanilla option.",
    )
    parser.add_argument(
        "--barrier_low",
        type=float,
        default=0.10,
        help="Lower bound for (B/S - 1) (e.g., 0.10 = 10%% above spot).",
    )
    parser.add_argument(
        "--barrier_high",
        type=float,
        default=0.20,
        help="Upper bound for (B/S - 1) (e.g., 0.20 = 20%% above spot).",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional cap on the number of vanilla options read from the CSV.",
    )
    parser.add_argument(
        "--grid_size_spatial",
        type=int,
        default=96,
        help="Number of spatial grid points N used by the FD baseline.",
    )
    parser.add_argument(
        "--grid_size_time",
        type=int,
        default=96,
        help="Number of time steps M used by the FD baseline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for barrier generation (when non-deterministic).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_barrier_dataset(
        csv_path=args.csv_path,
        output_path=args.output_path,
        n_barriers_per_option=args.n_barriers_per_option,
        barrier_low=args.barrier_low,
        barrier_high=args.barrier_high,
        max_rows=args.max_rows,
        grid_size_spatial=args.grid_size_spatial,
        grid_size_time=args.grid_size_time,
        seed=args.seed,
    )
