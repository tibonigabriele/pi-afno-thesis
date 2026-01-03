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
# - The overall structure (argparse CLI + loading a trained PyTorch model +
#   single-sample inference + optional numerical-baseline comparison) is a common
#   research engineering pattern and does not indicate direct reuse of code from
#   a specific public repository.
# - Conceptual references for context:
#   * FNO: Li et al., 2021 (Fourier Neural Operator).
#   * PINO: physics-informed operator learning literature.
# - The FD routines used as a baseline are implemented in this project under
#   src/numerics/fd_barrier.py and reflect a classical PDE-based reference for
#   barrier option pricing and Greeks.
# -----------------------------------------------------------------------------

import argparse
import math
import os

import numpy as np
import pandas as pd
import torch

from src.models.fno_family import FNOBarrier, FNOBarrierPINO, AFNOBarrierPINO
from src.data.dataset_barrier import load_barrier_option_dataloaders
from src.numerics.fd_barrier import price_barrier_fd, delta_fd, vega_fd
from src.utils.misc import get_device


MODEL_MAP = {
    "fno": FNOBarrier,
    "fno_pino": FNOBarrierPINO,
    "afno_pino": AFNOBarrierPINO,
}


def build_feature_vector(
    S0: float,
    K: float,
    sigma_imp: float,
    r: float,
    T: float,
    B: float,
    opt_type: str,
    x_min: torch.Tensor,
    x_max: torch.Tensor,
) -> torch.Tensor:
    """
    Build the raw feature vector and apply min-max normalization using (x_min, x_max),
    consistent with build_barrier_dataset.py.

    Feature order (aligned with build_barrier_dataset.py):
        0: log_moneyness = log(S/K)
        1: sigma_imp
        2: r
        3: T
        4: B_over_S = B/S
        5: is_call (1 for call, 0 for put)
    """
    opt_type = opt_type.strip().upper()
    if opt_type not in ("C", "P"):
        raise ValueError(f"Unknown option type: {opt_type}. Expected 'C' or 'P'.")

    is_call = 1.0 if opt_type == "C" else 0.0

    log_moneyness = math.log(S0 / K)
    B_over_S = B / S0

    x_raw = torch.tensor(
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

    if x_min is None or x_max is None:
        # No persisted normalization statistics: return raw features.
        return x_raw.view(1, -1, 1, 1)

    if x_min.numel() != x_raw.numel() or x_max.numel() != x_raw.numel():
        raise ValueError(
            f"Feature-count mismatch ({x_raw.numel()}) vs x_min/x_max "
            f"({x_min.numel()}, {x_max.numel()})."
        )

    x_range = (x_max - x_min).clamp_min(1e-8)
    x_norm = (x_raw - x_min) / x_range
    return x_norm.view(1, -1, 1, 1)  # shape [1, C_in, 1, 1]


def load_model(
    model_type: str,
    checkpoint_path: str,
    C_in: int,
    C_out: int,
    device: torch.device,
):
    """
    Instantiate the selected model, load checkpoint weights, and switch to eval().

    Both formats are supported:
      - raw state_dict checkpoints
      - dict checkpoints containing the "model_state_dict" key
    """
    if model_type not in MODEL_MAP:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Valid: {list(MODEL_MAP.keys())}"
        )

    ModelClass = MODEL_MAP[model_type]

    # Note: modes are required to be > 0 even when operating on 1x1 inputs.
    model = ModelClass(
        in_channels=C_in,
        out_channels=C_out,
        width=64,
        modes1=4,
        modes2=4,
        n_layers=4,
    )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_inference(
    csv_path: str,
    row_index: int,
    dataset_path: str,
    model_type: str,
    checkpoint_path: str,
    barrier_column: str | None = "B",
    default_barrier_u: float = 0.15,
    compare_fd: bool = False,
):
    """
    Run inference on a single CSV row.

    Parameters
    ----------
    csv_path:
        Path to a CSV containing at least:
          S, K, sigma_imp, r, T, type
        and optionally a barrier column B (absolute barrier level).
    row_index:
        0-based index of the row used for inference.
    dataset_path:
        Path to the .pt artifact generated by build_barrier_dataset.py
        (used to retrieve x_min/x_max and model input/output dimensions).
    model_type:
        One of {"fno", "fno_pino", "afno_pino"}.
    checkpoint_path:
        Path to the trained checkpoint (e.g., best_model.pt).
    barrier_column:
        Name of the CSV column containing the absolute barrier level B. If None or not
        present, use B = S * (1 + default_barrier_u).
    default_barrier_u:
        Relative barrier offset used when B is not provided in the CSV.
    compare_fd:
        If True, compute a finite-difference (FD) reference (price/delta/vega) for comparison.
    """
    device = get_device()

    # 1) Load dataset artifact to retrieve normalization statistics and dimensions.
    data = torch.load(dataset_path, map_location="cpu")
    x_min = data.get("x_min", None)
    x_max = data.get("x_max", None)
    meta = data.get("meta", {})
    C_in = meta.get("C_in", data["inputs"].shape[1])
    C_out = meta.get("C_out", data["targets"].shape[1])

    print(f"Dataset metadata loaded from: {dataset_path}")
    print(f"  C_in = {C_in}, C_out = {C_out}")
    if x_min is not None and x_max is not None:
        print("  Normalization statistics (per feature):")
        for i in range(C_in):
            print(f"    feature[{i}]: min={float(x_min[i]):.4f}, max={float(x_max[i]):.4f}")
    else:
        print("  WARNING: x_min/x_max not found; input normalization will be skipped.")

    # 2) Load the model and checkpoint.
    model = load_model(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        C_in=C_in,
        C_out=C_out,
        device=device,
    )

    # 3) Load CSV and select the requested row.
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = ["S", "K", "sigma_imp", "r", "T", "type"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"row_index {row_index} out of range [0, {len(df)-1}]")

    row = df.iloc[row_index]
    S0 = float(row["S"])
    K = float(row["K"])
    sigma_imp = float(row["sigma_imp"])
    r = float(row["r"])
    T = float(row["T"])
    opt_type = str(row["type"])

    if barrier_column is not None and barrier_column in df.columns:
        B = float(row[barrier_column])
        print(f"Barrier source: CSV column '{barrier_column}' → B = {B:.6f}")
    else:
        B = S0 * (1.0 + default_barrier_u)
        print(
            "Barrier source: not provided (or barrier_column=None). "
            f"Using default B = S * (1 + {default_barrier_u:.3f}) = {B:.6f}"
        )

    # 4) Build the (optionally normalized) feature tensor.
    if x_min is not None:
        x_min_t = x_min.clone().detach()
        x_max_t = x_max.clone().detach()
    else:
        x_min_t = None
        x_max_t = None

    x = build_feature_vector(
        S0=S0,
        K=K,
        sigma_imp=sigma_imp,
        r=r,
        T=T,
        B=B,
        opt_type=opt_type,
        x_min=x_min_t,
        x_max=x_max_t,
    ).to(device)   # shape [1, C_in, 1, 1]

    # 5) Forward pass.
    with torch.no_grad():
        y_pred = model(x)  # [1, C_out, 1, 1]

    price_pred = float(y_pred[0, 0, 0, 0].cpu())
    delta_pred = float(y_pred[0, 1, 0, 0].cpu()) if C_out > 1 else float("nan")
    vega_pred = float(y_pred[0, 2, 0, 0].cpu()) if C_out > 2 else float("nan")

    print("\n=== Model prediction (single sample) ===")
    print(f"Model family: {model_type}")
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"CSV row:      {row_index}")
    print(f"S0 = {S0:.4f}, K = {K:.4f}, sigma = {sigma_imp:.4f}, r = {r:.4f}, T = {T:.4f}")
    print(f"B  = {B:.4f}, type = {opt_type}")
    print(f"Predicted price: {price_pred:.6f}")
    print(f"Predicted delta: {delta_pred:.6f}")
    print(f"Predicted vega : {vega_pred:.6f}")

    if compare_fd:
        print("\n=== Finite-difference reference (Crank–Nicolson) ===")
        price_fd = price_barrier_fd(
            S0,
            K,
            B,
            r,
            sigma_imp,
            T,
            M=96,
            N=96,
            is_call=(opt_type.upper() == "C"),
            is_up_and_out=True,
        )
        delta_fd_val = delta_fd(
            S0,
            K,
            B,
            r,
            sigma_imp,
            T,
            M=96,
            N=96,
            is_call=(opt_type.upper() == "C"),
            is_up_and_out=True,
        )
        vega_fd_val = vega_fd(
            S0,
            K,
            B,
            r,
            sigma_imp,
            T,
            M=96,
            N=96,
            is_call=(opt_type.upper() == "C"),
            is_up_and_out=True,
        )

        print(f"FD price: {price_fd:.6f}")
        print(f"FD delta: {delta_fd_val:.6f}")
        print(f"FD vega : {vega_fd_val:.6f}")

        print("\n=== Absolute errors (model vs. FD) ===")
        print(f"|price_err| = {abs(price_pred - price_fd):.6e}")
        print(f"|delta_err| = {abs(delta_pred - delta_fd_val):.6e}")
        print(f"|vega_err|  = {abs(vega_pred - vega_fd_val):.6e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference (price, delta, vega) on a single CSV row using a trained model."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to CSV with columns [S, K, sigma_imp, r, T, type, (optional) B].",
    )
    parser.add_argument(
        "--row_index",
        type=int,
        required=True,
        help="0-based index of the CSV row to use.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/barrier_dataset.pt",
        help="Path to .pt dataset built by build_barrier_dataset.py (for x_min/x_max).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="fno",
        choices=["fno", "fno_pino", "afno_pino"],
        help="Type of model to use.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (e.g., checkpoints/fno_baseline/best_model.pt).",
    )
    parser.add_argument(
        "--barrier_column",
        type=str,
        default="B",
        help="Name of the CSV column containing barrier level B. "
             "If absent or set to 'none', B = S * (1 + default_barrier_u).",
    )
    parser.add_argument(
        "--default_barrier_u",
        type=float,
        default=0.15,
        help="If barrier_column is missing/None, use B = S * (1 + u).",
    )
    parser.add_argument(
        "--compare_fd",
        action="store_true",
        help="If set, compute price/delta/vega via FD and print comparison.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    barrier_col = args.barrier_column
    if barrier_col is not None and barrier_col.lower() == "none":
        barrier_col = None

    run_inference(
        csv_path=args.csv_path,
        row_index=args.row_index,
        dataset_path=args.dataset_path,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        barrier_column=barrier_col,
        default_barrier_u=args.default_barrier_u,
        compare_fd=args.compare_fd,
    )
