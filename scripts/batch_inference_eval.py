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
# Notes on originality and references:
# - The overall structure (argparse CLI + CSV iteration + PyTorch inference +
#   optional numerical baseline comparison) follows standard research-code
#   conventions and is not uniquely attributable to a specific public repository.
# - Methodological references for context:
#   * FNO: Li et al., 2021 (Fourier Neural Operator).
#   * PINO-style training: physics-informed operator learning literature.
#   * The finite-difference routines used as reference are project-specific
#     (see src/numerics/fd_barrier.py) and implement a classical PDE-based
#     baseline for barrier option pricing and Greeks.
# -----------------------------------------------------------------------------

import argparse
import math
import os

import numpy as np
import pandas as pd
import torch

from src.models.fno_family import FNOBarrier, FNOBarrierPINO, AFNOBarrierPINO
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
    consistently with the dataset construction pipeline (see build_barrier_dataset.py).

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
        # No stored normalization statistics: return raw features.
        return x_raw.view(1, -1, 1, 1)

    if x_min.numel() != x_raw.numel() or x_max.numel() != x_raw.numel():
        raise ValueError(
            f"Feature dimensionality mismatch: expected {x_raw.numel()} entries, "
            f"but received x_min/x_max of sizes ({x_min.numel()}, {x_max.numel()})."
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
    Instantiate the selected model family, load checkpoint weights, and switch to eval().

    The loader supports both:
      (i) legacy checkpoints containing a bare state_dict, and
      (ii) dictionary checkpoints of the form:
           {"model_state_dict": ..., "optimizer_state_dict": ..., "best_val_loss": ...}
    """
    if model_type not in MODEL_MAP:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Valid values are: {list(MODEL_MAP.keys())}"
        )

    ModelClass = MODEL_MAP[model_type]

    # IMPORTANT: hyperparameters are aligned with the training scripts:
    # width=96, modes1=6, modes2=6, n_layers=4
    model = ModelClass(
        in_channels=C_in,
        out_channels=C_out,
        width=96,
        modes1=6,
        modes2=6,
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


def batch_inference_eval(
    csv_path: str,
    output_csv: str,
    dataset_path: str,
    model_type: str,
    checkpoint_path: str,
    barrier_column: str | None = "B",
    default_barrier_u: float = 0.15,
    compare_fd: bool = False,
):
    """
    Run inference for all rows in an input CSV and save an enriched CSV containing:

        price_model, delta_model, vega_model

    If compare_fd=True, also append:

        price_fd, delta_fd, vega_fd,
        price_err, delta_err, vega_err

    Parameters
    ----------
    csv_path : str
        Input CSV with mandatory columns:
        S, K, sigma_imp, r, T, type
        plus an optional barrier column (absolute barrier level) if available.
    output_csv : str
        Output CSV path (enriched).
    dataset_path : str
        Dataset .pt produced by build_barrier_dataset.py. Used to recover x_min/x_max
        and (C_in, C_out) metadata.
    model_type : str
        One of {"fno", "fno_pino", "afno_pino"}.
    checkpoint_path : str
        Path to the trained checkpoint (e.g., best_model.pt).
    barrier_column : str | None
        Name of the CSV column containing B. If None or "none", use:
            B = S * (1 + default_barrier_u).
    default_barrier_u : float
        Relative barrier offset u used when B is not provided in the CSV.
    compare_fd : bool
        If True, compute FD reference price/delta/vega and report absolute errors.
    """
    device = get_device()

    # (1) Load dataset metadata and normalization statistics.
    data = torch.load(dataset_path, map_location="cpu")
    x_min = data.get("x_min", None)
    x_max = data.get("x_max", None)
    meta = data.get("meta", {})
    C_in = meta.get("C_in", data["inputs"].shape[1])
    C_out = meta.get("C_out", data["targets"].shape[1])

    print(f"Dataset metadata loaded from: {dataset_path}")
    print(f"  C_in = {C_in}, C_out = {C_out}")

    # (2) Load the trained model and checkpoint weights.
    model = load_model(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        C_in=C_in,
        C_out=C_out,
        device=device,
    )

    # (3) Load input CSV.
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = ["S", "K", "sigma_imp", "r", "T", "type"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    n_rows = len(df)
    print(f"Running batch inference on {n_rows} rows from: {csv_path}")

    # Result buffers.
    price_model_list = []
    delta_model_list = []
    vega_model_list = []

    price_fd_list = []
    delta_fd_list = []
    vega_fd_list = []

    price_err_list = []
    delta_err_list = []
    vega_err_list = []

    # (4) Row-wise evaluation.
    processed = 0
    fd_ok = 0

    for idx in range(n_rows):
        row = df.iloc[idx]
        try:
            S0 = float(row["S"])
            K = float(row["K"])
            sigma_imp = float(row["sigma_imp"])
            r = float(row["r"])
            T = float(row["T"])
            opt_type = str(row["type"]).strip().upper()
        except Exception as e:
            print(f"[WARNING] Row {idx} skipped due to parsing error: {e}")
            price_model_list.append(np.nan)
            delta_model_list.append(np.nan)
            vega_model_list.append(np.nan)
            if compare_fd:
                price_fd_list.append(np.nan)
                delta_fd_list.append(np.nan)
                vega_fd_list.append(np.nan)
                price_err_list.append(np.nan)
                delta_err_list.append(np.nan)
                vega_err_list.append(np.nan)
            continue

        if opt_type not in ("C", "P"):
            print(f"[WARNING] Row {idx} skipped: invalid option type '{opt_type}' (expected 'C' or 'P').")
            price_model_list.append(np.nan)
            delta_model_list.append(np.nan)
            vega_model_list.append(np.nan)
            if compare_fd:
                price_fd_list.append(np.nan)
                delta_fd_list.append(np.nan)
                vega_fd_list.append(np.nan)
                price_err_list.append(np.nan)
                delta_err_list.append(np.nan)
                vega_err_list.append(np.nan)
            continue

        if S0 <= 0 or K <= 0 or sigma_imp <= 0 or T <= 0:
            print(f"[WARNING] Row {idx} skipped: invalid S, K, sigma_imp, or T.")
            price_model_list.append(np.nan)
            delta_model_list.append(np.nan)
            vega_model_list.append(np.nan)
            if compare_fd:
                price_fd_list.append(np.nan)
                delta_fd_list.append(np.nan)
                vega_fd_list.append(np.nan)
                price_err_list.append(np.nan)
                delta_err_list.append(np.nan)
                vega_err_list.append(np.nan)
            continue

        # Barrier level handling.
        if barrier_column is not None and barrier_column in df.columns:
            try:
                B = float(row[barrier_column])
            except Exception as e:
                print(f"[WARNING] Row {idx}: unable to parse barrier from column '{barrier_column}': {e}")
                B = S0 * (1.0 + default_barrier_u)
        else:
            B = S0 * (1.0 + default_barrier_u)

        # Build the (optionally normalized) feature vector.
        if x_min is not None:
            x_min_t = x_min.clone().detach()
            x_max_t = x_max.clone().detach()
        else:
            x_min_t = None
            x_max_t = None

        try:
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
            ).to(device)
        except Exception as e:
            print(f"[WARNING] Row {idx} skipped during feature construction: {e}")
            price_model_list.append(np.nan)
            delta_model_list.append(np.nan)
            vega_model_list.append(np.nan)
            if compare_fd:
                price_fd_list.append(np.nan)
                delta_fd_list.append(np.nan)
                vega_fd_list.append(np.nan)
                price_err_list.append(np.nan)
                delta_err_list.append(np.nan)
                vega_err_list.append(np.nan)
            continue

        # Forward pass.
        with torch.no_grad():
            y_pred = model(x)  # [1, C_out, 1, 1]

        price_pred = float(y_pred[0, 0, 0, 0].cpu())
        delta_pred = float(y_pred[0, 1, 0, 0].cpu()) if C_out > 1 else np.nan
        vega_pred = float(y_pred[0, 2, 0, 0].cpu()) if C_out > 2 else np.nan

        price_model_list.append(price_pred)
        delta_model_list.append(delta_pred)
        vega_model_list.append(vega_pred)

        # Optional finite-difference reference (FD baseline).
        if compare_fd:
            try:
                price_fd_val = price_barrier_fd(
                    S0,
                    K,
                    B,
                    r,
                    sigma_imp,
                    T,
                    M=96,
                    N=96,
                    is_call=(opt_type == "C"),
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
                    is_call=(opt_type == "C"),
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
                    is_call=(opt_type == "C"),
                    is_up_and_out=True,
                )

                price_fd_list.append(price_fd_val)
                delta_fd_list.append(delta_fd_val)
                vega_fd_list.append(vega_fd_val)

                price_err_list.append(abs(price_pred - price_fd_val))
                delta_err_list.append(abs(delta_pred - delta_fd_val))
                vega_err_list.append(abs(vega_pred - vega_fd_val))

                fd_ok += 1
            except Exception as e:
                print(f"[WARNING] FD baseline failed on row {idx}: {e}")
                price_fd_list.append(np.nan)
                delta_fd_list.append(np.nan)
                vega_fd_list.append(np.nan)
                price_err_list.append(np.nan)
                delta_err_list.append(np.nan)
                vega_err_list.append(np.nan)

        processed += 1
        if processed % 100 == 0:
            print(f"  Progress: processed {processed}/{n_rows} rows...")

    # (5) Append columns to the DataFrame.
    df["price_model"] = price_model_list
    df["delta_model"] = delta_model_list
    df["vega_model"] = vega_model_list

    if compare_fd:
        df["price_fd"] = price_fd_list
        df["delta_fd"] = delta_fd_list
        df["vega_fd"] = vega_fd_list

        df["price_err"] = price_err_list
        df["delta_err"] = delta_err_list
        df["vega_err"] = vega_err_list

    # (6) Write enriched CSV to disk.
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\nEnriched CSV saved to: {output_csv}")
    print(f"Rows processed: {processed}/{n_rows}")

    if compare_fd and fd_ok > 0:
        price_err_arr = np.array([e for e in price_err_list if not np.isnan(e)])
        delta_err_arr = np.array([e for e in delta_err_list if not np.isnan(e)])
        vega_err_arr = np.array([e for e in vega_err_list if not np.isnan(e)])

        print("\n=== Absolute error summary (model vs. FD baseline) ===")
        print(f"FD baseline successfully evaluated on {fd_ok} rows.")
        if price_err_arr.size > 0:
            print(f"  mean(|price_err|) = {price_err_arr.mean():.6e}")
            print(f"  max (|price_err|) = {price_err_arr.max():.6e}")
        if delta_err_arr.size > 0:
            print(f"  mean(|delta_err|) = {delta_err_arr.mean():.6e}")
            print(f"  max (|delta_err|) = {delta_err_arr.max():.6e}")
        if vega_err_arr.size > 0:
            print(f"  mean(|vega_err|)  = {vega_err_arr.mean():.6e}")
            print(f"  max (|vega_err|)  = {vega_err_arr.max():.6e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch inference (price, delta, vega) on an option CSV using a trained model."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Input CSV with columns [S, K, sigma_imp, r, T, type, (optional) B].",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV path (enriched with model/FD columns).",
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
             "If absent or 'none', B = S * (1 + default_barrier_u).",
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
        help="If set, compute price/delta/vega via FD and print error stats.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    barrier_col = args.barrier_column
    if barrier_col is not None and barrier_col.lower() == "none":
        barrier_col = None

    batch_inference_eval(
        csv_path=args.csv_path,
        output_csv=args.output_csv,
        dataset_path=args.dataset_path,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        barrier_column=barrier_col,
        default_barrier_u=args.default_barrier_u,
        compare_fd=args.compare_fd,
    )
