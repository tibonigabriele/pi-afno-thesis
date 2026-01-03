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
# - The overall structure (CLI via argparse + latency micro-benchmarking with
#   warm-up, repeated runs, and optional CUDA synchronization) follows standard
#   and widely-used research engineering practices and is not uniquely attributable
#   to a specific public repository.
# - Conceptual references for context:
#   * FNO: Li et al., 2021 (Fourier Neural Operator).
#   * PINO: physics-informed operator learning literature.
# - The FD routines used here as a baseline are project-specific (see
#   src/numerics/fd_barrier.py) and implement a classical PDE-based benchmark for
#   barrier option pricing and first-order Greeks.
# -----------------------------------------------------------------------------

import argparse
import math
import os
import time
from typing import Tuple

import numpy as np
import torch

from src.models.fno_family import FNOBarrier, FNOBarrierPINO, AFNOBarrierPINO
from src.numerics.fd_barrier import price_barrier_fd, delta_fd, vega_fd
from src.utils.misc import get_device


MODEL_MAP = {
    "fno": FNOBarrier,
    "fno_pino": FNOBarrierPINO,
    "afno_pino": AFNOBarrierPINO,
}


def load_dataset_meta(dataset_path: str):
    """
    Load metadata from barrier_dataset.pt:
      - x_min, x_max for min-max feature normalization
      - meta dictionary with parameter ranges (e.g., S, T, barrier_range)
    """
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = torch.load(dataset_path, map_location="cpu")
    x_min = data.get("x_min", None)
    x_max = data.get("x_max", None)
    meta = data.get("meta", {})

    if x_min is None or x_max is None:
        raise ValueError(
            "The dataset file does not contain x_min/x_max. "
            "Ensure it was generated with build_barrier_dataset.py."
        )

    C_in = int(meta.get("C_in", data["inputs"].shape[1]))
    C_out = int(meta.get("C_out", data["targets"].shape[1]))

    return x_min, x_max, meta, C_in, C_out


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
    device: torch.device,
) -> torch.Tensor:
    """
    Build the raw feature vector and apply min-max normalization.

    Feature order (as in build_barrier_dataset.py):
        0: log_moneyness = log(S/K)
        1: sigma_imp
        2: r
        3: T
        4: B_over_S = B/S
        5: is_call = 1 (call) / 0 (put)

    Returns
    -------
    torch.Tensor
        A tensor with shape [1, C_in, 1, 1] allocated on the provided device.
    """
    opt_type = opt_type.strip().upper()
    if opt_type not in ("C", "P"):
        raise ValueError(f"Unknown option type: {opt_type}. Expected 'C' or 'P'.")

    is_call = 1.0 if opt_type == "C" else 0.0

    log_moneyness = math.log(S0 / K)
    B_over_S = B / S0

    x_raw = torch.tensor(
        [log_moneyness, sigma_imp, r, T, B_over_S, is_call],
        dtype=torch.float32,
    )

    if x_min.numel() != x_raw.numel() or x_max.numel() != x_raw.numel():
        raise ValueError(
            f"Feature size mismatch: x_raw has {x_raw.numel()} elements, "
            f"while x_min/x_max have {x_min.numel()}."
        )

    x_range = (x_max - x_min).clamp_min(1e-8)
    x_norm = (x_raw - x_min) / x_range

    return x_norm.view(1, -1, 1, 1).to(device)


def instantiate_model(
    model_type: str,
    checkpoint_path: str,
    C_in: int,
    C_out: int,
    device: torch.device,
) -> torch.nn.Module:
    """
    Instantiate the selected model, load checkpoint weights, and switch to eval().

    Both formats are supported:
      - raw state_dict checkpoints
      - dict checkpoints containing "model_state_dict"
    """
    if model_type not in MODEL_MAP:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Valid choices: {list(MODEL_MAP.keys())}"
        )

    ModelClass = MODEL_MAP[model_type]

    # Architecture aligned with the training scripts:
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


def sample_parameters(meta: dict) -> Tuple[float, float, float, float, float, float, str]:
    """
    Sample a parameter set (S0, K, sigma_imp, r, T, B, type) consistent with the
    dataset ranges and the experimental protocol described in the thesis.

    Sampling scheme:
      - S ~ U[S_min, S_max]
      - T ~ U[T_min, T_max]
      - B = S * (1 + u), with u ~ U[barrier_range[0], barrier_range[1]]
      - log(S/K) ~ U[-0.3, 0.3]
      - sigma_imp ~ U[0.10, 0.50]
      - r ~ U[0.01, 0.05]
      - type ∈ {C, P}
    """
    S_min = float(meta.get("S_min", 3000.0))
    S_max = float(meta.get("S_max", 6000.0))
    T_min = float(meta.get("T_min", 1.0 / 365.0))
    T_max = float(meta.get("T_max", 2.0))
    barrier_range = meta.get("barrier_range", [0.10, 0.20])
    u_low = float(barrier_range[0])
    u_high = float(barrier_range[1])

    S0 = np.random.uniform(S_min, S_max)
    T = np.random.uniform(T_min, T_max)
    sigma_imp = np.random.uniform(0.10, 0.50)
    r = np.random.uniform(0.01, 0.05)

    # Sample log-moneyness within a plausible operating regime.
    x_logm = np.random.uniform(-0.3, 0.3)
    K = S0 / math.exp(x_logm)

    u = np.random.uniform(u_low, u_high)
    B = S0 * (1.0 + u)

    opt_type = "C" if np.random.rand() < 0.5 else "P"

    return S0, K, sigma_imp, r, T, B, opt_type


def benchmark_model_latency(
    model: torch.nn.Module,
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    meta: dict,
    device: torch.device,
    n_warmup: int,
    n_runs: int,
    batch_size: int,
) -> float:
    """
    Measure the model forward-pass latency (mean and std, in milliseconds) for the
    specified batch_size. The model is assumed to output three channels
    (price, delta, vega).
    """
    times = []

    # Warm-up phase (not measured).
    with torch.no_grad():
        for _ in range(n_warmup):
            xs = []
            for _ in range(batch_size):
                S0, K, sigma_imp, r, T, B, opt_type = sample_parameters(meta)
                x_i = build_feature_vector(
                    S0, K, sigma_imp, r, T, B, opt_type, x_min, x_max, device
                )
                xs.append(x_i)  # each element is [1, C_in, 1, 1]
            x = torch.cat(xs, dim=0)  # [batch_size, C_in, 1, 1]
            _ = model(x)

    # Measured runs.
    with torch.no_grad():
        for _ in range(n_runs):
            xs = []
            for _ in range(batch_size):
                S0, K, sigma_imp, r, T, B, opt_type = sample_parameters(meta)
                x_i = build_feature_vector(
                    S0, K, sigma_imp, r, T, B, opt_type, x_min, x_max, device
                )
                xs.append(x_i)
            x = torch.cat(xs, dim=0)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            times.append((t1 - t0) * 1000.0)  # ms

    return float(np.mean(times)), float(np.std(times))


def benchmark_fd_latency(
    meta: dict,
    n_warmup: int,
    n_runs: int,
    grid_size_spatial: int = 96,
    grid_size_time: int = 96,
) -> float:
    """
    Measure the finite-difference (FD) latency (mean and std, in milliseconds) for
    a single instance (batch_size = 1), computing:
      - price
      - delta
      - vega
    on parameter sets sampled from the same distribution used for model benchmarks.
    """
    times = []

    # Warm-up phase (not measured).
    for _ in range(n_warmup):
        S0, K, sigma_imp, r, T, B, opt_type = sample_parameters(meta)
        is_call = (opt_type == "C")
        _ = price_barrier_fd(
            S0,
            K,
            B,
            r,
            sigma_imp,
            T,
            M=grid_size_time,
            N=grid_size_spatial,
            is_call=is_call,
            is_up_and_out=True,
        )
        _ = delta_fd(
            S0,
            K,
            B,
            r,
            sigma_imp,
            T,
            M=grid_size_time,
            N=grid_size_spatial,
            is_call=is_call,
            is_up_and_out=True,
        )
        _ = vega_fd(
            S0,
            K,
            B,
            r,
            sigma_imp,
            T,
            M=grid_size_time,
            N=grid_size_spatial,
            is_call=is_call,
            is_up_and_out=True,
        )

    # Measured runs.
    for _ in range(n_runs):
        S0, K, sigma_imp, r, T, B, opt_type = sample_parameters(meta)
        is_call = (opt_type == "C")

        t0 = time.perf_counter()
        _ = price_barrier_fd(
            S0,
            K,
            B,
            r,
            sigma_imp,
            T,
            M=grid_size_time,
            N=grid_size_spatial,
            is_call=is_call,
            is_up_and_out=True,
        )
        _ = delta_fd(
            S0,
            K,
            B,
            r,
            sigma_imp,
            T,
            M=grid_size_time,
            N=grid_size_spatial,
            is_call=is_call,
            is_up_and_out=True,
        )
        _ = vega_fd(
            S0,
            K,
            B,
            r,
            sigma_imp,
            T,
            M=grid_size_time,
            N=grid_size_spatial,
            is_call=is_call,
            is_up_and_out=True,
        )
        t1 = time.perf_counter()

        times.append((t1 - t0) * 1000.0)  # ms

    return float(np.mean(times)), float(np.std(times))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark latency: model vs. FD solver (price + delta + vega)."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/barrier_dataset.pt",
        help="Path to the .pt dataset built by build_barrier_dataset.py (for x_min/x_max and meta).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="fno",
        choices=["fno", "fno_pino", "afno_pino"],
        help="Model family to benchmark.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (best_model.pt).",
    )
    parser.add_argument(
        "--n_warmup",
        type=int,
        default=10,
        help="Number of warm-up runs (excluded from timing statistics).",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=100,
        help="Number of timed runs.",
    )
    parser.add_argument(
        "--grid_size_spatial",
        type=int,
        default=96,
        help="Number of spatial grid points N used for the FD benchmark.",
    )
    parser.add_argument(
        "--grid_size_time",
        type=int,
        default=96,
        help="Number of time steps M used for the FD benchmark.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size used for the model latency benchmark (e.g., 1 or 128).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    print(f"Device: {device}")

    # Load dataset metadata.
    x_min, x_max, meta, C_in, C_out = load_dataset_meta(args.dataset_path)
    print(f"Dataset metadata loaded from: {args.dataset_path}")
    print(f"  C_in = {C_in}, C_out = {C_out}")
    print(f"  S range ~ [{meta.get('S_min', 'N/A')}, {meta.get('S_max', 'N/A')}]")
    print(f"  T range ~ [{meta.get('T_min', 'N/A')}, {meta.get('T_max', 'N/A')}]")
    print(f"  barrier_range ~ {meta.get('barrier_range', [0.10, 0.20])}")

    # Instantiate the model.
    model = instantiate_model(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        C_in=C_in,
        C_out=C_out,
        device=device,
    )

    # Benchmark model latency.
    print("\nBenchmarking model latency (forward pass)...")
    model_mean_ms, model_std_ms = benchmark_model_latency(
        model=model,
        x_min=x_min,
        x_max=x_max,
        meta=meta,
        device=device,
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        batch_size=args.batch_size,
    )
    print(
        f"  Model ({args.model_type}) latency: "
        f"{model_mean_ms:.4f} ± {model_std_ms:.4f} ms (batch_size={args.batch_size})"
    )

    # Benchmark FD solver latency.
    print("\nBenchmarking finite-difference (FD) latency (price + delta + vega)...")
    fd_mean_ms, fd_std_ms = benchmark_fd_latency(
        meta=meta,
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        grid_size_spatial=args.grid_size_spatial,
        grid_size_time=args.grid_size_time,
    )
    print(
        f"  FD solver latency: {fd_mean_ms:.4f} ± {fd_std_ms:.4f} ms (batch_size=1)"
    )

    # Report speedup.
    if model_mean_ms > 0:
        speedup = fd_mean_ms / model_mean_ms
        print(f"\nSpeedup factor (FD / model): {speedup:.2f}×")


if __name__ == "__main__":
    main()
