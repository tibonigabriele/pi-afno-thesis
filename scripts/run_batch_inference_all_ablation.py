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
# Purpose and scope:
# - Convenience runner to execute scripts.batch_inference_eval over all models
#   produced by the ablation suite (with and without PFTD), using a consistent
#   evaluation setup and writing one enriched CSV per checkpoint.
# -----------------------------------------------------------------------------

from __future__ import annotations

import subprocess
from pathlib import Path


def infer_model_type(exp_name: str) -> str:
    if exp_name.startswith("fno_plain"):
        return "fno"
    if exp_name.startswith("fno_pino"):
        return "fno_pino"
    if exp_name.startswith("afno"):
        return "afno_pino"
    raise ValueError(f"Cannot infer model_type from exp_name='{exp_name}'")


def main():
    # --- shared eval config ---
    csv_path = "data/spx_spy_options_test.csv"
    dataset_path = "data/barrier_dataset_test.pt"
    out_dir = Path("results/ablation_eval_u015")
    out_dir.mkdir(parents=True, exist_ok=True)

    barrier_column = "none"
    default_barrier_u = "0.15"
    compare_fd_flag = "--compare_fd"

    # --- 16 experiments (8 base + 8 pftd) ---
    base = [
        "fno_plain",
        "fno_pino_lam0",
        "fno_pino",
        "fno_pino_lam1",
        "afno_no_phys",
        "afno_phys_lam0",
        "afno_phys",
        "afno_phys_lam1",
    ]
    experiments = base + [f"{e}_pftd" for e in base]

    for exp_name in experiments:
        model_type = infer_model_type(exp_name)
        ckpt = Path(f"checkpoints/ablation_{exp_name}/best_model.pt")
        if not ckpt.is_file():
            print(f"[SKIP] Missing checkpoint: {ckpt}")
            continue

        out_csv = out_dir / f"{exp_name}_test_eval_u015.csv"

        cmd = [
            "python",
            "-m",
            "scripts.batch_inference_eval",
            "--csv_path",
            csv_path,
            "--output_csv",
            str(out_csv),
            "--dataset_path",
            dataset_path,
            "--model_type",
            model_type,
            "--checkpoint_path",
            str(ckpt),
            "--barrier_column",
            barrier_column,
            "--default_barrier_u",
            default_barrier_u,
            compare_fd_flag,
        ]

        print(f"\n=== Running eval: {exp_name} ({model_type}) ===")
        subprocess.run(cmd, check=True)

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()