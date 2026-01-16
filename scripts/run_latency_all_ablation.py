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
# - This script is a lightweight experiment runner that invokes an existing CLI
#   benchmark module and writes stdout logs to disk. This orchestration pattern
#   (subprocess calls + log capture) is standard research engineering practice.
# - Conceptual references for context:
#   * FNO: Li et al., 2021 (Fourier Neural Operator).
#   * PINO: physics-informed operator learning literature.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass(frozen=True)
class ModelCfg:
    exp_name: str
    model_type: str  # must match scripts.benchmark_latency choices: fno / fno_pino / afno_pino
    checkpoint_path: str


def _run_and_log(cmd: List[str], log_path: str) -> int:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# Command:\n")
        f.write(" ".join(cmd) + "\n\n")
        f.write("# Timestamp:\n")
        f.write(datetime.now().isoformat(timespec="seconds") + "\n\n")
        f.write("# Output:\n")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)  # live echo to console
            f.write(line)

        return proc.wait()


def main() -> None:
    dataset_path = "data/barrier_dataset_test.pt"

    # Benchmark settings (aligned with your CLI examples)
    n_warmup = 10
    n_runs = 100
    batch_sizes = [1, 128]

    # 8 base configs (NO PFTD). These exp_name values must match your ablation runner.
    models: List[ModelCfg] = [
        # FNO supervised
        ModelCfg(
            exp_name="fno_plain",
            model_type="fno",
            checkpoint_path="checkpoints/ablation_fno_plain/best_model.pt",
        ),
        # FNO PINO (lambda sweep)
        ModelCfg(
            exp_name="fno_pino_lam001",
            model_type="fno_pino",
            checkpoint_path="checkpoints/ablation_fno_pino_lam001/best_model.pt",
        ),
        ModelCfg(
            exp_name="fno_pino_lam01",
            model_type="fno_pino",
            checkpoint_path="checkpoints/ablation_fno_pino_lam01/best_model.pt",
        ),
        ModelCfg(
            exp_name="fno_pino_lam1",
            model_type="fno_pino",
            checkpoint_path="checkpoints/ablation_fno_pino_lam1/best_model.pt",
        ),
        # AFNO supervised (no physics)
        ModelCfg(
            exp_name="afno_no_phys",
            model_type="afno_pino",  # AFNOBarrierPINO class is selected by "afno_pino"
            checkpoint_path="checkpoints/ablation_afno_no_phys/best_model.pt",
        ),
        # AFNO PINO (lambda sweep)
        ModelCfg(
            exp_name="afno_phys_lam001",
            model_type="afno_pino",
            checkpoint_path="checkpoints/ablation_afno_phys_lam001/best_model.pt",
        ),
        ModelCfg(
            exp_name="afno_phys_lam01",
            model_type="afno_pino",
            checkpoint_path="checkpoints/ablation_afno_phys_lam01/best_model.pt",
        ),
        ModelCfg(
            exp_name="afno_phys_lam1",
            model_type="afno_pino",
            checkpoint_path="checkpoints/ablation_afno_phys_lam1/best_model.pt",
        ),
    ]

    # Output folder
    out_dir = "results/latency"
    os.makedirs(out_dir, exist_ok=True)

    # Use the current python executable, run module as "-m scripts.benchmark_latency"
    py = sys.executable

    failures = 0
    for m in models:
        if not os.path.isfile(m.checkpoint_path):
            print(f"[WARNING] Missing checkpoint: {m.checkpoint_path}")
            failures += 1
            continue

        for bs in batch_sizes:
            log_path = os.path.join(out_dir, f"latency_{m.exp_name}_b{bs}.txt")
            cmd = [
                py,
                "-m",
                "scripts.benchmark_latency",
                "--dataset_path",
                dataset_path,
                "--model_type",
                m.model_type,
                "--checkpoint_path",
                m.checkpoint_path,
                "--n_warmup",
                str(n_warmup),
                "--n_runs",
                str(n_runs),
                "--batch_size",
                str(bs),
            ]

            print("\n" + "=" * 80)
            print(f"Running: {m.exp_name} | model_type={m.model_type} | batch_size={bs}")
            print("=" * 80)

            rc = _run_and_log(cmd, log_path)
            if rc != 0:
                print(f"[ERROR] Benchmark failed (exit={rc}) -> {log_path}")
                failures += 1
            else:
                print(f"[OK] Saved log: {log_path}")

    if failures > 0:
        raise SystemExit(f"\nCompleted with {failures} issue(s). Check warnings/errors above.")
    print("\nAll latency benchmarks completed successfully.")


if __name__ == "__main__":
    main()
