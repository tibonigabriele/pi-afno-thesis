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
# - The script follows a standard deep-learning training pattern (data loaders,
#   model instantiation, optimizer/scheduler setup, training loop, checkpointing).
#   This is a conventional research engineering structure and does not indicate
#   direct reuse of a specific public repository implementation.
# - Conceptual reference:
#   * FNO: Li et al., 2021 (Fourier Neural Operator).
# -----------------------------------------------------------------------------

import torch
import torch.optim as optim

from src.data.dataset_barrier import load_barrier_option_dataloaders
from src.models.fno_family import FNOBarrier
from src.training.loop import train_model
from src.utils.misc import set_seed, get_device, ensure_dir
from src.utils.data_stats import compute_target_stats


def main():
    set_seed(42)
    device = get_device()

    data_path = "data/barrier_dataset.pt"
    batch_size = 512
    num_epochs = 200
    lr = 3e-4
    weight_decay = 1e-4  # L2 regularization (weight decay)
    checkpoint_dir = "checkpoints/fno_baseline"
    ensure_dir(checkpoint_dir)

    # === Data ===
    train_loader, val_loader, meta = load_barrier_option_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        val_ratio=0.1,
        shuffle=True,
    )

    C_in = meta["C_in"]
    C_out = meta["C_out"]

    print(f"Dataset loaded from: {data_path}")
    print(f"  N_total: {meta['N_total']} (train={meta['N_train']}, val={meta['N_val']})")
    print(f"  C_in = {C_in}, C_out = {C_out}")

    # === Target statistics (computed from the dataset) ===
    target_stats = compute_target_stats(data_path)

    # === Model ===
    model = FNOBarrier(
        in_channels=C_in,
        out_channels=C_out,
        width=96,
        modes1=6,
        modes2=6,
        n_layers=4,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Scheduler: reduce the learning rate when the validation loss plateaus.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15,
    )

    # Baseline training: purely supervised (no physics-informed term).
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        physics_loss_fn=None,
        lambda_phys=0.0,
        checkpoint_dir=checkpoint_dir,
        print_every=1,
        scheduler=scheduler,
        grad_clip=1.0,
        augment_fn=None,
        target_stats=target_stats,
        normalize_loss=False,      # explicitly disabled: report loss on the raw scale
    )


if __name__ == "__main__":
    main()
