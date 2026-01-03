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
# - The overall training-script structure (PyTorch dataloaders + model + optimizer
#   + training loop) follows standard and widely-used patterns in the PyTorch
#   ecosystem and is not uniquely attributable to a specific public repository.
# - The models referenced by name are based on the literature on Neural Operators:
#   * Fourier Neural Operator (FNO): Li et al., 2021.
#   * Adaptive Fourier Neural Operator (AFNO): adaptive/gated spectral variants as
#     used in recent operator-learning literature.
# - The optional input augmentation is inspired by padding-based Fourier denoising
#   / masking strategies described in the P-FTD line of work; here it is applied
#   solely to inputs while keeping targets unchanged.
# -----------------------------------------------------------------------------

import torch
import torch.optim as optim

from src.data.dataset_barrier import load_barrier_option_dataloaders
from src.data.augment_pftd import pftd_augment
from src.models.fno_family import FNOBarrier, AFNOBarrier
from src.training.loop import train_model
from src.utils.misc import set_seed, get_device, ensure_dir


def run_experiment(
    exp_name: str,
    model_class,
    use_pftd: bool = False,
    num_epochs: int = 20,
):
    set_seed(42)
    device = get_device()

    data_path = "data/barrier_dataset.pt"
    batch_size = 32
    lr = 1e-3
    checkpoint_dir = f"checkpoints/ablation_{exp_name}"
    ensure_dir(checkpoint_dir)

    train_loader, val_loader, meta = load_barrier_option_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        val_ratio=0.1,
        shuffle=True,
    )

    C_in = meta["C_in"]
    C_out = meta["C_out"]

    model = model_class(
        in_channels=C_in,
        out_channels=C_out,
        width=64,
        modes1=4,
        modes2=4,
        n_layers=4,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Default setting: no data augmentation is applied.
    augment_fn = None

    if use_pftd:
        # P-FTD-style input augmentation:
        # - applied exclusively to the input tensor batch_x
        # - targets batch_y are returned unchanged
        # The callable must respect the signature (x, y) -> (x_aug, y).
        def augment_fn(batch_x: torch.Tensor, batch_y: torch.Tensor):
            """
            Parameters
            ----------
            batch_x : torch.Tensor
                Input batch with shape (B, C, H, W).
            batch_y : torch.Tensor
                Target batch with shape (B, C_out, H, W).

            Returns
            -------
            (torch.Tensor, torch.Tensor)
                Augmented inputs with the same shape as batch_x and the original targets.
            """
            B = batch_x.shape[0]
            augmented = []
            for b in range(B):
                augmented.append(
                    pftd_augment(
                        batch_x[b],        # (C, H, W)
                        keep_fraction=0.6,
                        noise_std=0.01,
                        pad_size=4,
                        pad_mode="mirror",
                    )
                )
            batch_x_aug = torch.stack(augmented, dim=0)
            return batch_x_aug, batch_y

    print(
        f"\n=== Experiment: {exp_name} | Model: {model_class.__name__} | "
        f"Input augmentation (P-FTD): {use_pftd} ==="
    )
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
        augment_fn=augment_fn,
    )


def main():
    # (1) Baseline FNO on the reference dataset.
    run_experiment(
        exp_name="fno_plain",
        model_class=FNOBarrier,
        use_pftd=False,
        num_epochs=20,
    )

    # (2) AFNO (mode-gated) on the same dataset, without augmentation.
    run_experiment(
        exp_name="afno_plain",
        model_class=AFNOBarrier,
        use_pftd=False,
        num_epochs=20,
    )

    # (3) FNO with P-FTD-style input augmentation.
    run_experiment(
        exp_name="fno_pftd",
        model_class=FNOBarrier,
        use_pftd=True,
        num_epochs=20,
    )

    # (4) AFNO with P-FTD-style input augmentation.
    run_experiment(
        exp_name="afno_pftd",
        model_class=AFNOBarrier,
        use_pftd=True,
        num_epochs=20,
    )


if __name__ == "__main__":
    main()