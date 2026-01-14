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

from src.data.augment_pftd import pftd_augment
from src.data.dataset_barrier import load_barrier_option_dataloaders
from src.models.fno_family import FNOBarrier, FNOBarrierPINO, AFNOBarrierPINO
from src.training.loop import train_model
from src.training.physics_loss import make_black_scholes_physics_loss
from src.utils.data_stats import compute_target_stats
from src.utils.misc import ensure_dir, get_device, set_seed


def make_pftd_augment_fn(
    keep_fraction: float = 0.6,
    noise_std: float = 0.01,
    pad_size: int = 4,
    pad_mode: str = "mirror",
):
    # signature (x, y) -> (x_aug, y)
    def augment_fn(batch_x: torch.Tensor, batch_y: torch.Tensor):
        B = batch_x.shape[0]
        augmented = []
        for b in range(B):
            augmented.append(
                pftd_augment(
                    batch_x[b],  # (C, H, W)
                    keep_fraction=keep_fraction,
                    noise_std=noise_std,
                    pad_size=pad_size,
                    pad_mode=pad_mode,
                )
            )
        return torch.stack(augmented, dim=0), batch_y

    return augment_fn


def run_one(
    exp_name: str,
    model_class,
    use_pftd: bool,
    use_physics: bool,
    *,
    data_path: str,
    batch_size: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    lambda_phys: float,
):
    set_seed(42)
    device = get_device()

    ckpt_dir = f"checkpoints/ablation_{exp_name}"
    ensure_dir(ckpt_dir)

    train_loader, val_loader, meta = load_barrier_option_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        val_ratio=0.1,
        shuffle=True,
    )

    C_in = meta["C_in"]
    C_out = meta["C_out"]
    x_min = meta.get("x_min")
    x_max = meta.get("x_max")

    model = model_class(
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

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15,
    )

    augment_fn = make_pftd_augment_fn() if use_pftd else None

    target_stats = compute_target_stats(data_path)
    physics_loss_fn = None
    lam = 0.0

    if use_physics:
        if x_min is None or x_max is None:
            raise RuntimeError("meta['x_min'/'x_max'] missing: required for physics loss.")
        physics_loss_fn = make_black_scholes_physics_loss(
            x_min=x_min,
            x_max=x_max,
            lambda_barrier=1.0,
            price_scale=target_stats["std"][0],
        )
        lam = lambda_phys

    print(
        f"\n=== {exp_name} | model={model_class.__name__} | pftd={use_pftd} | "
        f"physics={use_physics} (lambda={lam}) ==="
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        physics_loss_fn=physics_loss_fn,
        lambda_phys=lam,
        checkpoint_dir=ckpt_dir,
        print_every=1,
        scheduler=scheduler,
        grad_clip=1.0,
        augment_fn=augment_fn,
        target_stats=target_stats,
        normalize_loss=True,
    )


def main():
    data_path = "data/barrier_dataset.pt"

    # Keep consistent with main training unless you intentionally want faster ablations
    batch_size = 256
    num_epochs = 200
    lr = 3e-4
    weight_decay = 1e-4
    lambda_phys = 0.1

    # --- 6 experiments ---
    # 1) FNO plain
    run_one(
        exp_name="fno_plain",
        model_class=FNOBarrier,
        use_pftd=False,
        use_physics=False,
        data_path=data_path,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        lambda_phys=lambda_phys,
    )

    # 2) PINO plain (FNO + physics)
    run_one(
        exp_name="pino_plain",
        model_class=FNOBarrierPINO,
        use_pftd=False,
        use_physics=True,
        data_path=data_path,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        lambda_phys=lambda_phys,
    )

    # 3) AFNO plain (AFNO + physics)
    run_one(
        exp_name="afno_plain",
        model_class=AFNOBarrierPINO,
        use_pftd=False,
        use_physics=True,
        data_path=data_path,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        lambda_phys=lambda_phys,
    )

    # 4) FNO + PFTD
    run_one(
        exp_name="fno_pftd",
        model_class=FNOBarrier,
        use_pftd=True,
        use_physics=False,
        data_path=data_path,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        lambda_phys=lambda_phys,
    )

    # 5) PINO + PFTD
    run_one(
        exp_name="pino_pftd",
        model_class=FNOBarrierPINO,
        use_pftd=True,
        use_physics=True,
        data_path=data_path,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        lambda_phys=lambda_phys,
    )

    # 6) AFNO + PFTD
    run_one(
        exp_name="afno_pftd",
        model_class=AFNOBarrierPINO,
        use_pftd=True,
        use_physics=True,
        data_path=data_path,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        lambda_phys=lambda_phys,
    )


if __name__ == "__main__":
    main()
