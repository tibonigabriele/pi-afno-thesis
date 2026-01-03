# pi-afno-thesis — Physics-Informed (A)FNO for Barrier Option Pricing and Greeks

This repository contains the implementation developed for the MSc thesis of **Gabriele Tiboni** (Computer Engineering, University of Padua — UniPD). The work targets **low-latency pricing and Greeks estimation for up-and-out barrier options** under the **Black–Scholes PDE**, using **Fourier Neural Operator (FNO)** baselines and **physics-informed** and **adaptive** variants.

**Supervisor:** Prof. **Loris Nanni** (UniPD).

The codebase is organized as a **reproducible pipeline**:

**dataset preparation → model definition → training → offline inference → latency benchmarking**

> **Scope note.** The “AFNO-like” model implemented here uses a lightweight **Fourier-mode gating** mechanism inspired by AFNO-style ideas; it is **not** intended as a 1:1 reproduction of any full AFNO architecture from the literature.

---

## Main idea

Given a market configuration described by:
- log-moneyness `log(S/K)`
- implied volatility `sigma_imp`
- risk-free rate `r`
- time to maturity `T`
- normalized barrier ratio `B/S`
- option type indicator `1_call`

the models learn to output in a **single forward pass**:
- barrier option price `V`
- delta `Δ`
- vega `ν`

Ground-truth labels are generated with a **Crank–Nicolson finite-difference solver** for the 1D Black–Scholes PDE with **knock-out barrier conditions**, and Greeks are computed via **central finite differences** reusing the same solver.

---

## Repository structure (high-level)

A typical layout is:

- `data/`  
  Persistent artifacts (CSV option chains, serialized `.pt` datasets, checkpoints).

- `src/`  
  Core reusable modules:
  - `src/models/` — FNO baseline and AFNO-like gated variants  
  - `src/numerics/` — Crank–Nicolson FD solver + Greeks via finite differences  
  - `src/training/` — generic training loop, physics-loss helpers  
  - `src/utils/` — seeding, device selection, utility functions

- `scripts/`  
  Entry points that orchestrate workflow stages (dataset building, training, inference, benchmarking).

---

## Requirements

- Python 3.10+ recommended
- PyTorch (CUDA optional but recommended for training and latency runs)
- NumPy

Example (CPU-only PyTorch shown; replace with your preferred CUDA build if needed):
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install --upgrade pip
pip install torch numpy

```

----------

## Quickstart

### 1) Prepare data (CSV → barrier dataset)

The workflow assumes an initial CSV of vanilla SPX/SPY-like option quotes, then expands each row into multiple barrier configurations (e.g., barriers sampled in a controlled range above spot).

Typical pipeline:

```bash
# Example: build a barrier dataset from an existing CSV
python -m scripts.build_barrier_dataset \
  --csv_path data/spx_spy_options.csv \
  --output_path data/barrier_dataset.pt

```

### 2) Train a model (baseline or physics-informed)

Training is handled via scripts in `scripts/`, using the reusable training loop in `src/training/`.  
Check the available entry points in your local `scripts/` folder (the repository is designed so that swapping models does not require changes in the core training logic).

### 3) Offline evaluation (batch inference → CSV)

Offline inference is performed on an unseen dataset split and exported to CSV for subsequent analysis:

```bash
python -m scripts.batch_inference_eval \
  --dataset_path data/barrier_dataset_test.pt \
  --checkpoint_path checkpoints/<run_name>/best_model.pt \
  --model_type <fno|fno_pino|afno|afno_pino> \
  --output_csv results/<run_name>_eval.csv

```

### 4) Latency benchmarking

Forward-pass latency is benchmarked separately from training:

```bash
python -m scripts.benchmark_latency \
  --dataset_path data/barrier_dataset.pt \
  --checkpoint_path checkpoints/<run_name>/best_model.pt \
  --model_type <fno|fno_pino|afno|afno_pino> \
  --n_warmup 10 \
  --n_runs 100

```

----------

## Physics-informed training (PINO-style)

Physics-informed variants incorporate a **Black–Scholes residual penalty** and (optionally) a **barrier boundary penalty**. Two implementations are available in this codebase:

1.  **Finite-difference residual on a grid** (PDE residual computed by FD stencils).
    
2.  **Autodiff residual in transformed coordinates** (derivatives computed via `torch.autograd` w.r.t. normalized inputs, then mapped to the raw scale).
    

The training loop supports a generic `physics_loss_fn(model, batch, outputs)` hook and a scalar weight `lambda_phys`.

----------

## Reproducibility

-   Deterministic seeds (Python, NumPy, PyTorch) via utilities in `src/utils/`.
    
-   Device selection (CPU/GPU) is centralized.
    
-   Checkpoints are saved as `best_model.pt` containing:
    
    -   `model_state_dict`
        
    -   `optimizer_state_dict` (when available)
        
    -   `best_val_loss`
        

----------

## Attribution and similarity notes (IP / rights)

This repository contains original code written for an academic thesis. However, it implements widely used mathematical definitions and numerical schemes; therefore, parts of the structure will naturally resemble many public implementations.

-   **FNO spectral layers.** The FFT-based spectral convolution pattern (FFT → low-frequency truncation → inverse FFT) follows the canonical formulation introduced by Li et al. (2021).
    
-   **Padding strategies (ZFNO/MFNO).** Zero-padding and mirror-padding are standard approaches to mitigate non-periodicity when using Fourier-based operators, as discussed in Lee et al. (2025).
    
-   **Physics-informed operator learning (PINO).** The physics-informed residual penalty follows the established PINO/PINN practice of penalizing PDE residual violations and constraint terms.
    
-   **Crank–Nicolson / Thomas algorithm.** These are standard numerical methods for parabolic PDEs (including Black–Scholes). Similarities to other solvers are expected due to the shared discretization.
    

To the best of the author’s knowledge, **no code has been copied verbatim** from external repositories. If you identify any segment that appears substantially identical to an existing implementation, please open an issue so proper attribution can be added.

----------

## Citation

If you use this code in academic work, please cite:

-   **Gabriele Tiboni**, _Fast Pricing and Greeks Computation for Barrier Options via Physics-Informed Fourier Neural Operators_, M.Sc. in Computer Engineering, University of Padua (UniPD), 2025/2026.  
    Supervisor: **Loris Nanni**.
    

----------

## Key references (as used in Chapters 1–3)

-   Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar, _Fourier Neural Operator for Parametric Partial Differential Equations_, 2021.
    
-   N. Kovachki et al., _Neural Operator: Learning Maps Between Function Spaces_, 2021.
    
-   D. Lee et al., _Fourier Neural Operators for Non-Markovian Processes: Approximation Theorems and Experiments_, 2025.  
    (Used as the primary reference for ZFNO/MFNO padding discussions.)
    
-   A. Ibrahim et al., _Space–Time Parallel Scaling of Parareal with a Physics-Informed Fourier Neural Operator Coarse Propagator Applied to the Black–Scholes Equation_, 2025.
    
-   H. Song et al., _Forecasting Stock Market Indices Using Padding-Based Fourier Transform Denoising and Time Series Deep Learning Models_, 2021.
    
-   M. Raissi, P. Perdikaris, and G. Karniadakis, _Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations_, 2019.
    
-   J. Sirignano and R. Cont, _Universal features of price formation in financial markets: perspectives from deep learning_, 2018.
    
-   Standard references for the **Black–Scholes model**, **finite-difference pricing**, and **Greeks via numerical differentiation** (e.g., J. Hull; P. Wilmott).
    

----------

## License

No license is provided in this repository. Unless an explicit `LICENSE` file is added, the contents should be considered **all rights reserved** by default.

----------

## Disclaimer

This repository is provided for **research and educational purposes** only. It does **not** constitute financial advice and is **not** production-ready for trading or risk management without independent validation, controls, and compliance review.