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
# - The Crank–Nicolson finite-difference scheme for the 1D Black–Scholes PDE and
#   the Thomas algorithm for tridiagonal systems are standard numerical methods
#   widely described in the computational finance literature. This file provides
#   a self-contained implementation tailored to barrier (knock-out) conditions
#   and to the thesis dataset generation pipeline.
# - The overall structure (CN discretization + tridiagonal solver + finite-diff
#   Greeks) follows common textbook/reference patterns, but this code is written
#   specifically for this thesis codebase (grid construction, barrier handling,
#   interface, and defaults).
# -----------------------------------------------------------------------------

import numpy as np


def crank_nicolson_barrier(
    S0,
    K,
    B,
    r,
    sigma,
    T,
    M=200,
    N=200,
    is_call=True,
    is_up_and_out=True,
    S_min=0.0,
    S_max_factor=4.0,
):
    """
    Crank–Nicolson finite-difference solver for 1D Black–Scholes barrier options
    (up-and-out or down-and-out).

    Returns
    -------
    float
        Option price at S0 (linear interpolation on the spatial grid).
    """
    # Spatial grid.
    S_max = S_max_factor * max(S0, K, B)
    S = np.linspace(S_min, S_max, N + 1)
    dS = S[1] - S[0]
    dt = T / M

    # Terminal payoff at maturity.
    if is_call:
        payoff = np.maximum(S - K, 0.0)
    else:
        payoff = np.maximum(K - S, 0.0)

    # Enforce knock-out barrier at maturity (Dirichlet condition on the payoff).
    if B is not None:
        if is_up_and_out:
            payoff[S >= B] = 0.0
        else:
            payoff[S <= B] = 0.0

    V = payoff.copy()

    # Crank–Nicolson coefficients for internal nodes.
    j = np.arange(1, N)
    alpha = 0.25 * dt * (sigma**2 * j**2 - r * j)
    beta = -0.5 * dt * (sigma**2 * j**2 + r)
    gamma = 0.25 * dt * (sigma**2 * j**2 + r * j)

    # Tridiagonal system matrices:
    #   A * V^{n+1} = Bmat * V^{n} + boundary_terms
    A_diag = 1 - beta
    A_lower = -alpha[1:]
    A_upper = -gamma[:-1]

    B_diag = 1 + beta
    B_lower = alpha[1:]
    B_upper = gamma[:-1]

    # Backward time-marching from maturity to t=0.
    for _ in range(M):
        # Right-hand side (internal nodes only).
        V_inner = V[1:N]
        rhs = B_diag * V_inner
        rhs[1:] += B_lower * V_inner[:-1]
        rhs[:-1] += B_upper * V_inner[1:]

        # Boundary conditions (Dirichlet at the extremes).
        # These are standard asymptotic/anchoring conditions for Black–Scholes.
        if is_call:
            # At S=0: call value ≈ 0. At large S: call behaves like S - K e^{-r tau}.
            rhs[0] += alpha[0] * 0.0
            rhs[-1] += gamma[-1] * (S_max - K * np.exp(-r * dt))
        else:
            # At S=0: put value ≈ K e^{-r tau}. At large S: put value ≈ 0.
            rhs[0] += alpha[0] * (K * np.exp(-r * dt))
            rhs[-1] += gamma[-1] * 0.0

        # Solve the tridiagonal linear system A * x = rhs (Thomas algorithm).
        V_inner = thomas_solver(A_lower, A_diag, A_upper, rhs)
        V[1:N] = V_inner

        # Re-enforce knock-out barrier after each time step (Dirichlet condition).
        if B is not None:
            if is_up_and_out:
                V[S >= B] = 0.0
            else:
                V[S <= B] = 0.0

    # Linear interpolation to obtain V(S0) from the grid.
    return np.interp(S0, S, V)


def thomas_solver(a, d, c, b):
    """
    Thomas algorithm (tridiagonal matrix solver).

    Parameters
    ----------
    a : array_like
        Subdiagonal of length (N-1).
    d : array_like
        Diagonal of length N.
    c : array_like
        Superdiagonal of length (N-1).
    b : array_like
        Right-hand side of length N.

    Returns
    -------
    np.ndarray
        Solution vector of length N.
    """
    n = len(d)
    ac, dc, cc, bc = map(np.array, (a, d, c, b))

    # Forward elimination.
    for i in range(1, n):
        mc = ac[i - 1] / dc[i - 1]
        dc[i] = dc[i] - mc * cc[i - 1]
        bc[i] = bc[i] - mc * bc[i - 1]

    # Back substitution.
    xc = dc
    xc[-1] = bc[-1] / dc[-1]

    for i in range(n - 2, -1, -1):
        xc[i] = (bc[i] - cc[i] * xc[i + 1]) / dc[i]
    return xc


def price_barrier_fd(*args, **kwargs):
    """Convenience wrapper returning a Python float for the FD price."""
    return float(crank_nicolson_barrier(*args, **kwargs))


def delta_fd(
    S0,
    K,
    B,
    r,
    sigma,
    T,
    h=1e-2,
    **kwargs,
):
    """
    Delta via central finite differences:
        Δ ≈ (V(S0 + h) - V(S0 - h)) / (2h)
    """
    up = price_barrier_fd(S0 + h, K, B, r, sigma, T, **kwargs)
    down = price_barrier_fd(S0 - h, K, B, r, sigma, T, **kwargs)
    return (up - down) / (2 * h)


def vega_fd(
    S0,
    K,
    B,
    r,
    sigma,
    T,
    h=1e-3,
    **kwargs,
):
    """
    Vega via central finite differences:
        ν ≈ (V(sigma + h) - V(sigma - h)) / (2h)
    """
    up = price_barrier_fd(S0, K, B, r, sigma + h, T, **kwargs)
    down = price_barrier_fd(S0, K, B, r, sigma - h, T, **kwargs)
    return (up - down) / (2 * h)
