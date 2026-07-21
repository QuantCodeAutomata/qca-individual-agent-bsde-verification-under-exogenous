"""
Mean-Field Game (MFG) equilibrium solver for the defaultable market model.

Implements the nested fixed-point construction:
  - Inner fixed point: psi* = S(psi*) where S(psi) = E[exp(-q^psi)/gamma]
  - Outer fixed point: v = Gamma(v) via Markovian PDE for coupled regime system
  - Recovery of equilibrium premium theta^{mfg}

Mathematical structure:
  rho^psi = gamma(v^{0,parallel} eta - u) - ghat(E[v^{0,parallel}] eta + lambda |eta|^2 psi)
  q^psi = Q_a(rho^psi),  a = lambda |eta|^2
  S(psi) = E[exp(-q^psi) / gamma]   (average over gamma distribution)

  Equilibrium premium:
  theta^{mfg} = -ghat * E[Z^{0,parallel}] - ghat * lambda * eta * psi*

# Custom — Context7 found no library equivalent (paper MFG BSDE with default jump)
"""
import numpy as np
from scipy.linalg import solve_banded
from typing import Dict, Tuple, Optional
from src.model import ModelParams
from src.qsolver import solve_q_array
from src.bsde_individual import (
    make_spatial_grid,
    build_diffusion_operator,
    compute_z0,
)


def _precompute_lambda_eta(
    times: np.ndarray,
    x_grid: np.ndarray,
    params: ModelParams,
    j: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute lambda and eta on the full (n_t+1, n_x) grid for regime j.

    Parameters
    ----------
    times : np.ndarray  shape (n_t+1,)
    x_grid : np.ndarray  shape (n_x,)
    params : ModelParams
    j : int  Regime indicator (0=pre-default, 1=post-default)

    Returns
    -------
    lam_grid, eta_grid : np.ndarray  shape (n_t+1, n_x)
    """
    n_t_p1 = len(times)
    n_x = len(x_grid)
    lam_grid = np.zeros((n_t_p1, n_x))
    eta_grid = np.zeros((n_t_p1, n_x))
    for k, t_k in enumerate(times):
        for i, x in enumerate(x_grid):
            lam_grid[k, i] = params.lambda_func(t_k, x, j=j)
            eta_grid[k, i] = params.eta_func(t_k, x, j=j)
    return lam_grid, eta_grid


# ──────────────────────────────────────────────────────────────────────────────
# Inner fixed point: psi* = S(psi*)
# ──────────────────────────────────────────────────────────────────────────────

def inner_fixed_point_S(
    psi: np.ndarray,
    v0_parallel: np.ndarray,
    u_grid: np.ndarray,
    lam_grid: np.ndarray,
    eta_grid: np.ndarray,
    params: ModelParams,
    gamma_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate S(psi) = E_gamma[exp(-q^psi)/gamma] — fully vectorized.

    For each (t, x) point and each gamma_g in gamma_arr:
      rho^psi = gamma_g*(v0p*eta - u) - ghat*(v0p*eta + lambda*|eta|^2*psi)
      q^psi   = Q_a(rho^psi),  a = lambda*|eta|^2
      S(psi)  = mean_gamma[exp(-q^psi)/gamma_g]

    Parameters
    ----------
    psi : (n_t+1, n_x)  Current psi iterate.
    v0_parallel : (n_t+1, n_x)  Ebar[Z^{0,parallel}].
    u_grid : (n_t+1, n_x)  U = w1 - w0.
    lam_grid : (n_t+1, n_x)  Pre-computed lambda values.
    eta_grid : (n_t+1, n_x)  Pre-computed eta values.
    params : ModelParams
    gamma_arr : (n_gamma,)  Quadrature nodes.

    Returns
    -------
    S_psi : (n_t+1, n_x)
    q_rep : (n_t+1, n_x)  q for representative gamma=params.gamma
    rho_rep : (n_t+1, n_x)  rho for representative gamma
    """
    ghat = params.ghat(gamma_arr)
    gamma_rep = params.gamma

    a = lam_grid * eta_grid ** 2                          # (n_t+1, n_x)
    common = v0_parallel * eta_grid - u_grid              # (n_t+1, n_x)
    ghat_term = ghat * (v0_parallel * eta_grid + lam_grid * eta_grid ** 2 * psi)

    S_psi = np.zeros_like(psi)
    for gamma_g in gamma_arr:
        rho = gamma_g * common - ghat_term                # (n_t+1, n_x)
        q_g, _, _ = solve_q_array(a, rho)
        Phi_g = np.exp(np.clip(-q_g, -200.0, 200.0)) / gamma_g
        S_psi += Phi_g / len(gamma_arr)

    # Representative gamma for diagnostics
    rho_rep = gamma_rep * common - ghat_term
    q_rep, _, _ = solve_q_array(a, rho_rep)

    return S_psi, q_rep, rho_rep


def solve_inner_fixed_point(
    v0_parallel: np.ndarray,
    u_grid: np.ndarray,
    lam_grid: np.ndarray,
    eta_grid: np.ndarray,
    params: ModelParams,
    gamma_arr: np.ndarray,
    psi_init: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Iterate psi^{k+1} = S(psi^k) until convergence.

    Parameters
    ----------
    v0_parallel : (n_t+1, n_x)  Population mean Z^0.
    u_grid : (n_t+1, n_x)  Jump U.
    lam_grid : (n_t+1, n_x)  Pre-computed lambda.
    eta_grid : (n_t+1, n_x)  Pre-computed eta.
    params : ModelParams
    gamma_arr : (n_gamma,)  Gamma quadrature nodes.
    psi_init : (n_t+1, n_x), optional  Initial psi.

    Returns
    -------
    psi_star : (n_t+1, n_x)
    q_star : (n_t+1, n_x)  q at representative gamma.
    diag : dict
    """
    if psi_init is None:
        psi = np.ones_like(v0_parallel) * (1.0 / params.gamma)
    else:
        psi = psi_init.copy()

    residuals = []
    for _ in range(params.max_iter_inner):
        S_psi, q_curr, rho_curr = inner_fixed_point_S(
            psi, v0_parallel, u_grid, lam_grid, eta_grid, params, gamma_arr
        )
        res = float(np.max(np.abs(S_psi - psi)))
        residuals.append(res)
        psi = S_psi.copy()
        if res < params.tol_inner_fp:
            break

    _, q_final, _ = inner_fixed_point_S(
        psi, v0_parallel, u_grid, lam_grid, eta_grid, params, gamma_arr
    )

    diag = {
        "inner_residuals": residuals,
        "inner_converged": residuals[-1] < params.tol_inner_fp if residuals else False,
        "inner_iterations": len(residuals),
        "psi_min": float(np.min(psi)),
        "psi_max": float(np.max(psi)),
    }
    return psi, q_final, diag


# ──────────────────────────────────────────────────────────────────────────────
# MFG BSDE driver and PDE solver
# ──────────────────────────────────────────────────────────────────────────────

def mfg_driver_post_default(
    x_grid: np.ndarray,
    w1: np.ndarray,
    dx: float,
    params: ModelParams,
    gamma_arr: np.ndarray,
) -> np.ndarray:
    """
    MFG BSDE driver for post-default regime (lambda=0).

    Post-default: lambda=0 so the driver simplifies to
      f^{mfg,(1)} = -Z^0 * theta - |theta|^2/(2*gamma)
    where theta = -ghat * z0 (Brownian component only).

    Parameters
    ----------
    x_grid : np.ndarray  shape (n_x,)
    w1 : np.ndarray  shape (n_x,)
    dx : float
    params : ModelParams
    gamma_arr : np.ndarray

    Returns
    -------
    np.ndarray  shape (n_x,)
    """
    ghat = params.ghat(gamma_arr)
    gamma_rep = params.gamma
    z0 = compute_z0(w1, dx, params.sigma_x)
    theta_approx = -ghat * z0
    return -z0 * theta_approx - theta_approx ** 2 / (2.0 * gamma_rep)


def mfg_driver_pre_default(
    x_grid: np.ndarray,
    w0: np.ndarray,
    w1: np.ndarray,
    dx: float,
    lam_arr: np.ndarray,       # (n_x,)
    eta_arr: np.ndarray,       # (n_x,)
    psi_grid: np.ndarray,      # (n_x,)
    z0_mean: np.ndarray,       # (n_x,)
    params: ModelParams,
    gamma_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MFG BSDE driver for pre-default regime — fully vectorized over x.

    f^{mfg} = ghat*Z0*M - ghat^2/(2*gamma)*|M|^2
              - lambda*U + lambda*Phi* - lambda/gamma
              + gamma*lambda^2*|eta|^2*Phi*^2/2

    where M = Ebar[Z0] + lambda*eta*psi*, Phi*=exp(-q*)/gamma.

    Parameters
    ----------
    x_grid, w0, w1 : arrays, shape (n_x,)
    dx : float
    lam_arr, eta_arr, psi_grid, z0_mean : (n_x,)
    params, gamma_arr

    Returns
    -------
    f, q_star, Phi_star : (n_x,)
    """
    gamma = params.gamma
    ghat = params.ghat(gamma_arr)

    z0 = compute_z0(w0, dx, params.sigma_x)
    U = w1 - w0

    M_arr = z0_mean + lam_arr * eta_arr * psi_grid
    theta_mfg_arr = -ghat * M_arr

    a_arr = lam_arr * eta_arr ** 2
    r_arr = gamma * z0 * eta_arr - gamma * U + theta_mfg_arr * eta_arr

    q_star, _, _ = solve_q_array(a_arr, r_arr)
    exp_neg_q = np.exp(np.clip(-q_star, -200.0, 200.0))
    Phi_star = exp_neg_q / gamma

    f = (
        ghat * z0 * M_arr
        - ghat ** 2 / (2.0 * gamma) * M_arr ** 2
        - lam_arr * U
        + lam_arr * Phi_star
        - lam_arr / gamma
        + gamma * lam_arr ** 2 * eta_arr ** 2 * Phi_star ** 2 / 2.0
    )
    return f, q_star, Phi_star


def solve_mfg_pde(
    params: ModelParams,
    gamma_arr: np.ndarray,
    psi_star: np.ndarray,
    z0_mean: np.ndarray,
    lam_grid: np.ndarray,
    eta_grid: np.ndarray,
    u_old: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Solve the MFG coupled PDE system (pre/post-default) backward in time.

    Parameters
    ----------
    params : ModelParams
    gamma_arr : np.ndarray
    psi_star : (n_t+1, n_x)  Converged inner fixed point.
    z0_mean : (n_t+1, n_x)  Ebar[Z^{0,parallel}].
    lam_grid : (n_t+1, n_x)  Pre-computed lambda.
    eta_grid : (n_t+1, n_x)  Pre-computed eta.
    u_old : optional

    Returns
    -------
    dict with keys: times, x_grid, w0, w1, z0, U_grid, q_star, Phi_star
    """
    times = np.linspace(0.0, params.T, params.n_t + 1)
    x_grid = make_spatial_grid(params)
    dt = times[1] - times[0]
    dx = x_grid[1] - x_grid[0]
    n_t = params.n_t
    n_x = params.n_x

    ab = build_diffusion_operator(n_x, dx, params.mu_x, params.sigma_x, dt)

    w0 = np.zeros((n_t + 1, n_x))
    w1 = np.zeros((n_t + 1, n_x))
    q_star = np.zeros((n_t + 1, n_x))
    Phi_star_grid = np.zeros((n_t + 1, n_x))

    w0[n_t] = np.array([params.phi_0(x) for x in x_grid])
    w1[n_t] = np.array([params.phi_1(x) for x in x_grid])

    for k in range(n_t - 1, -1, -1):
        # Post-default
        f1 = mfg_driver_post_default(x_grid, w1[k + 1], dx, params, gamma_arr)
        rhs1 = w1[k + 1] + dt * f1
        rhs1[0] = params.phi_1(x_grid[0])
        rhs1[-1] = params.phi_1(x_grid[-1])
        w1[k] = solve_banded((1, 1), ab, rhs1)

        # Pre-default — pass pre-computed lam/eta slices
        f0, q_k, Phi_k = mfg_driver_pre_default(
            x_grid, w0[k + 1], w1[k + 1], dx,
            lam_grid[k], eta_grid[k],
            psi_star[k], z0_mean[k],
            params, gamma_arr,
        )
        q_star[k] = q_k
        Phi_star_grid[k] = Phi_k

        rhs0 = w0[k + 1] + dt * f0
        rhs0[0] = params.phi_0(x_grid[0])
        rhs0[-1] = params.phi_0(x_grid[-1])
        w0[k] = solve_banded((1, 1), ab, rhs0)

    q_star[n_t] = 0.0
    Phi_star_grid[n_t] = 1.0 / params.gamma

    z0 = np.vstack([compute_z0(w0[k], dx, params.sigma_x) for k in range(n_t + 1)])
    U_grid = w1 - w0

    return {
        "times": times,
        "x_grid": x_grid,
        "w0": w0,
        "w1": w1,
        "z0": z0,
        "U_grid": U_grid,
        "q_star": q_star,
        "Phi_star": Phi_star_grid,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Outer fixed point: v = Gamma(v)
# ──────────────────────────────────────────────────────────────────────────────

def solve_mfg_equilibrium(
    params: ModelParams,
) -> Dict[str, np.ndarray]:
    """
    Solve the mean-field equilibrium via nested fixed-point iteration.

    Outer loop: Gamma(v) until convergence.
    Inner loop: psi* = S(psi*) for each outer iterate.

    Fully vectorized: no Python loops over spatial or time grid inside
    the inner fixed point or premium recovery.

    Parameters
    ----------
    params : ModelParams

    Returns
    -------
    dict with equilibrium objects and diagnostics.
    """
    params.validate()
    times = np.linspace(0.0, params.T, params.n_t + 1)
    x_grid = make_spatial_grid(params)
    n_t = params.n_t
    n_x = params.n_x

    gamma_arr = params.gamma_samples_uniform()
    ghat = params.ghat(gamma_arr)

    # Pre-compute lambda/eta on full grid once — avoids repeated Python loops
    lam_grid, eta_grid = _precompute_lambda_eta(times, x_grid, params, j=0)

    # Initialize outer iterate
    z0_mean = np.zeros((n_t + 1, n_x))
    u_grid = np.zeros((n_t + 1, n_x))
    psi_star = np.ones((n_t + 1, n_x)) * (1.0 / params.gamma)

    outer_residuals = []
    inner_diag_list = []
    pde_results = None

    for _outer_iter in range(params.max_iter_outer):
        # Inner fixed point (vectorized)
        psi_new, _q_inner, inner_diag = solve_inner_fixed_point(
            z0_mean, u_grid, lam_grid, eta_grid, params, gamma_arr,
            psi_init=psi_star,
        )
        inner_diag_list.append(inner_diag)

        # Solve MFG PDE
        pde_results = solve_mfg_pde(
            params, gamma_arr, psi_new, z0_mean, lam_grid, eta_grid, u_grid
        )

        z0_new = pde_results["z0"].copy()
        u_new = pde_results["U_grid"].copy()

        res_outer = float(np.max(np.abs(z0_new - z0_mean)))
        outer_residuals.append(res_outer)

        psi_star = psi_new.copy()
        z0_mean = z0_new.copy()
        u_grid = u_new.copy()

        if res_outer < params.tol_outer_fp:
            break

    # ── Vectorized premium recovery ──────────────────────────────────────────
    # theta^{mfg} = -ghat * Ebar[Z0] - ghat * lambda * eta * psi*
    theta_brownian = -ghat * z0_mean                              # (n_t+1, n_x)
    theta_default = -ghat * lam_grid * eta_grid * psi_star        # (n_t+1, n_x)
    theta_mfg = theta_brownian + theta_default                    # (n_t+1, n_x)

    # ── Vectorized market-clearing residual ──────────────────────────────────
    # p*(t,x) averaged over gamma distribution
    a_cl = lam_grid * eta_grid ** 2                               # (n_t+1, n_x)
    clearing_residual = np.zeros((n_t + 1, n_x))
    for gamma_g in gamma_arr:
        r_cl = (gamma_g * z0_mean * eta_grid
                - gamma_g * u_grid
                + theta_mfg * eta_grid)
        q_g, _, _ = solve_q_array(a_cl, r_cl)
        exp_neg_q = np.exp(np.clip(-q_g, -200.0, 200.0))
        p_g = z0_mean + theta_mfg / gamma_g + (lam_grid / gamma_g) * eta_grid * exp_neg_q
        clearing_residual += p_g / len(gamma_arr)

    return {
        "times": times,
        "x_grid": x_grid,
        "w0": pde_results["w0"],
        "w1": pde_results["w1"],
        "z0": z0_mean,
        "U_grid": u_grid,
        "psi_star": psi_star,
        "q_star": pde_results["q_star"],
        "Phi_star": pde_results["Phi_star"],
        "theta_mfg": theta_mfg,
        "theta_brownian": theta_brownian,
        "theta_default": theta_default,
        "clearing_residual": clearing_residual,
        "ghat": ghat,
        "gamma_arr": gamma_arr,
        "outer_residuals": outer_residuals,
        "inner_diag_list": inner_diag_list,
        "outer_converged": outer_residuals[-1] < params.tol_outer_fp if outer_residuals else False,
        "outer_iterations": len(outer_residuals),
    }
