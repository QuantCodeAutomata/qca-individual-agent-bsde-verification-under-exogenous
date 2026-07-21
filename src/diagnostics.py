"""
Diagnostics for BSDE and MFG equilibrium solutions.

Computes:
- BSDE residuals (approximation quality)
- q-root residuals (scalar solver quality)
- Boundedness checks for Y, Z, U, q, psi, theta
- Utility gap under control perturbations (optimality verification)
- Pre/post-default control decomposition
"""
import numpy as np
from typing import Dict, Tuple, Optional
from src.model import ModelParams


def check_boundedness(
    arr: np.ndarray,
    name: str,
    expected_max: Optional[float] = None,
) -> Dict:
    """
    Check that an array is bounded and finite.

    Parameters
    ----------
    arr : np.ndarray
    name : str
    expected_max : float, optional
        If provided, warn if |arr| > expected_max.

    Returns
    -------
    dict with stats.
    """
    max_abs = float(np.max(np.abs(arr)))
    is_finite = bool(np.all(np.isfinite(arr)))
    within_bound = (max_abs <= expected_max) if expected_max is not None else True

    return {
        "name": name,
        "max_abs": max_abs,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "is_finite": is_finite,
        "within_bound": within_bound,
        "mean": float(np.mean(arr)),
    }


def compute_bsde_residuals(
    sol: Dict,
    params: ModelParams,
) -> Dict:
    """
    Compute BSDE approximation residuals for the individual BSDE solution.

    Checks: max |q_residuals|, boundedness of Y=w0, Z^0, U.

    Parameters
    ----------
    sol : dict
        Output of solve_individual_bsde.
    params : ModelParams

    Returns
    -------
    dict of residual diagnostics.
    """
    diag = {}

    # q-root residuals
    diag["q_residual_max"] = float(np.max(np.abs(sol["q_residuals"])))
    diag["q_residual_mean"] = float(np.mean(np.abs(sol["q_residuals"])))

    # Boundedness checks
    diag["Y_pre"] = check_boundedness(sol["w0"], "Y_pre (w0)")
    diag["Y_post"] = check_boundedness(sol["w1"], "Y_post (w1)")
    diag["Z0_pre"] = check_boundedness(sol["z0_pre"], "Z0_pre")
    diag["Z0_post"] = check_boundedness(sol["z0_post"], "Z0_post")
    diag["U"] = check_boundedness(sol["U_grid"], "U = w1-w0")
    diag["q_star"] = check_boundedness(sol["q_star"], "q_star")
    diag["p_star_pre"] = check_boundedness(sol["p_star_pre"], "p_star_pre")
    diag["pi_star_pre"] = check_boundedness(sol["pi_star_pre"], "pi_star_pre")

    # Check that q residuals are small
    diag["q_residuals_ok"] = diag["q_residual_max"] < params.tol_q * 1e4

    return diag


def compute_utility_gap(
    params: ModelParams,
    sol: Dict,
    path: Dict,
    n_perturbations: int = 5,
    eps_values: Optional[np.ndarray] = None,
    n_mc: int = 1000,
    seed: int = 42,
) -> Dict:
    """
    Verify local optimality of p* by comparing utility under perturbations.

    For each eps in eps_values, construct p^{eps} = p* + eps * h (bounded h)
    and compute utility via Monte Carlo. Verify U(p*) >= U(p^{eps}).

    Parameters
    ----------
    params : ModelParams
    sol : dict
        BSDE solution (individual).
    path : dict
        Simulation path (used for default structure).
    n_perturbations : int
    eps_values : np.ndarray, optional
    n_mc : int
        Monte Carlo replications for utility.
    seed : int

    Returns
    -------
    dict with utility values and gap comparisons.
    """
    from src.simulation import simulate_one_path, simulate_wealth, compute_utility

    if eps_values is None:
        eps_values = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])

    times = sol["times"]
    x_grid = sol["x_grid"]
    n_t = len(times) - 1

    # Evaluate p* at x=0 (center of grid) across time for a 1D trajectory
    x_center_idx = len(x_grid) // 2
    p_star_at_center = sol["p_star_pre"][:, x_center_idx]

    # Perturbation direction: bounded sinusoidal
    h = np.sin(np.pi * times / params.T)  # bounded in [-1, 1]

    rng = np.random.default_rng(seed)
    utilities = {}

    for eps in eps_values:
        p_eps = p_star_at_center + eps * h
        util_list = []
        for r in range(n_mc):
            path_r = simulate_one_path(
                params,
                seed_common=seed + r * 100,
                seed_idio=seed + r * 100 + 1,
                x0_init=x_grid[x_center_idx],
            )
            W_r = simulate_wealth(params, path_r, p_eps, xi=params.xi)
            # Terminal liability
            x_T = path_r["x_path"][-1]
            j_T = int(path_r["N"][-1])
            F_T = params.phi_0(x_T) if j_T == 0 else params.phi_1(x_T)
            util_list.append(compute_utility(params.gamma, W_r[-1], F_T))
        utilities[float(eps)] = float(np.mean(util_list))

    # Verify monotonicity near eps=0
    u_star = utilities.get(0.0, None)
    gaps = {eps: u_star - u for eps, u in utilities.items() if u_star is not None}
    optimality_verified = all(gap >= -1e-4 for eps, gap in gaps.items() if eps != 0.0)

    return {
        "utilities": utilities,
        "utility_star": u_star,
        "gaps": gaps,
        "optimality_verified": optimality_verified,
    }


def pre_post_default_comparison(
    sol: Dict,
    params: ModelParams,
    p_star_post: np.ndarray,
) -> Dict:
    """
    Compare pre-default vs post-default optimal controls.

    Pre-default p* includes default term: (lambda/gamma)*eta*exp(-q*)
    Post-default p* has no default term (lambda=0)

    Parameters
    ----------
    sol : dict
    params : ModelParams
    p_star_post : (n_t+1, n_x)

    Returns
    -------
    dict with decomposition statistics.
    """
    times = sol["times"]
    x_grid = sol["x_grid"]
    n_t = len(times) - 1
    n_x = len(x_grid)

    # Default contribution: p_pre - (Z^0 + theta/gamma)
    default_term = np.zeros((n_t + 1, n_x))
    brownian_term = np.zeros((n_t + 1, n_x))

    for k in range(n_t + 1):
        t_k = times[k]
        for i, x in enumerate(x_grid):
            theta = params.theta_exog(t_k, x, j=0)
            lam = params.lambda_func(t_k, x, j=0)
            eta = params.eta_func(t_k, x, j=0)
            q_ki = sol["q_star"][k, i]
            exp_neg_q = np.exp(np.clip(-q_ki, -200, 200))

            brownian_term[k, i] = sol["z0_pre"][k, i] + theta / params.gamma
            default_term[k, i] = (lam / params.gamma) * eta * exp_neg_q

    return {
        "p_star_pre": sol["p_star_pre"],
        "p_star_post": p_star_post,
        "brownian_term": brownian_term,
        "default_term": default_term,
        "default_term_max": float(np.max(np.abs(default_term))),
        "default_term_mean": float(np.mean(np.abs(default_term))),
        "control_diff_pre_post": float(np.mean(np.abs(sol["p_star_pre"] - p_star_post))),
    }


def mfg_diagnostics(
    mfg_sol: Dict,
    params: ModelParams,
) -> Dict:
    """
    Compute diagnostics for MFG equilibrium solution.

    Parameters
    ----------
    mfg_sol : dict
        Output of solve_mfg_equilibrium.
    params : ModelParams

    Returns
    -------
    dict of diagnostics.
    """
    diag = {}

    # Convergence
    diag["outer_converged"] = mfg_sol["outer_converged"]
    diag["outer_iterations"] = mfg_sol["outer_iterations"]
    diag["outer_final_residual"] = (
        mfg_sol["outer_residuals"][-1] if mfg_sol["outer_residuals"] else np.nan
    )

    # Inner FP last iteration
    if mfg_sol["inner_diag_list"]:
        last_inner = mfg_sol["inner_diag_list"][-1]
        diag["inner_converged"] = last_inner["inner_converged"]
        diag["inner_iterations"] = last_inner["inner_iterations"]
        diag["psi_min"] = last_inner["psi_min"]
        diag["psi_max"] = last_inner["psi_max"]

    # Boundedness
    diag["psi_star"] = check_boundedness(mfg_sol["psi_star"], "psi_star")
    diag["theta_mfg"] = check_boundedness(mfg_sol["theta_mfg"], "theta_mfg")
    diag["theta_brownian"] = check_boundedness(mfg_sol["theta_brownian"], "theta_brownian")
    diag["theta_default"] = check_boundedness(mfg_sol["theta_default"], "theta_default")
    diag["clearing_residual"] = check_boundedness(
        mfg_sol["clearing_residual"], "clearing_residual"
    )
    diag["clearing_max"] = float(np.max(np.abs(mfg_sol["clearing_residual"])))

    # Positivity check: psi_star > 0
    diag["psi_positive"] = bool(np.all(mfg_sol["psi_star"] > 0))

    # Sign prediction for default component
    beta = params.beta
    eta = params.beta / params.sigma_stock  # 1D
    default_comp_mean = float(np.mean(mfg_sol["theta_default"]))
    diag["default_component_mean"] = default_comp_mean
    # If beta < 0: eta < 0, -ghat*lambda*eta*psi* > 0 (positive default premium)
    # If beta > 0: eta > 0, -ghat*lambda*eta*psi* < 0 (negative default premium)
    if beta < 0:
        diag["sign_prediction_ok"] = default_comp_mean > -1e-6
    elif beta > 0:
        diag["sign_prediction_ok"] = default_comp_mean < 1e-6
    else:
        diag["sign_prediction_ok"] = abs(default_comp_mean) < 1e-6

    return diag
