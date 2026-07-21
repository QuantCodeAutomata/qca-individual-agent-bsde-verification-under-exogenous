"""
Simulation utilities for:
- Common Brownian motion W^0
- Idiosyncratic Brownian motion W^1 (or W^i for each agent)
- Common factor X^0 process
- Default time tau from intensity lambda
- Default indicator N_t and compensated martingale M_t

Follows paper conventions: left-limit coefficient evaluation at jumps,
lambda_t = 0 after default.
"""
import numpy as np
from typing import Optional, Tuple, List
from src.model import ModelParams


def make_time_grid(T: float, n_t: int) -> np.ndarray:
    """
    Uniform time grid t_k = k * Delta_t over [0, T].

    Parameters
    ----------
    T : float
        Horizon.
    n_t : int
        Number of time steps.

    Returns
    -------
    np.ndarray
        Time grid of shape (n_t + 1,).
    """
    return np.linspace(0.0, T, n_t + 1)


def simulate_brownian(
    n_t: int,
    dt: float,
    n_paths: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate standard Brownian motion increments dW on a uniform grid.

    Parameters
    ----------
    n_t : int
        Number of time steps.
    dt : float
        Time step size.
    n_paths : int
        Number of independent paths.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Shape (n_paths, n_t): Brownian increments dW_k = W_{t_{k+1}} - W_{t_k}.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_paths, n_t)) * np.sqrt(dt)


def simulate_factor_path(
    params: ModelParams,
    dW0: np.ndarray,
    x0_init: float = 0.0,
) -> np.ndarray:
    """
    Simulate common factor X^0 path: dX^0 = mu_x dt + sigma_x dW^0.

    Parameters
    ----------
    params : ModelParams
    dW0 : np.ndarray
        Brownian increments, shape (n_t,).
    x0_init : float
        Initial value of X^0.

    Returns
    -------
    np.ndarray
        X^0 path of shape (n_t + 1,).
    """
    n_t = len(dW0)
    dt = params.T / n_t
    x = np.zeros(n_t + 1)
    x[0] = x0_init
    for k in range(n_t):
        x[k + 1] = x[k] + params.mu_x * dt + params.sigma_x * dW0[k]
    return x


def simulate_default_time(
    params: ModelParams,
    times: np.ndarray,
    x_path: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Simulate default time tau from intensity lambda(t, X^0_t, 0)
    using the integrated hazard inverse method.

    tau = inf{t: Lambda(t) >= E} where E ~ Exp(1) and Lambda(t) = int_0^t lambda_s ds.

    Parameters
    ----------
    params : ModelParams
    times : np.ndarray
        Time grid of shape (n_t + 1,).
    x_path : np.ndarray
        Common factor path of shape (n_t + 1,).
    rng : np.random.Generator, optional

    Returns
    -------
    float
        Default time tau (may exceed T if no default in [0,T]).
    """
    if rng is None:
        rng = np.random.default_rng()

    E = rng.exponential(1.0)  # exponential threshold
    dt = times[1] - times[0]
    n_t = len(times) - 1

    Lambda = 0.0
    for k in range(n_t):
        t_k = times[k]
        x_k = x_path[k]
        lam_k = params.lambda_func(t_k, x_k, j=0)
        Lambda += lam_k * dt
        if Lambda >= E:
            # Interpolate default time within the interval
            # tau in [t_k, t_{k+1}]
            lam_prev = params.lambda_func(times[max(k - 1, 0)], x_path[max(k - 1, 0)], j=0)
            return float(times[k] + (E - (Lambda - lam_k * dt)) / max(lam_k, 1e-15))

    return float(np.inf)  # no default


def compute_default_indicator(
    times: np.ndarray,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute default indicator N_t = 1_{tau <= t} and
    compensated martingale M_t = N_t - int_0^t lambda_s ds.

    Parameters
    ----------
    times : np.ndarray
        Time grid of shape (n_t + 1,).
    tau : float
        Default time.

    Returns
    -------
    N : np.ndarray
        Default indicator (0 or 1), shape (n_t + 1,).
    lambda_int : np.ndarray
        Cumulative hazard int_0^t lambda_s ds, shape (n_t + 1,)
        (returned for use in computing M_t = N_t - lambda_int).
    """
    N = (times >= tau).astype(float)
    return N


def compute_compensated_martingale(
    times: np.ndarray,
    N: np.ndarray,
    params: ModelParams,
    x_path: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute compensated martingale M_t = N_t - Lambda_t where Lambda_t = int_0^t lambda_s ds.

    Uses left-limit convention for lambda before default.

    Parameters
    ----------
    times : np.ndarray
        Time grid of shape (n_t + 1,).
    N : np.ndarray
        Default indicator.
    params : ModelParams
    x_path : np.ndarray
        Common factor path.

    Returns
    -------
    M : np.ndarray
        Compensated martingale M_t, shape (n_t + 1,).
    Lambda : np.ndarray
        Cumulative hazard Lambda_t, shape (n_t + 1,).
    """
    n_t = len(times) - 1
    dt = times[1] - times[0]
    Lambda = np.zeros(n_t + 1)

    for k in range(n_t):
        t_k = times[k]
        x_k = x_path[k]
        # Use left-limit regime: j = N[k] (value just before potential jump at k+1)
        j_k = int(N[k])
        lam_k = params.lambda_func(t_k, x_k, j=j_k)
        Lambda[k + 1] = Lambda[k] + lam_k * dt

    M = N - Lambda
    return M, Lambda


def simulate_one_path(
    params: ModelParams,
    seed_common: int = 0,
    seed_idio: int = 1,
    x0_init: float = 0.0,
) -> dict:
    """
    Simulate one complete path: X^0, W^0, W^1, tau, N, M.

    Parameters
    ----------
    params : ModelParams
    seed_common : int
        Seed for W^0 (common Brownian).
    seed_idio : int
        Seed for W^1 (idiosyncratic Brownian).
    x0_init : float
        Initial state X^0_0.

    Returns
    -------
    dict with keys:
        times, dW0, dW1, x_path, tau, N, M, Lambda, dt
    """
    times = make_time_grid(params.T, params.n_t)
    dt = times[1] - times[0]
    n_t = params.n_t

    rng_common = np.random.default_rng(seed_common)
    rng_idio = np.random.default_rng(seed_idio)

    dW0 = rng_common.standard_normal(n_t) * np.sqrt(dt)
    dW1 = rng_idio.standard_normal(n_t) * np.sqrt(dt)

    x_path = simulate_factor_path(params, dW0, x0_init)
    tau = simulate_default_time(params, times, x_path, rng=rng_common)
    N = compute_default_indicator(times, tau)
    M, Lambda = compute_compensated_martingale(times, N, params, x_path)

    # Build cumulative BM paths W0, W1 (shape n_t+1)
    W0 = np.concatenate([[0.0], np.cumsum(dW0)])
    W1 = np.concatenate([[0.0], np.cumsum(dW1)])

    return {
        "times": times,
        "dt": dt,
        "dW0": dW0,
        "dW1": dW1,
        "W0": W0,
        "W1": W1,
        "x_path": x_path,
        "tau": tau,
        "N": N,
        "M": M,
        "Lambda": Lambda,
    }


def simulate_wealth(
    params: ModelParams,
    path: dict,
    p_star: np.ndarray,
    xi: float = 1.0,
) -> np.ndarray:
    """
    Simulate wealth process W_t = xi + int_0^t p_s dW^0_s + int_0^t p_s theta_s ds
    + int_0^t p_s eta_s dN_s.

    In 1D under the exogenous theta:
    dW_t = p_t dW^0_t + p_t theta_t dt + p_t eta_t dN_t

    Parameters
    ----------
    params : ModelParams
    path : dict
        Simulation output from simulate_one_path.
    p_star : np.ndarray
        Optimal exposure at each time, shape (n_t + 1,).
    xi : float
        Initial wealth.

    Returns
    -------
    np.ndarray
        Wealth path W_t of shape (n_t + 1,).
    """
    times = path["times"]
    dW0 = path["dW0"]
    N = path["N"]
    x_path = path["x_path"]
    n_t = len(dW0)
    dt = path["dt"]

    W = np.zeros(n_t + 1)
    W[0] = xi

    for k in range(n_t):
        t_k = times[k]
        x_k = x_path[k]
        j_k = int(N[k])
        p_k = p_star[k]

        sigma_k = params.sigma_func(t_k, x_k, j_k)
        eta_k = params.eta_func(t_k, x_k, j_k)
        theta_k = params.theta_exog(t_k, x_k, j_k)

        dN_k = N[k + 1] - N[k]  # 0 or 1

        W[k + 1] = (
            W[k]
            + p_k * dW0[k]
            + p_k * theta_k * dt
            + p_k * eta_k * dN_k
        )

    return W


def compute_utility(
    gamma: float,
    W_T: float,
    F_T: float,
) -> float:
    """
    Compute exponential utility U = -exp(-gamma * (W_T - F_T)).

    Parameters
    ----------
    gamma : float
        Risk aversion.
    W_T : float
        Terminal wealth.
    F_T : float
        Terminal liability F = phi(X^0_T, N_T).

    Returns
    -------
    float
        Utility value (negative, in [-inf, 0)).
    """
    return float(-np.exp(-gamma * (W_T - F_T)))
