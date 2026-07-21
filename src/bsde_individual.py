"""
Individual BSDE solver for the single-agent exponential utility problem
under a fixed, bounded exogenous risk premium theta.

Uses a Markovian reduction to a 1D PDE system:
  - Post-default regime (j=1): w^{(1)} solves a semilinear PDE with lambda=0
  - Pre-default regime (j=0): w^{(0)} solves a coupled PDE with U = w^{(1)} - w^{(0)}

Spatial state: x = X^0 (common factor).
Time: backward from T to 0 using backward Euler for linear terms,
      explicit for nonlinear driver.

# Custom — Context7 found no library equivalent (paper individual BSDE with default jump)
"""
import numpy as np
from scipy.linalg import solve_banded
from typing import Tuple, Dict, Optional
from src.model import ModelParams
from src.qsolver import solve_q_array
from src.projection import compute_eta_array


def make_spatial_grid(params: ModelParams) -> np.ndarray:
    """Return spatial grid x_i for i=0,...,n_x-1."""
    return np.linspace(params.x_min, params.x_max, params.n_x)


def build_diffusion_operator(
    n_x: int,
    dx: float,
    mu_x: float,
    sigma_x: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the tridiagonal matrix for (I - dt * L) where L = mu_x D_x + sigma_x^2/2 D_{xx}.

    Uses central differences for D_x and D_{xx} with Dirichlet boundary conditions.

    Parameters
    ----------
    n_x : int
    dx : float
    mu_x : float
        Drift of X^0.
    sigma_x : float
        Diffusion of X^0.
    dt : float
        Time step.

    Returns
    -------
    ab : np.ndarray
        Banded matrix (3 x n_x) for solve_banded.
    """
    nu = sigma_x ** 2 / (2.0 * dx ** 2)   # diffusion coefficient
    c = mu_x / (2.0 * dx)                  # advection coefficient (central diff)

    # Diagonal entries
    diag = 1.0 + 2.0 * nu * dt
    off_lo = -dt * (nu - c)   # subdiagonal (i-1)
    off_hi = -dt * (nu + c)   # superdiagonal (i+1)

    # Build banded matrix for scipy.linalg.solve_banded
    # ab[0] = superdiagonal, ab[1] = diagonal, ab[2] = subdiagonal
    ab = np.zeros((3, n_x))
    ab[1, :] = diag                  # main diagonal
    ab[0, 1:] = off_hi               # superdiagonal (shifted right)
    ab[2, :-1] = off_lo              # subdiagonal (shifted left)

    # Dirichlet BC at boundaries: set to identity at i=0 and i=n_x-1
    ab[0, 1] = 0.0
    ab[1, 0] = 1.0
    ab[2, n_x - 2] = 0.0
    ab[1, n_x - 1] = 1.0

    return ab


def compute_z0(
    w: np.ndarray,
    dx: float,
    sigma_x: float,
) -> np.ndarray:
    """
    Compute Z^0 = sigma_x * d_x w using central differences.

    Z^0_i = sigma_x * (w_{i+1} - w_{i-1}) / (2 * dx)

    Parameters
    ----------
    w : np.ndarray
        Value function on spatial grid, shape (n_x,).
    dx : float
    sigma_x : float

    Returns
    -------
    np.ndarray
        Z^0 values, shape (n_x,).
    """
    n_x = len(w)
    z = np.zeros(n_x)
    # Interior: central differences
    z[1:-1] = sigma_x * (w[2:] - w[:-2]) / (2.0 * dx)
    # Boundaries: one-sided
    z[0] = sigma_x * (w[1] - w[0]) / dx
    z[-1] = sigma_x * (w[-1] - w[-2]) / dx
    return z


def driver_post_default(
    t: float,
    x_grid: np.ndarray,
    w1: np.ndarray,
    dx: float,
    params: ModelParams,
) -> np.ndarray:
    """
    Driver f^{(1)} for post-default regime (lambda=0, U=0).

    f^{(1)} = -Z^0 * theta - |theta|^2 / (2*gamma)

    where Z^0 = sigma_x * d_x w^{(1)}.
    (Z^{0,perp}=0, Z^1=0 in our simplified 1D model with no y-dependence)

    Parameters
    ----------
    t : float
    x_grid : np.ndarray
        Spatial grid, shape (n_x,).
    w1 : np.ndarray
        Post-default value function, shape (n_x,).
    dx : float
    params : ModelParams

    Returns
    -------
    np.ndarray
        Driver values, shape (n_x,).
    """
    sigma_x = params.sigma_x
    gamma = params.gamma
    n_x = len(x_grid)

    z0 = compute_z0(w1, dx, sigma_x)

    f = np.zeros(n_x)
    for i, x in enumerate(x_grid):
        theta = params.theta_exog(t, x, j=1)  # post-default regime
        f[i] = -z0[i] * theta - theta ** 2 / (2.0 * gamma)

    return f


def driver_pre_default(
    t: float,
    x_grid: np.ndarray,
    w0: np.ndarray,
    w1: np.ndarray,
    dx: float,
    params: ModelParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Driver f^{(0)} for pre-default regime.

    f^{(0)} = -Z^0 theta - |theta|^2/(2gamma) + gamma/2 |Z^1|^2
              - lambda U + lambda^2 |eta|^2/(2gamma) exp(-2q*)
              + lambda/gamma (exp(-q*) - 1)

    where:
      Z^0 = sigma_x * d_x w^{(0)}
      U = w^{(1)} - w^{(0)}
      a = lambda * |eta|^2
      r = gamma * Z^0 * eta - gamma * U + theta * eta
      q* = Q_a(r)

    Parameters
    ----------
    t : float
    x_grid : np.ndarray
    w0 : np.ndarray
        Pre-default value function, shape (n_x,).
    w1 : np.ndarray
        Post-default value function, shape (n_x,).
    dx : float
    params : ModelParams

    Returns
    -------
    f : np.ndarray
        Driver, shape (n_x,).
    q_star : np.ndarray
        Optimal q* values, shape (n_x,).
    q_residuals : np.ndarray
        Root residuals, shape (n_x,).
    """
    sigma_x = params.sigma_x
    gamma = params.gamma
    n_x = len(x_grid)

    z0 = compute_z0(w0, dx, sigma_x)
    U = w1 - w0  # jump in value at default

    # Compute coefficients at each x
    lam_arr = np.array([params.lambda_func(t, x, j=0) for x in x_grid])
    sigma_arr = np.array([params.sigma_func(t, x, j=0) for x in x_grid])
    beta_arr = np.array([params.beta_func(t, x, j=0) for x in x_grid])
    theta_arr = np.array([params.theta_exog(t, x, j=0) for x in x_grid])
    eta_arr = beta_arr / sigma_arr  # eta = beta/sigma in 1D

    # Scalar equation parameters
    a_arr = lam_arr * eta_arr ** 2
    r_arr = gamma * z0 * eta_arr - gamma * U + theta_arr * eta_arr

    # Solve q* for each x
    q_star, q_residuals, _ = solve_q_array(a_arr, r_arr)

    # Compute driver
    exp_neg_q = np.exp(np.clip(-q_star, -200, 200))
    exp_neg_2q = np.exp(np.clip(-2 * q_star, -200, 200))

    f = (
        -z0 * theta_arr
        - theta_arr ** 2 / (2.0 * gamma)
        - lam_arr * U
        + lam_arr ** 2 * eta_arr ** 2 / (2.0 * gamma) * exp_neg_2q
        + lam_arr / gamma * (exp_neg_q - 1.0)
    )

    return f, q_star, q_residuals


def solve_individual_bsde(
    params: ModelParams,
) -> Dict[str, np.ndarray]:
    """
    Solve the individual BSDE under fixed exogenous theta using backward PDE.

    Solves the coupled (w^{(0)}, w^{(1)}) system backward from T to 0.

    Parameters
    ----------
    params : ModelParams
        Model specification.

    Returns
    -------
    dict with keys:
        times : (n_t+1,) time grid
        x_grid : (n_x,) spatial grid
        w0 : (n_t+1, n_x) pre-default value function
        w1 : (n_t+1, n_x) post-default value function
        z0_pre : (n_t+1, n_x) Z^0 in pre-default regime
        z0_post : (n_t+1, n_x) Z^0 in post-default regime
        U_grid : (n_t+1, n_x) jump U = w^{(1)} - w^{(0)}
        q_star : (n_t+1, n_x) optimal q*
        p_star_pre : (n_t+1, n_x) optimal exposure p* (pre-default)
        pi_star_pre : (n_t+1, n_x) optimal holdings pi* (pre-default)
        q_residuals : (n_t+1, n_x)
        bsde_residuals : diagnostics
    """
    params.validate()
    times = np.linspace(0.0, params.T, params.n_t + 1)
    x_grid = make_spatial_grid(params)
    dt = times[1] - times[0]
    dx = x_grid[1] - x_grid[0]
    n_t = params.n_t
    n_x = params.n_x

    # Build diffusion operator
    ab = build_diffusion_operator(n_x, dx, params.mu_x, params.sigma_x, dt)

    # Storage
    w0 = np.zeros((n_t + 1, n_x))  # pre-default value function
    w1 = np.zeros((n_t + 1, n_x))  # post-default value function
    q_star = np.zeros((n_t + 1, n_x))
    q_residuals = np.zeros((n_t + 1, n_x))

    # Terminal conditions
    w0[n_t] = np.array([params.phi_0(x) for x in x_grid])
    w1[n_t] = np.array([params.phi_1(x) for x in x_grid])

    # Backward pass
    for k in range(n_t - 1, -1, -1):
        t_k = times[k]

        # --- Solve post-default w^{(1)} (simpler, no default term) ---
        f1 = driver_post_default(t_k, x_grid, w1[k + 1], dx, params)
        rhs1 = w1[k + 1] + dt * f1
        # Apply Dirichlet BC: keep boundary values from terminal condition
        rhs1[0] = params.phi_1(x_grid[0])
        rhs1[-1] = params.phi_1(x_grid[-1])
        w1[k] = solve_banded((1, 1), ab, rhs1)

        # --- Solve pre-default w^{(0)} using w^{(1)} at k+1 ---
        f0, q_k, qr_k = driver_pre_default(t_k, x_grid, w0[k + 1], w1[k + 1], dx, params)
        q_star[k + 1] = q_k
        q_residuals[k + 1] = qr_k

        rhs0 = w0[k + 1] + dt * f0
        rhs0[0] = params.phi_0(x_grid[0])
        rhs0[-1] = params.phi_0(x_grid[-1])
        w0[k] = solve_banded((1, 1), ab, rhs0)

    # Compute at t=0
    f0_0, q_0, qr_0 = driver_pre_default(times[0], x_grid, w0[0], w1[0], dx, params)
    q_star[0] = q_0
    q_residuals[0] = qr_0

    # Compute Z^0, U, optimal controls at each time
    z0_pre = np.zeros((n_t + 1, n_x))
    z0_post = np.zeros((n_t + 1, n_x))
    U_grid = np.zeros((n_t + 1, n_x))
    p_star_pre = np.zeros((n_t + 1, n_x))
    pi_star_pre = np.zeros((n_t + 1, n_x))

    for k in range(n_t + 1):
        t_k = times[k]
        z0_pre[k] = compute_z0(w0[k], dx, params.sigma_x)
        z0_post[k] = compute_z0(w1[k], dx, params.sigma_x)
        U_grid[k] = w1[k] - w0[k]

        # Recover optimal exposure p* = Z^0 + theta/gamma + (lambda/gamma)*eta*exp(-q*)
        for i, x in enumerate(x_grid):
            lam = params.lambda_func(t_k, x, j=0)
            sigma = params.sigma_func(t_k, x, j=0)
            eta = params.eta_func(t_k, x, j=0)
            theta = params.theta_exog(t_k, x, j=0)
            q_ki = q_star[k, i]
            exp_neg_q = np.exp(np.clip(-q_ki, -200, 200))

            # Paper formula: p* = Z^{0,parallel} + theta/gamma + (lambda/gamma)*eta*exp(-q*)
            p_star_pre[k, i] = (
                z0_pre[k, i]
                + theta / params.gamma
                + (lam / params.gamma) * eta * exp_neg_q
            )
            # pi* = p* / sigma in 1D
            pi_star_pre[k, i] = p_star_pre[k, i] / sigma

    return {
        "times": times,
        "x_grid": x_grid,
        "w0": w0,
        "w1": w1,
        "z0_pre": z0_pre,
        "z0_post": z0_post,
        "U_grid": U_grid,
        "q_star": q_star,
        "p_star_pre": p_star_pre,
        "pi_star_pre": pi_star_pre,
        "q_residuals": q_residuals,
        "dt": dt,
        "dx": dx,
    }


def recover_post_default_control(
    sol: Dict[str, np.ndarray],
    params: ModelParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recover optimal control in post-default regime.

    After default: lambda=0, q=r=theta*eta/gamma (since a=0),
    exp(-q*)=1 and the default term vanishes.

    p*_post = Z^0_post + theta/gamma  (no default term since lambda=0)

    Parameters
    ----------
    sol : dict
        Output of solve_individual_bsde.
    params : ModelParams

    Returns
    -------
    p_star_post : (n_t+1, n_x)
    pi_star_post : (n_t+1, n_x)
    """
    times = sol["times"]
    x_grid = sol["x_grid"]
    z0_post = sol["z0_post"]
    n_t = len(times) - 1
    n_x = len(x_grid)

    p_star_post = np.zeros((n_t + 1, n_x))
    pi_star_post = np.zeros((n_t + 1, n_x))

    for k in range(n_t + 1):
        t_k = times[k]
        for i, x in enumerate(x_grid):
            # Post-default: lambda=0, so only Brownian hedging term
            theta = params.theta_exog(t_k, x, j=1)
            sigma = params.sigma_func(t_k, x, j=1)
            p_star_post[k, i] = z0_post[k, i] + theta / params.gamma
            pi_star_post[k, i] = p_star_post[k, i] / sigma

    return p_star_post, pi_star_post
