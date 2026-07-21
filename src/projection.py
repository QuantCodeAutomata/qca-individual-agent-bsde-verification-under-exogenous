"""
Projection and structural maps for the BSDE/MFG model.

In the 1D case (n=1, d0=1), sigma is a positive scalar:
  Pi_t(z) = z * sigma^T * (sigma sigma^T)^{-1} * sigma = z  (identity projection)
  eta_t = sigma^T * (sigma sigma^T)^{-1} * beta = beta / sigma

So Z^{0,parallel} = Z^0 and Z^{0,perp} = 0 in 1D.

For general n,d0: Pi_t(z) = z * sigma^T * (sigma sigma^T)^{-1} * sigma.
"""
import numpy as np
from typing import Tuple


def compute_eta(sigma: float, beta: float) -> float:
    """
    Compute eta = sigma^T (sigma sigma^T)^{-1} beta in 1D.

    In 1D: eta = beta / sigma.

    Parameters
    ----------
    sigma : float
        Asset volatility (strictly positive scalar).
    beta : float
        Jump size (scalar, > -1).

    Returns
    -------
    float
        eta value.
    """
    assert sigma > 0, f"sigma must be positive, got {sigma}"
    return beta / sigma


def compute_eta_array(sigma_arr: np.ndarray, beta_arr: np.ndarray) -> np.ndarray:
    """
    Vectorized eta computation over arrays.

    Parameters
    ----------
    sigma_arr : np.ndarray
        Array of sigma values (positive).
    beta_arr : np.ndarray
        Array of beta values.

    Returns
    -------
    np.ndarray
        eta = beta / sigma.
    """
    sigma_arr = np.asarray(sigma_arr, dtype=float)
    beta_arr = np.asarray(beta_arr, dtype=float)
    assert np.all(sigma_arr > 0), "All sigma values must be positive"
    return beta_arr / sigma_arr


def project_parallel(z0: float, sigma: float) -> float:
    """
    Project Z^0 onto the range of sigma in 1D.

    In 1D, this is the identity: Z^{0,parallel} = Z^0.

    Parameters
    ----------
    z0 : float
        Z^0 component.
    sigma : float
        Asset volatility.

    Returns
    -------
    float
        Z^{0,parallel} = Z^0 (1D identity projection).
    """
    # In 1D, sigma^T (sigma sigma^T)^{-1} sigma = sigma / sigma^2 * sigma = 1
    return float(z0)


def project_perp(z0: float, sigma: float) -> float:
    """
    Perpendicular component Z^{0,perp} = Z^0 - Z^{0,parallel}.

    In 1D, this is always zero.

    Parameters
    ----------
    z0 : float
    sigma : float

    Returns
    -------
    float
        Z^{0,perp} = 0 in 1D.
    """
    return 0.0


def holdings_from_exposure(p: float, sigma: float) -> float:
    """
    Convert optimal exposure p to stock holdings pi.

    pi* = (sigma sigma^T)^{-1} sigma p = p / sigma in 1D.

    Parameters
    ----------
    p : float
        Optimal exposure.
    sigma : float
        Asset volatility.

    Returns
    -------
    float
        Optimal holdings pi*.
    """
    assert sigma > 0
    return p / sigma


def holdings_from_exposure_array(p_arr: np.ndarray, sigma_arr: np.ndarray) -> np.ndarray:
    """Vectorized version of holdings_from_exposure."""
    return np.asarray(p_arr) / np.asarray(sigma_arr)


def check_nondegeneracy(sigma: float, c_sigma: float = 0.01) -> bool:
    """
    Verify that sigma sigma^T >= c_sigma * I (nondegeneracy).

    In 1D, this requires sigma^2 >= c_sigma.

    Parameters
    ----------
    sigma : float
    c_sigma : float
        Nondegeneracy lower bound.

    Returns
    -------
    bool
    """
    return sigma ** 2 >= c_sigma


# ──────────────────────────────────────────────────────────────────────────────
# Aliases for test compatibility
# ──────────────────────────────────────────────────────────────────────────────

def eta_from_sigma_beta(sigma: float, beta: float) -> float:
    """
    Alias for compute_eta: eta = beta / sigma in 1D.

    Parameters
    ----------
    sigma : float
        Positive scalar volatility.
    beta : float
        Jump size (> -1).

    Returns
    -------
    float
    """
    return compute_eta(sigma, beta)


def check_sigma_nondeg(sigma: float, lower: float = 0.01) -> bool:
    """
    Alias for check_nondegeneracy with named 'lower' parameter.

    Parameters
    ----------
    sigma : float
    lower : float
        Nondegeneracy lower bound (default 0.01).

    Returns
    -------
    bool
    """
    return check_nondegeneracy(sigma, c_sigma=lower)
