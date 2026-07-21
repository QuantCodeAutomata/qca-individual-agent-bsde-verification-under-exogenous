"""
Scalar root solver for the implicit equation q - a * exp(-q) = r.

This is the core scalar equation arising in the individual agent's optimal
exposure formula and the mean-field inner fixed-point computation.

The function Q_a(r) solves: q - a * exp(-q) = r
for given a >= 0 and r in R, with unique solution q in R.

Uniqueness follows because d/dq [q - a*exp(-q)] = 1 + a*exp(-q) > 0.

References: Paper Section on individual optimization, Eq. for q*.
"""
import numpy as np
from typing import Tuple


def solve_q(
    a: float,
    r: float,
    tol: float = 1e-12,
    max_iter: int = 100,
    q0: float = None,
) -> Tuple[float, float, int]:
    """
    Solve the scalar equation q - a * exp(-q) = r via safeguarded Newton.

    Parameters
    ----------
    a : float
        Non-negative coefficient multiplying exp(-q). Typically a = lambda * |eta|^2.
    r : float
        Right-hand side.
    tol : float
        Convergence tolerance on |q - a*exp(-q) - r|.
    max_iter : int
        Maximum Newton iterations.
    q0 : float, optional
        Initial guess. If None, uses robust heuristic.

    Returns
    -------
    q : float
        Solution.
    residual : float
        Final |q - a*exp(-q) - r|.
    n_iter : int
        Number of iterations performed.

    Notes
    -----
    # Custom — Context7 found no library equivalent (paper Eq. for q*)
    When a=0, exact solution q=r.
    When a>0, Newton with positive derivative 1 + a*exp(-q) always converges.
    """
    if a < 0:
        raise ValueError(f"a must be non-negative, got {a}")

    # When a=0: exact solution
    if a == 0.0 or np.isclose(a, 0.0, atol=1e-15):
        q = r
        return q, 0.0, 0

    # Robust initial guess
    if q0 is None:
        if abs(a) < 1e-8:
            q0 = r
        else:
            # log(a + exp(r)) bounds the solution from above
            q0 = float(np.log(a + np.exp(np.clip(r, -50, 50))))

    q = float(q0)

    for i in range(max_iter):
        eq = np.exp(np.clip(-q, -200, 200))
        f_val = q - a * eq - r          # residual: should be 0
        f_der = 1.0 + a * eq            # derivative: always > 0

        delta = f_val / f_der
        q -= delta

        # Check convergence
        eq2 = np.exp(np.clip(-q, -200, 200))
        res = abs(q - a * eq2 - r)
        if res < tol:
            return q, res, i + 1

    res = abs(q - a * np.exp(np.clip(-q, -200, 200)) - r)
    return q, res, max_iter


def solve_q_array(
    a_arr: np.ndarray,
    r_arr: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 100,
    q0_arr: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized version of solve_q over arrays a and r.

    Parameters
    ----------
    a_arr : np.ndarray
        Array of non-negative a values.
    r_arr : np.ndarray
        Array of r values (same shape as a_arr).
    tol : float
        Per-element convergence tolerance.
    max_iter : int
        Maximum iterations.
    q0_arr : np.ndarray, optional
        Initial guesses (same shape).

    Returns
    -------
    q_arr : np.ndarray
        Solution array.
    res_arr : np.ndarray
        Residual array.
    iter_arr : np.ndarray
        Iteration counts.

    Notes
    -----
    # Custom — Context7 found no library equivalent (paper Eq. for q*)
    Uses vectorized Newton's method with shared iterations.
    All elements iterate together; convergence is checked per-element.
    """
    a_arr = np.asarray(a_arr, dtype=float)
    r_arr = np.asarray(r_arr, dtype=float)
    assert a_arr.shape == r_arr.shape, "a_arr and r_arr must have same shape"

    shape = a_arr.shape
    q = np.zeros(shape, dtype=float)

    # Exact solution where a=0
    zero_mask = a_arr < 1e-15
    q[zero_mask] = r_arr[zero_mask]

    # Initial guess for non-zero a
    nz = ~zero_mask
    if q0_arr is not None:
        q[nz] = np.asarray(q0_arr, dtype=float)[nz]
    else:
        small_a = (a_arr < 1e-8) & nz
        large_a = (~small_a) & nz
        q[small_a] = r_arr[small_a]
        # log(a + exp(r)) as initial guess, clipped for numerical safety
        q[large_a] = np.log(
            a_arr[large_a] + np.exp(np.clip(r_arr[large_a], -50, 50))
        )

    res_arr = np.zeros(shape, dtype=float)
    iter_arr = np.zeros(shape, dtype=int)
    converged = zero_mask.copy()

    for i in range(max_iter):
        if converged.all():
            break
        active = ~converged
        eq = np.exp(np.clip(-q[active], -200, 200))
        f_val = q[active] - a_arr[active] * eq - r_arr[active]
        f_der = 1.0 + a_arr[active] * eq
        delta = f_val / f_der
        q[active] -= delta

        # Check convergence
        eq2 = np.exp(np.clip(-q[active], -200, 200))
        res = np.abs(q[active] - a_arr[active] * eq2 - r_arr[active])
        newly_converged = active.copy()
        newly_converged[active] = res < tol
        converged |= newly_converged
        iter_arr[newly_converged] = i + 1

    # Compute final residuals
    eq_f = np.exp(np.clip(-q, -200, 200))
    res_arr = np.abs(q - a_arr * eq_f - r_arr)
    res_arr[zero_mask] = 0.0
    iter_arr[zero_mask] = 0

    return q, res_arr, iter_arr


def validate_q_solution(
    a: float,
    r: float,
    q: float,
    tol: float = 1e-8,
) -> bool:
    """
    Check that q satisfies q - a*exp(-q) = r within tolerance.

    Parameters
    ----------
    a, r, q : float
        Equation parameters and candidate solution.
    tol : float
        Acceptance threshold.

    Returns
    -------
    bool
        True if residual < tol.
    """
    eq = np.exp(np.clip(-q, -200, 200))
    residual = abs(q - a * eq - r)
    return residual < tol
