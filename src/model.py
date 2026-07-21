"""
Model specification for the defaultable MFG with exponential utility.

Implements an admissible Markovian model with:
- Common factor X^0 (1D)
- Idiosyncratic noise W^1 (1D, d=1)
- One risky asset (n=1, d0=1)
- Default intensity lambda(t, x, j) with lambda^{(1)}=0
- Bounded sigma, beta, mu coefficients
- Terminal liabilities phi^{(0)}, phi^{(1)}
"""
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np


@dataclass
class ModelParams:
    """
    Admissible Markovian model parameters for the defaultable BSDE/MFG system.

    All coefficient functions map (t, x) -> scalar, where x = X^0.
    Regime j in {0,1}: j=0 pre-default, j=1 post-default.
    """
    # Time horizon
    T: float = 0.5

    # Dimensions: n=1 risky asset, d0=1 common BM, d=1 idiosyncratic BM
    n: int = 1
    d0: int = 1
    d: int = 1

    # Common factor X^0 dynamics: dX^0 = mu_x dt + sigma_x dW^0
    mu_x: float = 0.1          # drift of common factor
    sigma_x: float = 0.3       # diffusion of common factor

    # Risky asset coefficients (scalar in 1D, bounded by assumption)
    # sigma_stock: risky asset volatility w.r.t. common BM W^0
    sigma_stock: float = 0.4   # sigma(t,x,j): bounded, uniformly nondegenerate
    # mu_stock: expected return (used if theta not externally fixed)
    mu_stock: float = 0.08

    # Jump size beta(t,x,j): beta > -1 required
    beta: float = -0.3         # negative: loss at default

    # Default intensity parameters (pre-default only)
    lambda_bar: float = 1.0    # maximum intensity
    lambda_scale: float = 0.5  # baseline intensity scaling

    # Risk aversion: single agent (exp_1) or distribution (exp_2-4)
    gamma: float = 2.0         # individual risk aversion coefficient

    # Distribution of gamma (for MFG): sample from uniform [gamma_lo, gamma_hi]
    gamma_lo: float = 1.0
    gamma_hi: float = 3.0
    n_gamma_samples: int = 100  # quadrature / MC samples for E[1/gamma]

    # Exogenous risk premium theta (for exp_1: fixed bounded process)
    theta_fixed: float = 0.15  # scalar theta in 1D

    # Initial wealth
    xi: float = 1.0

    # Spatial grid for PDE solver
    x_min: float = -2.0
    x_max: float = 2.0
    n_x: int = 60

    # Time grid
    n_t: int = 100

    # Solver tolerances
    tol_q: float = 1e-12       # q-root residual tolerance
    tol_picard: float = 1e-6   # BSDE Picard iteration tolerance
    tol_inner_fp: float = 1e-8  # Inner fixed point (psi) tolerance
    tol_outer_fp: float = 1e-5  # Outer fixed point (Gamma) tolerance
    max_iter_picard: int = 50
    max_iter_inner: int = 200
    max_iter_outer: int = 30

    # Terminal liability functions (Markovian): phi^{(j)}(x)
    # Use lambda functions; override for experiments
    phi_0: Optional[Callable] = None  # pre-default liability
    phi_1: Optional[Callable] = None  # post-default liability

    def __post_init__(self) -> None:
        """Set default terminal liabilities if not provided."""
        if self.phi_0 is None:
            # Smooth bounded terminal liability depending on common factor
            self.phi_0 = lambda x: 0.1 * np.tanh(x)
        if self.phi_1 is None:
            # Post-default liability (can differ to test liability sensitivity)
            self.phi_1 = lambda x: 0.05 * np.tanh(x) - 0.02

    def validate(self) -> None:
        """Validate admissibility conditions."""
        assert self.T > 0, "Horizon must be positive"
        assert self.sigma_stock > 0, "sigma_stock must be positive (nondegeneracy)"
        assert self.beta > -1, "beta > -1 required for compensated martingale"
        assert self.lambda_bar >= 0, "lambda_bar must be non-negative"
        assert self.gamma > 0, "Risk aversion must be positive"
        assert self.gamma_lo > 0 and self.gamma_hi >= self.gamma_lo, "Invalid gamma distribution"
        assert self.x_min < self.x_max, "Invalid spatial grid"

    def lambda_func(self, t: float, x: float, j: int) -> float:
        """
        Default intensity lambda(t, x, j).
        lambda^{(1)} = 0 (no more defaults after first default).
        Pre-default: lambda = lambda_bar * lambda_scale * sigmoid(x)
        bounded in [0, lambda_bar].
        """
        if j == 1:
            return 0.0
        val = self.lambda_bar * self.lambda_scale * (1.0 / (1.0 + np.exp(-x)))
        return float(np.clip(val, 0.0, self.lambda_bar))

    def sigma_func(self, t: float, x: float, j: int) -> float:
        """Asset volatility sigma(t,x,j): bounded, nondegenerate scalar."""
        return self.sigma_stock

    def beta_func(self, t: float, x: float, j: int) -> float:
        """Jump size beta(t,x,j) with beta > -1."""
        return self.beta

    def mu_func(self, t: float, x: float, j: int) -> float:
        """Expected return mu(t,x,j)."""
        return self.mu_stock

    def eta_func(self, t: float, x: float, j: int) -> float:
        """
        eta = sigma^T (sigma sigma^T)^{-1} beta = beta / sigma in 1D.
        Gives the component of default jump in common-noise direction.
        """
        sig = self.sigma_func(t, x, j)
        bet = self.beta_func(t, x, j)
        return bet / sig

    def theta_exog(self, t: float, x: float, j: int) -> float:
        """
        Exogenous risk premium (for exp_1).
        Fixed bounded scalar theta in 1D.
        """
        return self.theta_fixed

    def ghat(self, gamma_samples: Optional[np.ndarray] = None) -> float:
        """
        Harmonic mean coefficient ghat = E[1/gamma]^{-1}.
        Computed from gamma distribution samples.
        """
        if gamma_samples is None:
            gammas = np.linspace(self.gamma_lo, self.gamma_hi, self.n_gamma_samples)
        else:
            gammas = gamma_samples
        return 1.0 / np.mean(1.0 / gammas)

    def gamma_samples_uniform(self) -> np.ndarray:
        """Return quadrature nodes for gamma ~ Uniform[gamma_lo, gamma_hi]."""
        return np.linspace(self.gamma_lo, self.gamma_hi, self.n_gamma_samples)


def make_scenario(
    T: float = 0.5,
    beta: float = -0.3,
    lambda_bar: float = 1.0,
    lambda_scale: float = 0.5,
    gamma: float = 2.0,
    gamma_lo: float = 1.0,
    gamma_hi: float = 3.0,
    sigma_stock: float = 0.4,
    theta_fixed: float = 0.15,
    phi_0: Optional[Callable] = None,
    phi_1: Optional[Callable] = None,
    n_x: int = 60,
    n_t: int = 100,
) -> ModelParams:
    """
    Factory function to create a ModelParams scenario with custom parameters.
    Used by comparative statics experiments.
    """
    p = ModelParams(
        T=T,
        beta=beta,
        lambda_bar=lambda_bar,
        lambda_scale=lambda_scale,
        gamma=gamma,
        gamma_lo=gamma_lo,
        gamma_hi=gamma_hi,
        sigma_stock=sigma_stock,
        theta_fixed=theta_fixed,
        phi_0=phi_0,
        phi_1=phi_1,
        n_x=n_x,
        n_t=n_t,
    )
    p.validate()
    return p
