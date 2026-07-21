"""
Tests for ModelParams and simulation utilities.

Covers:
- Model admissibility checks (beta > -1, lambda >= 0, sigma > 0)
- Derived quantities (eta, ghat)
- Simulation correctness (time grid, default indicator)
- Wealth process
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ModelParams, make_scenario
from src.simulation import (
    simulate_one_path,
    simulate_default_time,
    compute_default_indicator,
    make_time_grid,
    simulate_wealth,
    compute_utility,
)


class TestModelParams:
    """Tests for ModelParams creation and admissibility."""

    def test_make_scenario_default(self):
        """make_scenario should return a valid ModelParams without errors."""
        p = make_scenario()
        assert p.T > 0
        assert p.gamma > 0
        assert p.sigma_stock > 0
        assert p.lambda_bar >= 0

    def test_beta_above_minus_one(self):
        """beta must satisfy beta > -1 (admissibility)."""
        p = make_scenario(beta=0.3)
        assert p.beta > -1.0

    def test_ghat_positive(self):
        """Harmonic mean ghat = E[1/gamma]^{-1} should be positive."""
        p = make_scenario(gamma=2.0, gamma_lo=1.0, gamma_hi=3.0)
        ghat = p.ghat()
        assert ghat > 0

    def test_ghat_bounded_by_harmonic(self):
        """ghat should equal gamma when distribution is degenerate."""
        # When gamma_lo = gamma_hi = gamma, harmonic mean = gamma
        p = make_scenario(gamma=2.0, gamma_lo=2.0 - 1e-4, gamma_hi=2.0 + 1e-4)
        ghat = p.ghat()
        assert abs(ghat - 2.0) < 0.1, f"ghat={ghat} should be ~2.0 for degenerate distribution"

    def test_ghat_less_than_or_equal_arithmetic_mean(self):
        """Harmonic mean <= arithmetic mean (Jensen's inequality)."""
        p = make_scenario(gamma=2.0, gamma_lo=1.0, gamma_hi=3.0)
        ghat = p.ghat()
        gamma_mean = (p.gamma_lo + p.gamma_hi) / 2.0
        assert ghat <= gamma_mean + 1e-10, \
            f"Harmonic mean {ghat} should <= arithmetic mean {gamma_mean}"

    def test_eta_consistent_with_beta_sigma(self):
        """eta = beta / sigma (1D case)."""
        p = make_scenario(beta=-0.3, sigma_stock=0.4)
        eta = p.eta_func(0.0, 0.0, 0)
        expected = -0.3 / 0.4
        assert abs(eta - expected) < 1e-10

    def test_lambda_post_default_is_zero(self):
        """After default (j=1), lambda must be 0."""
        p = make_scenario(lambda_bar=1.0, lambda_scale=0.5)
        lam_post = p.lambda_func(0.1, 0.0, j=1)
        assert lam_post == 0.0, "Post-default lambda must be 0"

    def test_lambda_pre_default_nonneg(self):
        """Pre-default lambda(t,x,0) should be non-negative."""
        p = make_scenario(lambda_bar=2.0, lambda_scale=1.0)
        for t in [0.0, 0.1, 0.3]:
            for x in [-1.0, 0.0, 1.0]:
                lam = p.lambda_func(t, x, j=0)
                assert lam >= 0.0

    def test_phi_callable(self):
        """phi_0 and phi_1 should be callable."""
        p = make_scenario()
        x_arr = np.linspace(-1, 1, 10)
        out0 = p.phi_0(x_arr)
        out1 = p.phi_1(x_arr)
        assert len(out0) == 10
        assert len(out1) == 10


class TestTimeGrid:
    """Tests for make_time_grid."""

    def test_time_grid_length(self):
        """Time grid should have n_t+1 points from 0 to T."""
        times = make_time_grid(T=1.0, n_t=100)
        assert len(times) == 101

    def test_time_grid_starts_at_zero(self):
        times = make_time_grid(T=0.5, n_t=50)
        assert times[0] == 0.0

    def test_time_grid_ends_at_T(self):
        T = 0.5
        times = make_time_grid(T=T, n_t=50)
        assert abs(times[-1] - T) < 1e-12

    def test_time_grid_uniform(self):
        """Grid should be uniformly spaced."""
        times = make_time_grid(T=1.0, n_t=100)
        dts = np.diff(times)
        assert np.allclose(dts, dts[0]), "Time grid should be uniform"


class TestSimulation:
    """Tests for simulation functions."""

    def test_simulate_one_path_keys(self):
        """simulate_one_path should return required keys."""
        p = make_scenario(n_t=50, n_x=20)
        path = simulate_one_path(p, seed_common=0, seed_idio=1, x0_init=0.0)
        for key in ["times", "x_path", "W0", "W1", "N", "tau"]:
            assert key in path, f"Missing key: {key}"

    def test_simulate_path_length(self):
        """Path arrays should have length n_t+1."""
        p = make_scenario(n_t=50, n_x=20)
        path = simulate_one_path(p, seed_common=0, seed_idio=1, x0_init=0.0)
        n_t = p.n_t
        assert len(path["times"]) == n_t + 1
        assert len(path["x_path"]) == n_t + 1
        assert len(path["N"]) == n_t + 1

    def test_default_indicator_nondecreasing(self):
        """N_t = 1_{tau <= t} should be non-decreasing."""
        p = make_scenario(lambda_bar=2.0, lambda_scale=1.0, n_t=100)
        path = simulate_one_path(p, seed_common=42, seed_idio=43, x0_init=0.0)
        N = path["N"]
        assert np.all(np.diff(N) >= 0), "N_t should be non-decreasing"

    def test_default_indicator_binary(self):
        """N_t should be 0 or 1 only."""
        p = make_scenario(n_t=100)
        path = simulate_one_path(p, seed_common=0, seed_idio=1, x0_init=0.0)
        assert set(path["N"].astype(int)).issubset({0, 1})

    def test_compute_default_indicator_before_tau(self):
        """N_t = 0 for t < tau."""
        times = np.linspace(0, 1, 101)
        tau = 0.5
        N = compute_default_indicator(times, tau)
        assert np.all(N[times < tau] == 0)

    def test_compute_default_indicator_after_tau(self):
        """N_t = 1 for t >= tau."""
        times = np.linspace(0, 1, 101)
        tau = 0.5
        N = compute_default_indicator(times, tau)
        assert np.all(N[times >= tau] == 1)

    def test_default_time_positive(self):
        """Default time tau should be positive."""
        p = make_scenario(lambda_bar=2.0, n_t=100)
        path = simulate_one_path(p, seed_common=10, seed_idio=11, x0_init=0.0)
        assert path["tau"] >= 0.0

    def test_no_default_when_lambda_zero(self):
        """When lambda_bar=0, there should be no default within finite time."""
        p = make_scenario(lambda_bar=0.0, lambda_scale=0.0, T=1.0, n_t=100)
        rng = np.random.default_rng(0)
        times = make_time_grid(p.T, p.n_t)
        x_path = np.zeros(p.n_t + 1)
        tau = simulate_default_time(p, times, x_path, rng=rng)
        assert tau > p.T, f"No default expected when lambda=0; tau={tau}"

    def test_simulate_wealth_length(self):
        """Wealth process should have length n_t+1."""
        p = make_scenario(n_t=50)
        path = simulate_one_path(p, seed_common=0, seed_idio=1, x0_init=0.0)
        p_ctrl = np.zeros(p.n_t + 1)  # zero control
        W = simulate_wealth(p, path, p_ctrl, xi=0.0)
        assert len(W) == p.n_t + 1

    def test_zero_control_constant_wealth(self):
        """Zero control (no trading) should give zero additional wealth."""
        p = make_scenario(n_t=50, T=1.0)
        path = simulate_one_path(p, seed_common=0, seed_idio=1, x0_init=0.0)
        p_ctrl = np.zeros(p.n_t + 1)
        W = simulate_wealth(p, path, p_ctrl, xi=0.0)
        # With zero control, wealth stays at xi=0
        assert abs(W[0]) < 1e-12


class TestComputeUtility:
    """Tests for utility computation."""

    def test_utility_negative(self):
        """Exponential utility -exp(-gamma*X) should be negative."""
        u = compute_utility(gamma=1.0, W_T=1.0, F_T=0.0)
        assert u < 0

    def test_utility_increases_with_wealth(self):
        """More wealth -> higher utility."""
        u1 = compute_utility(gamma=1.0, W_T=1.0, F_T=0.0)
        u2 = compute_utility(gamma=1.0, W_T=2.0, F_T=0.0)
        assert u2 > u1

    def test_utility_at_zero_surplus(self):
        """U(-exp(-gamma * 0)) = -exp(0) = -1."""
        u = compute_utility(gamma=1.0, W_T=0.0, F_T=0.0)
        assert abs(u - (-1.0)) < 1e-10

    def test_utility_range(self):
        """Utility -exp(-gamma * X) is always < 0."""
        for W in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            u = compute_utility(gamma=2.0, W_T=W, F_T=0.0)
            assert u < 0
