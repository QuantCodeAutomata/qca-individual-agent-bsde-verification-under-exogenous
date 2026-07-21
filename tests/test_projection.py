"""
Tests for projection utilities and BSDE/MFG solver modules.

Covers:
- Projection operator Pi_t idempotency and boundedness
- eta computation
- Individual BSDE solver correctness
- MFG equilibrium existence and properties
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.projection import (
    project_parallel,
    project_perp,
    eta_from_sigma_beta,
    holdings_from_exposure,
    check_sigma_nondeg,
)
from src.model import make_scenario
from src.bsde_individual import solve_individual_bsde, recover_post_default_control
from src.mfg_solver import solve_mfg_equilibrium


class TestProjection:
    """Tests for projection utilities."""

    def test_project_parallel_idempotent(self):
        """Pi_t(Pi_t(z)) = Pi_t(z) (idempotency)."""
        sigma = 0.4
        z = 1.5
        z_par1 = project_parallel(z, sigma)
        z_par2 = project_parallel(z_par1, sigma)
        assert abs(z_par1 - z_par2) < 1e-12

    def test_project_parallel_recovers_z_for_1d(self):
        """In 1D, z is always in the sigma subspace, so z_par = z."""
        sigma = 0.3
        z = 2.1
        z_par = project_parallel(z, sigma)
        assert abs(z_par - z) < 1e-12, \
            f"In 1D, projection should be identity; z={z}, z_par={z_par}"

    def test_project_perp_zero_in_1d(self):
        """In 1D, z_perp = z - z_par = 0."""
        sigma = 0.3
        z = 2.1
        z_perp = project_perp(z, sigma)
        assert abs(z_perp) < 1e-12

    def test_perp_plus_parallel_equals_z(self):
        """z_par + z_perp = z."""
        sigma = 0.4
        z = 1.7
        z_par = project_parallel(z, sigma)
        z_perp = project_perp(z, sigma)
        assert abs(z_par + z_perp - z) < 1e-12

    def test_eta_formula(self):
        """eta = sigma^T (sigma sigma^T)^{-1} beta = beta/sigma in 1D."""
        sigma = 0.4
        beta = -0.3
        eta = eta_from_sigma_beta(sigma, beta)
        expected = beta / sigma
        assert abs(eta - expected) < 1e-10

    def test_eta_sign_consistent_with_beta(self):
        """eta should have the same sign as beta in 1D."""
        for beta in [-0.5, -0.1, 0.0, 0.2, 0.6]:
            eta = eta_from_sigma_beta(0.4, beta)
            assert np.sign(eta) == np.sign(beta), \
                f"eta sign {np.sign(eta)} != beta sign {np.sign(beta)}"

    def test_holdings_from_exposure_1d(self):
        """pi = p/sigma in 1D."""
        sigma = 0.4
        p = 1.2
        pi = holdings_from_exposure(p, sigma)
        assert abs(pi - p / sigma) < 1e-12

    def test_check_sigma_nondeg_pass(self):
        """Positive sigma should pass nondegeneracy check."""
        assert check_sigma_nondeg(0.4, lower=0.01)

    def test_check_sigma_nondeg_fail(self):
        """Zero sigma should fail."""
        assert not check_sigma_nondeg(0.0, lower=0.01)


class TestIndividualBSDE:
    """Tests for the individual BSDE solver."""

    @pytest.fixture(scope="class")
    def small_model_sol(self):
        """Solve a small model for testing."""
        params = make_scenario(
            T=0.2, beta=-0.2, lambda_bar=0.5, lambda_scale=0.3,
            gamma=2.0, sigma_stock=0.4, theta_fixed=0.1,
            n_x=20, n_t=40,
            phi_0=lambda x: 0.05 * np.tanh(x),
            phi_1=lambda x: 0.02 * np.tanh(x),
        )
        sol = solve_individual_bsde(params)
        return params, sol

    def test_bsde_solution_keys(self, small_model_sol):
        """BSDE solution should contain required keys."""
        _, sol = small_model_sol
        for key in ["w0", "w1", "z0_pre", "z0_post", "U_grid", "q_star", "q_residuals",
                    "p_star_pre", "pi_star_pre", "times", "x_grid"]:
            assert key in sol, f"Missing key: {key}"

    def test_bsde_terminal_condition_pre(self, small_model_sol):
        """w0(T, x) = phi_0(x) (terminal condition pre-default)."""
        params, sol = small_model_sol
        w0_T = sol["w0"][-1]   # Last time point
        x_grid = sol["x_grid"]
        phi0_vals = params.phi_0(x_grid)
        assert np.allclose(w0_T, phi0_vals, atol=1e-4), \
            f"Terminal condition not satisfied: max error = {np.max(np.abs(w0_T - phi0_vals))}"

    def test_bsde_terminal_condition_post(self, small_model_sol):
        """w1(T, x) = phi_1(x) (terminal condition post-default)."""
        params, sol = small_model_sol
        w1_T = sol["w1"][-1]
        x_grid = sol["x_grid"]
        phi1_vals = params.phi_1(x_grid)
        assert np.allclose(w1_T, phi1_vals, atol=1e-4), \
            f"Terminal condition not satisfied: max error = {np.max(np.abs(w1_T - phi1_vals))}"

    def test_bsde_finite_values(self, small_model_sol):
        """All solution components should be finite."""
        _, sol = small_model_sol
        for key in ["w0", "w1", "z0_pre", "z0_post", "U_grid", "q_star"]:
            assert np.all(np.isfinite(sol[key])), f"{key} contains non-finite values"

    def test_q_residuals_small(self, small_model_sol):
        """q-root residuals should be below tolerance."""
        _, sol = small_model_sol
        max_res = np.max(np.abs(sol["q_residuals"]))
        assert max_res < 1e-4, f"q-root residual {max_res} too large"

    def test_post_default_control_has_no_default_term(self, small_model_sol):
        """Post-default control p* (lambda=0) should have no default jump term."""
        params, sol = small_model_sol
        p_post, pi_post = recover_post_default_control(sol, params)

        # Post-default: p* = z0_post + theta/gamma (no lambda term)
        # Verify by construction that p_post = z0_post + theta/gamma
        times = sol["times"]
        x_grid = sol["x_grid"]
        n_t, n_x = len(times) - 1, len(x_grid)

        for k in range(n_t + 1):
            for i in range(n_x):
                t_k = times[k]
                x_i = x_grid[i]
                theta = params.theta_exog(t_k, x_i, j=1)  # post-default theta
                expected_p = sol["z0_post"][k, i] + theta / params.gamma
                assert abs(p_post[k, i] - expected_p) < 1e-10, \
                    f"Post-default control mismatch at k={k}, i={i}"

    def test_pre_default_q_star_finite(self, small_model_sol):
        """Pre-default q* should be finite everywhere."""
        _, sol = small_model_sol
        assert np.all(np.isfinite(sol["q_star"]))

    def test_p_star_pi_consistent_1d(self, small_model_sol):
        """pi* = p*/sigma in 1D (no matrix inversion needed)."""
        params, sol = small_model_sol
        sigma = params.sigma_stock
        ratio = sol["p_star_pre"] / (sol["pi_star_pre"] + 1e-15)
        # ratio should be sigma everywhere
        assert np.allclose(ratio, sigma, atol=1e-8)


class TestMFGSolver:
    """Tests for the MFG equilibrium solver."""

    @pytest.fixture(scope="class")
    def small_mfg_sol(self):
        """Solve a small MFG equilibrium for testing."""
        params = make_scenario(
            T=0.2, beta=-0.3, lambda_bar=0.5, lambda_scale=0.3,
            gamma=2.0, gamma_lo=1.0, gamma_hi=3.0,
            sigma_stock=0.4, n_x=20, n_t=40,
            phi_0=lambda x: 0.05 * np.tanh(x),
            phi_1=lambda x: 0.02 * np.tanh(x),
        )
        params.n_gamma_samples = 10
        params.max_iter_outer = 15
        params.max_iter_inner = 60
        params.tol_outer_fp = 1e-3
        params.tol_inner_fp = 1e-5
        mfg_sol = solve_mfg_equilibrium(params)
        return params, mfg_sol

    def test_mfg_solution_keys(self, small_mfg_sol):
        """MFG solution should contain required keys."""
        _, mfg_sol = small_mfg_sol
        for key in ["w0", "w1", "z0", "U_grid", "psi_star", "theta_mfg",
                    "theta_brownian", "theta_default", "clearing_residual",
                    "outer_converged", "outer_iterations"]:
            assert key in mfg_sol, f"Missing key: {key}"

    def test_psi_star_positive(self, small_mfg_sol):
        """psi* should be positive everywhere."""
        _, mfg_sol = small_mfg_sol
        assert np.all(mfg_sol["psi_star"] > 0), \
            f"psi* not positive; min={np.min(mfg_sol['psi_star'])}"

    def test_outer_fp_residual_decreasing(self, small_mfg_sol):
        """Outer FP residuals should decrease (convergence indicator)."""
        _, mfg_sol = small_mfg_sol
        residuals = mfg_sol["outer_residuals"]
        if len(residuals) > 2:
            # At least some decrease
            assert residuals[-1] <= residuals[0] + 1e-6, \
                "Outer FP residuals should generally decrease"

    def test_theta_mfg_finite(self, small_mfg_sol):
        """theta^{mfg} should be finite."""
        _, mfg_sol = small_mfg_sol
        assert np.all(np.isfinite(mfg_sol["theta_mfg"]))

    def test_clearing_residual_small(self, small_mfg_sol):
        """Market clearing residual should be small."""
        _, mfg_sol = small_mfg_sol
        max_clearing = np.max(np.abs(mfg_sol["clearing_residual"]))
        # Should be reasonable for small model
        assert max_clearing < 5.0, f"Clearing residual {max_clearing} too large"

    def test_premium_decomposition_sums(self, small_mfg_sol):
        """theta_mfg ≈ theta_brownian + theta_default."""
        _, mfg_sol = small_mfg_sol
        diff = mfg_sol["theta_mfg"] - mfg_sol["theta_brownian"] - mfg_sol["theta_default"]
        assert np.max(np.abs(diff)) < 1e-8, \
            f"Decomposition mismatch: max diff = {np.max(np.abs(diff))}"

    def test_ghat_stored_correctly(self, small_mfg_sol):
        """ghat stored in mfg_sol should match params.ghat()."""
        params, mfg_sol = small_mfg_sol
        assert abs(mfg_sol["ghat"] - params.ghat()) < 1e-10
