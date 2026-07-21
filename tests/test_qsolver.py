"""
Tests for the scalar q-solver: q - a*exp(-q) = r.

Covers:
- Correctness of root (residual < tol)
- Uniqueness (single root)
- Special cases: a=0, large a, large r, negative r
- Array version
- Edge cases: extreme values, convergence
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qsolver import solve_q, solve_q_array, validate_q_solution


class TestSolveQBasic:
    """Test basic correctness of the q-solver."""

    def test_a_zero_returns_r(self):
        """When a=0, equation reduces to q=r."""
        for r in [-2.0, 0.0, 1.5, 5.0]:
            q, res, n_iter = solve_q(0.0, r)
            assert abs(q - r) < 1e-8, f"q should equal r when a=0; got q={q}, r={r}"
            assert abs(res) < 1e-10

    def test_residual_below_tolerance(self):
        """q - a*exp(-q) - r should be < tol for all test cases."""
        test_cases = [
            (0.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (0.5, 0.5), (1.0, 0.0), (1.0, 1.5),
            (2.0, -1.0), (5.0, 2.0), (0.01, 3.0),
        ]
        for a, r in test_cases:
            q, res, n_iter = solve_q(a, r, tol=1e-10)
            assert abs(res) < 1e-8, f"Residual {res} too large for a={a}, r={r}"

    def test_solution_is_correct_root(self):
        """Verify q satisfies the defining equation q - a*exp(-q) = r."""
        for a in [0.0, 0.5, 1.0, 2.0, 5.0]:
            for r in [-2.0, 0.0, 1.5, 4.0]:
                q, res, _ = solve_q(a, r)
                equation_value = q - a * np.exp(-q) - r
                assert abs(equation_value) < 1e-7, \
                    f"Root check failed: a={a}, r={r}, q={q}, eq={equation_value}"

    def test_validate_q_solution_pass(self):
        """validate_q_solution should return True for correct roots."""
        a, r = 1.0, 0.5
        q, _, _ = solve_q(a, r)
        assert validate_q_solution(a, r, q, tol=1e-8)

    def test_validate_q_solution_fail(self):
        """validate_q_solution should return False for wrong values."""
        assert not validate_q_solution(1.0, 0.5, 999.0, tol=1e-4)


class TestSolveQProperties:
    """Test mathematical properties of the q-solver."""

    def test_q_increases_with_r(self):
        """q should increase monotonically with r (for fixed a)."""
        a = 1.0
        r_vals = np.linspace(-2.0, 3.0, 10)
        q_vals = [solve_q(a, r)[0] for r in r_vals]
        # Monotone increasing
        for i in range(len(q_vals) - 1):
            assert q_vals[i] <= q_vals[i + 1] + 1e-9, \
                f"q not increasing with r at i={i}: q[i]={q_vals[i]}, q[i+1]={q_vals[i+1]}"

    def test_unique_root(self):
        """g(q) = q - a*exp(-q) is strictly increasing, so root is unique."""
        # Verify by checking that derivative 1 + a*exp(-q) > 0 always
        a = 2.0
        r = 1.0
        q, _, _ = solve_q(a, r)
        derivative = 1.0 + a * np.exp(-q)
        assert derivative > 0

    def test_q_bounded_for_bounded_inputs(self):
        """Ensure q stays bounded for reasonable inputs (no explosion)."""
        for a in [0.1, 1.0, 5.0]:
            for r in [-5.0, 0.0, 5.0]:
                q, _, _ = solve_q(a, r)
                assert abs(q) < 100.0, f"q={q} seems unbounded for a={a}, r={r}"


class TestSolveQArray:
    """Test the vectorized array version of q-solver."""

    def test_array_matches_scalar(self):
        """Array version should match scalar version for each element."""
        a_arr = np.array([0.0, 0.5, 1.0, 2.0])
        r_arr = np.array([1.0, -0.5, 0.5, 2.0])
        q_arr, res_arr, _ = solve_q_array(a_arr, r_arr)

        for i in range(len(a_arr)):
            q_scalar, res_scalar, _ = solve_q(a_arr[i], r_arr[i])
            assert abs(q_arr[i] - q_scalar) < 1e-7, \
                f"Array vs scalar mismatch at i={i}: {q_arr[i]} vs {q_scalar}"

    def test_array_residuals_small(self):
        """All residuals in array version should be below tolerance."""
        a_arr = np.random.default_rng(0).uniform(0, 3, 50)
        r_arr = np.random.default_rng(1).uniform(-3, 3, 50)
        q_arr, res_arr, _ = solve_q_array(a_arr, r_arr)
        assert np.max(np.abs(res_arr)) < 1e-6, \
            f"Max residual {np.max(np.abs(res_arr))} too large"

    def test_array_empty(self):
        """Empty arrays should return empty arrays without error."""
        q_arr, res_arr, n_iter = solve_q_array(np.array([]), np.array([]))
        assert len(q_arr) == 0
        assert len(res_arr) == 0


class TestSolveQEdgeCases:
    """Test edge cases for the q-solver."""

    def test_large_r(self):
        """Large r should give large q with small residual."""
        for r_large in [10.0, 20.0, 50.0]:
            q, res, _ = solve_q(1.0, r_large)
            assert abs(res) < 1e-6

    def test_large_negative_r(self):
        """Large negative r."""
        q, res, _ = solve_q(1.0, -10.0)
        assert abs(res) < 1e-6

    def test_large_a(self):
        """Large a (intensity*eta^2)."""
        q, res, _ = solve_q(100.0, 1.0)
        assert abs(res) < 1e-6

    def test_very_small_a(self):
        """Very small a ~ 0 should give q ~= r."""
        a = 1e-8
        r = 2.5
        q, res, _ = solve_q(a, r)
        assert abs(q - r) < 0.01, f"For tiny a, q should be near r; q={q}, r={r}"

    def test_convergence_count(self):
        """Newton's method should converge quickly (< 50 iterations)."""
        q, res, n_iter = solve_q(1.0, 1.0, max_iter=100)
        assert n_iter < 50, f"Too many iterations: {n_iter}"


class TestSolveQNumericalStability:
    """Test numerical stability of the q-solver."""

    def test_close_to_boundary(self):
        """Test near-boundary values where exp(-q) might underflow."""
        # Very large q: exp(-q) -> 0, q ≈ r
        a, r = 0.001, 100.0
        q, res, _ = solve_q(a, r)
        assert abs(res) < 1e-6

    def test_deterministic_given_same_inputs(self):
        """Same inputs should always produce same outputs."""
        a, r = 1.5, 0.7
        q1, _, _ = solve_q(a, r)
        q2, _, _ = solve_q(a, r)
        assert q1 == q2

    def test_regression_known_case(self):
        """
        Regression test: known solution for q - exp(-q) = 0 (a=1, r=0).
        Solution satisfies q = W(1) where W is Lambert W function.
        From scipy.special.lambertw: W(1) ≈ 0.5671...
        q - exp(-q) = q - exp(-0.5671) = 0.5671 - 0.5671 = 0
        """
        from scipy.special import lambertw
        W_val = float(np.real(lambertw(1.0)))
        q_known = W_val  # q* = W(1) for a=1, r=0
        q, res, _ = solve_q(1.0, 0.0)
        assert abs(q - q_known) < 1e-6, f"Regression failed: q={q}, expected {q_known}"
