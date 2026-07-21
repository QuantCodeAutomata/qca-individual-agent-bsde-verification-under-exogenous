"""
Experiment 4: Large-Population Asymptotic Market-Clearing Verification.

Verifies the paper's large-population claim by:
1. Solving the MFG equilibrium premium theta^{mfg}
2. Simulating N heterogeneous agents under the MFG premium
3. Computing finite-population market-clearing imbalance
4. Estimating decay rate with N (expected: O(1/N))
5. Log-log slope diagnostics

E[int_0^T |(1/N) sum_i pi_t^{i,*}|^2 dt] ~ C/N
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from typing import Dict, List, Optional
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ModelParams, make_scenario
from src.mfg_solver import solve_mfg_equilibrium
from src.qsolver import solve_q_array
from src.simulation import make_time_grid, simulate_default_time, compute_default_indicator
from src.projection import holdings_from_exposure

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def solve_baseline_equilibrium(params: ModelParams) -> Dict:
    """Solve MFG equilibrium and return callable control functions."""
    params.n_gamma_samples = 10
    params.max_iter_outer = 15
    params.max_iter_inner = 40
    params.tol_outer_fp = 1e-4
    params.tol_inner_fp = 1e-5
    return solve_mfg_equilibrium(params)


def interpolate_path_on_grid(
    mfg_sol: Dict,
    times: np.ndarray,
    x_path: np.ndarray,
) -> tuple:
    """
    Vectorised interpolation of MFG solution along a given (t, x) path.

    Interpolates (Z^0, U, theta^{mfg}) for all time steps at once using
    bilinear interpolation on the (time, x) grid, avoiding per-step Python loops.

    Parameters
    ----------
    mfg_sol : dict
        Solved MFG equilibrium.
    times : np.ndarray, shape (n_t+1,)
        Time grid.
    x_path : np.ndarray, shape (n_t+1,)
        Common-factor path at each time step.

    Returns
    -------
    z0_path, U_path, theta_path : each np.ndarray of shape (n_t+1,)
    """
    grid_times = mfg_sol["times"]
    x_grid = mfg_sol["x_grid"]
    n_t1 = len(times)

    # Clamp
    t_clamped = np.clip(times, grid_times[0], grid_times[-1])
    x_clamped = np.clip(x_path, x_grid[0], x_grid[-1])

    # Time indices
    k_arr = np.searchsorted(grid_times, t_clamped, side="right") - 1
    k_arr = np.clip(k_arr, 0, len(grid_times) - 2)
    t_frac = (t_clamped - grid_times[k_arr]) / (
        grid_times[k_arr + 1] - grid_times[k_arr] + 1e-15
    )

    # Space indices
    xi_arr = np.searchsorted(x_grid, x_clamped, side="right") - 1
    xi_arr = np.clip(xi_arr, 0, len(x_grid) - 2)
    x_frac = (x_clamped - x_grid[xi_arr]) / (
        x_grid[xi_arr + 1] - x_grid[xi_arr] + 1e-15
    )

    def bilinear_path(arr: np.ndarray) -> np.ndarray:
        """arr has shape (n_t_grid+1, n_x_grid)."""
        v00 = arr[k_arr, xi_arr]
        v01 = arr[k_arr, xi_arr + 1]
        v10 = arr[k_arr + 1, xi_arr]
        v11 = arr[k_arr + 1, xi_arr + 1]
        return (1 - t_frac) * ((1 - x_frac) * v00 + x_frac * v01) + \
               t_frac * ((1 - x_frac) * v10 + x_frac * v11)

    z0_path = bilinear_path(mfg_sol["z0"])
    U_path = bilinear_path(mfg_sol["U_grid"])
    theta_path = bilinear_path(mfg_sol["theta_mfg"])

    return z0_path, U_path, theta_path


def _build_common_path(
    params: ModelParams,
    times: np.ndarray,
    seed_common: int,
) -> tuple:
    """Simulate common factor path and default indicator for one scenario."""
    n_t = len(times) - 1
    dt = times[1] - times[0]
    rng_c = np.random.default_rng(seed_common)
    dW0 = rng_c.standard_normal(n_t) * np.sqrt(dt)
    x_path = np.zeros(n_t + 1)
    for k in range(n_t):
        x_path[k + 1] = x_path[k] + params.mu_x * dt + params.sigma_x * dW0[k]
    tau = simulate_default_time(params, times, x_path, rng=rng_c)
    N_ind = compute_default_indicator(times, tau)
    return x_path, N_ind


def simulate_pi_matrix(
    gamma_arr: np.ndarray,
    z0_path: np.ndarray,
    U_path: np.ndarray,
    theta_path: np.ndarray,
    lam_path: np.ndarray,
    eta_path: np.ndarray,
    sigma_path: np.ndarray,
    N_ind: np.ndarray,
) -> np.ndarray:
    """
    Compute the (N, n_t+1) matrix of individual optimal holdings π^{i,*}.

    p_t^{i,*} = Z_t^0 + θ_t/γ_i + (λ_t/γ_i)·η_t·exp(−q_t^{i,*})
    q_t^{i,*} solves: q − a_t·exp(−q) = γ_i·Z_t^0·η_t − γ_i·U_t + θ_t·η_t

    Parameters
    ----------
    gamma_arr : np.ndarray, shape (N,)
    z0_path, U_path, theta_path, lam_path, eta_path, sigma_path : shape (n_t+1,)
    N_ind : np.ndarray, shape (n_t+1,) – default indicator

    Returns
    -------
    pi_mat : np.ndarray, shape (N, n_t+1)
    """
    N = len(gamma_arr)
    n_t1 = len(z0_path)

    g = gamma_arr[:, None]       # (N, 1)
    z0 = z0_path[None, :]        # (1, n_t+1)
    U = U_path[None, :]
    th = theta_path[None, :]
    lam = lam_path[None, :]
    eta = eta_path[None, :]
    sig = sigma_path[None, :]

    r_mat = g * z0 * eta - g * U + th * eta          # (N, n_t+1)
    a_mat = (lam * eta ** 2)                          # (1, n_t+1)

    a_flat = np.broadcast_to(a_mat, (N, n_t1)).ravel()
    q_flat, _, _ = solve_q_array(a_flat, r_mat.ravel())
    q_mat = q_flat.reshape(N, n_t1)
    exp_neg_q = np.exp(np.clip(-q_mat, -200, 200))

    active = (lam > 1e-10) & (N_ind[None, :] == 0)  # (1, n_t+1)
    p_mat = z0 + th / g + np.where(active, (lam / g) * eta * exp_neg_q, 0.0)
    return np.where(sig > 1e-10, p_mat / sig, 0.0)   # (N, n_t+1)


def simulate_population(
    N: int,
    params: ModelParams,
    mfg_sol: Dict,
    seed_common: int,
    seed_gamma: int,
) -> np.ndarray:
    """
    Simulate N agents' average imbalance under the MFG premium.

    Returns (1/N) Σ π_t^{i,*} of shape (n_t+1,).
    """
    times = mfg_sol["times"]
    x_path, N_ind = _build_common_path(params, times, seed_common)
    z0_path, U_path, theta_path = interpolate_path_on_grid(mfg_sol, times, x_path)
    lam_path = np.array([params.lambda_func(times[k], x_path[k], int(N_ind[k])) for k in range(len(times))])
    eta_path = np.array([params.eta_func(times[k], x_path[k], int(N_ind[k])) for k in range(len(times))])
    sigma_path = np.array([params.sigma_func(times[k], x_path[k], int(N_ind[k])) for k in range(len(times))])
    gamma_arr = np.random.default_rng(seed_gamma).uniform(params.gamma_lo, params.gamma_hi, size=N)
    pi_mat = simulate_pi_matrix(gamma_arr, z0_path, U_path, theta_path, lam_path, eta_path, sigma_path, N_ind)
    return pi_mat.mean(axis=0)   # (n_t+1,)


def estimate_clearing_error(
    N_grid: List[int],
    R: int,
    params: ModelParams,
    mfg_sol: Dict,
    seed_base: int = 42,
    N_oracle: int = 2000,
) -> Dict:
    """
    Estimate the raw and idiosyncratic clearing-error metrics for all N in N_grid.

    For each MC replication r we:
      1. Draw one common factor path (CRN: same seed for all N).
      2. Estimate the conditional mean E[π|X0] using N_oracle large sample (oracle).
      3. For each N, draw N agents and compute:
         - raw E_N   = ∫ |(1/N)Σπ^i|² dt
         - centered  = ∫ |(1/N)Σπ^i − oracle_mean|² dt   (isolates 1/N term)
         - var-based = ∫ Var(π^i|X0) dt / N               (analytic 1/N bound)

    Parameters
    ----------
    N_grid : list of int
    R      : number of MC replications
    params : ModelParams
    mfg_sol : solved equilibrium dict
    seed_base : int
    N_oracle : int – agents used per replication to estimate E[π|X0]

    Returns
    -------
    dict with keys matching N values; each value is a dict with
    'raw', 'centered', 'var_based' arrays of length R, plus summary stats.
    """
    times = mfg_sol["times"]
    dt = times[1] - times[0]

    raw_lists = {N: [] for N in N_grid}
    cen_lists = {N: [] for N in N_grid}
    var_lists = {N: [] for N in N_grid}

    for r in range(R):
        seed_common = seed_base + r * 1000

        # Build common path once per replication (CRN across all N)
        x_path, N_ind = _build_common_path(params, times, seed_common)
        z0_path, U_path, theta_path = interpolate_path_on_grid(mfg_sol, times, x_path)
        lam_path = np.array([params.lambda_func(times[k], x_path[k], int(N_ind[k])) for k in range(len(times))])
        eta_path = np.array([params.eta_func(times[k], x_path[k], int(N_ind[k])) for k in range(len(times))])
        sigma_path = np.array([params.sigma_func(times[k], x_path[k], int(N_ind[k])) for k in range(len(times))])

        # Oracle: large sample to estimate E[π | X0, tau]
        gamma_oracle = np.random.default_rng(seed_base + r * 1000 + 9999).uniform(
            params.gamma_lo, params.gamma_hi, size=N_oracle)
        pi_oracle = simulate_pi_matrix(gamma_oracle, z0_path, U_path, theta_path,
                                       lam_path, eta_path, sigma_path, N_ind)
        oracle_mean = pi_oracle.mean(axis=0)          # (n_t+1,)
        oracle_var  = pi_oracle.var(axis=0, ddof=1)   # (n_t+1,) per-agent variance

        for N in N_grid:
            gamma_N = np.random.default_rng(seed_base + r * 1000 + N).uniform(
                params.gamma_lo, params.gamma_hi, size=N)
            pi_N = simulate_pi_matrix(gamma_N, z0_path, U_path, theta_path,
                                      lam_path, eta_path, sigma_path, N_ind)
            imb = pi_N.mean(axis=0)                    # (n_t+1,)

            raw_lists[N].append(float(np.sum(imb ** 2) * dt))
            cen_lists[N].append(float(np.sum((imb - oracle_mean) ** 2) * dt))
            var_lists[N].append(float(np.sum(oracle_var) * dt / N))

    results = {}
    for N in N_grid:
        raw = np.array(raw_lists[N])
        cen = np.array(cen_lists[N])
        vbased = np.array(var_lists[N])
        results[N] = {
            "raw_mean": raw.mean(), "raw_se": raw.std(ddof=1) / np.sqrt(R),
            "cen_mean": cen.mean(), "cen_se": cen.std(ddof=1) / np.sqrt(R),
            "var_mean": vbased.mean(), "var_se": vbased.std(ddof=1) / np.sqrt(R),
            "raw_list": raw, "cen_list": cen,
        }
    return results


def _loglog_fit(N_arr: np.ndarray, E_arr: np.ndarray) -> tuple:
    """Fit log(E) = a + b*log(N) and return (slope, intercept, R²)."""
    log_N = np.log(N_arr.astype(float))
    log_E = np.log(np.maximum(E_arr, 1e-16))
    slope, intercept, r_value, _, _ = scipy_stats.linregress(log_N, log_E)
    return slope, intercept, r_value ** 2


def run_exp4_large_population() -> dict:
    """
    Large-population asymptotic market-clearing verification.

    Methodology (faithful to paper):
      1. Solve MFG equilibrium once.
      2. For R replications, fix one common-factor path (CRN across N).
      3. Per replication, draw N_oracle agents to estimate E[π|X0] (oracle mean).
      4. For each N in grid, draw N agents; compute:
         - raw metric   : ∫|(1/N)Σπ^i|² dt
         - centered     : ∫|(1/N)Σπ^i − oracle_mean|² dt  (idiosyncratic; ~C/N)
         - variance-est : ∫Var(π^i|X0) dt / N             (analytic bound)
      5. Log-log regression on centered metric — expected slope ≈ −1.
    """
    print("=" * 60)
    print("EXP 4: Large-Population Asymptotic Market-Clearing Verification")
    print("=" * 60)

    params = make_scenario(
        T=0.3,
        beta=-0.3,
        lambda_bar=1.0,
        lambda_scale=0.5,
        gamma=2.0,
        gamma_lo=1.0,
        gamma_hi=3.0,
        sigma_stock=0.4,
        n_x=30,
        n_t=60,
        phi_0=lambda x: 0.1 * np.tanh(x),
        phi_1=lambda x: 0.05 * np.tanh(x) - 0.02,
    )

    print("\nSolving MFG equilibrium...")
    mfg_sol = solve_baseline_equilibrium(params)
    print(f"  Converged: {mfg_sol['outer_converged']} "
          f"in {mfg_sol['outer_iterations']} iterations")

    N_grid = [10, 20, 50, 100, 200, 500]
    R = 100        # MC replications (CRN; same common path for all N)
    N_oracle = 2000

    print(f"\nRunning {R} CRN replications, N_oracle={N_oracle}, N_grid={N_grid}...")
    ce = estimate_clearing_error(N_grid, R, params, mfg_sol, seed_base=42, N_oracle=N_oracle)

    results_list = []
    for N in N_grid:
        r = ce[N]
        results_list.append({
            "N": N,
            "raw_mean": r["raw_mean"],  "raw_se": r["raw_se"],
            "cen_mean": r["cen_mean"],  "cen_se": r["cen_se"],
            "var_mean": r["var_mean"],  "var_se": r["var_se"],
        })
        print(f"  N={N:4d}: raw={r['raw_mean']:.6f}±{1.96*r['raw_se']:.6f}  "
              f"centered={r['cen_mean']:.6f}±{1.96*r['cen_se']:.6f}  "
              f"var_est={r['var_mean']:.6f}")

    df = pd.DataFrame(results_list)
    df.to_csv(os.path.join(RESULTS_DIR, "exp4_clearing_error.csv"), index=False)

    # Log-log regression on the CENTERED metric (isolates 1/N decay)
    N_arr = np.array(N_grid, dtype=float)
    cen_arr = np.array([ce[N]["cen_mean"] for N in N_grid])
    raw_arr = np.array([ce[N]["raw_mean"] for N in N_grid])
    var_arr = np.array([ce[N]["var_mean"] for N in N_grid])

    slope_cen, int_cen, r2_cen = _loglog_fit(N_arr, cen_arr)
    slope_raw, int_raw, r2_raw = _loglog_fit(N_arr, raw_arr)
    slope_var, int_var, r2_var = _loglog_fit(N_arr, var_arr)

    print(f"\nLog-log slopes (expected −1.0):")
    print(f"  Centered metric: slope={slope_cen:.3f}, R²={r2_cen:.3f}")
    print(f"  Raw metric:      slope={slope_raw:.3f}, R²={r2_raw:.3f}")
    print(f"  Variance-based:  slope={slope_var:.3f}, R²={r2_var:.3f}")

    # Robustness: two beta scenarios
    print("\nRobustness checks:")
    robustness = []
    for bv, lb in [(-0.3, 1.0), (0.3, 1.0)]:
        p_rb = make_scenario(T=0.3, beta=bv, lambda_bar=lb, sigma_stock=0.4,
                             gamma=2.0, gamma_lo=1.0, gamma_hi=3.0,
                             n_x=30, n_t=60,
                             phi_0=lambda x: 0.1 * np.tanh(x),
                             phi_1=lambda x: 0.05 * np.tanh(x) - 0.02)
        mfg_rb = solve_baseline_equilibrium(p_rb)
        ce_rb = estimate_clearing_error([10, 100], 50, p_rb, mfg_rb, seed_base=99, N_oracle=2000)
        ratio_raw = ce_rb[10]["raw_mean"] / (ce_rb[100]["raw_mean"] + 1e-15)
        ratio_cen = ce_rb[10]["cen_mean"] / (ce_rb[100]["cen_mean"] + 1e-15)
        robustness.append({
            "beta": bv, "lambda_bar": lb,
            "raw_N10": ce_rb[10]["raw_mean"], "raw_N100": ce_rb[100]["raw_mean"],
            "cen_N10": ce_rb[10]["cen_mean"], "cen_N100": ce_rb[100]["cen_mean"],
            "ratio_raw": ratio_raw, "ratio_cen": ratio_cen,
        })
        print(f"  β={bv:+.1f}, λ={lb}: cen(N=10)={ce_rb[10]['cen_mean']:.6f}, "
              f"cen(N=100)={ce_rb[100]['cen_mean']:.6f}, ratio={ratio_cen:.1f} (target~10)")

    df_rb = pd.DataFrame(robustness)
    df_rb.to_csv(os.path.join(RESULTS_DIR, "exp4_robustness.csv"), index=False)

    _plot_exp4(df, slope_cen, int_cen, r2_cen,
               slope_raw, int_raw, r2_raw, df_rb)

    print("\nExp 4 complete.")
    return {
        "N_grid": N_grid,
        "results": results_list,
        "slope_centered": slope_cen,
        "r2_centered": r2_cen,
        "slope_raw": slope_raw,
        "r2_raw": r2_raw,
        "robustness": robustness,
    }


def _plot_exp4(df, sl_cen, int_cen, r2_cen, sl_raw, int_raw, r2_raw, df_rb):
    """Generate plots for Exp 4 with three-metric comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Exp 4: Large-Population Asymptotic Market-Clearing Verification",
                 fontsize=12, fontweight="bold")

    N_vals = df["N"].values

    # (a) Log-log: centered vs raw metrics
    ax = axes[0]
    ax.loglog(N_vals, df["cen_mean"], "bo-", label="Centered (idiosyncratic)", lw=2)
    ax.loglog(N_vals, df["raw_mean"], "gs--", label="Raw total", lw=1.5, alpha=0.7)
    ax.loglog(N_vals, df["var_mean"], "r^:", label="Variance-based bound", lw=1.5)
    # Fitted line for centered
    N_fit = np.array([N_vals[0], N_vals[-1]], dtype=float)
    ax.loglog(N_fit, np.exp(int_cen + sl_cen * np.log(N_fit)), "b--",
              label=f"Fit slope={sl_cen:.2f}", lw=1, alpha=0.6)
    # Ideal 1/N reference anchored to centered
    C_ref = df["cen_mean"].iloc[0] * N_vals[0]
    ax.loglog(N_fit, C_ref / N_fit, "k:", lw=1.5, label="slope=−1 reference")
    ax.set_xlabel("Population size N")
    ax.set_ylabel("Clearing error (log scale)")
    ax.set_title("Log-Log: Three Metrics")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

    # (b) Centered metric with 95% CI
    ax = axes[1]
    ax.errorbar(N_vals, df["cen_mean"], yerr=1.96 * df["cen_se"],
                fmt="bo-", capsize=5, label="Centered E_N ± 1.96 SE")
    C_ref2 = df["cen_mean"].iloc[0] * N_vals[0]
    N_lin = np.linspace(N_vals[0], N_vals[-1], 200)
    ax.plot(N_lin, C_ref2 / N_lin, "r--", label="C/N reference")
    ax.set_xlabel("Population size N")
    ax.set_ylabel("Centered clearing error")
    ax.set_title(f"Centered metric (slope≈{sl_cen:.2f}, R²={r2_cen:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Robustness: centered N=10 vs N=100
    ax = axes[2]
    labels = [f"β={row['beta']:+.1f}\nλ={row['lambda_bar']}" for _, row in df_rb.iterrows()]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, df_rb["cen_N10"], 0.35, label="N=10", color="steelblue")
    ax.bar(x + 0.2, df_rb["cen_N100"], 0.35, label="N=100", color="orange")
    for i, row in df_rb.iterrows():
        ax.text(i, max(row["cen_N10"], row["cen_N100"]) * 1.02,
                f"×{row['ratio_cen']:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, ha="center")
    ax.set_ylabel("Centered clearing error")
    ax.set_title("Robustness: Centered E_N by Scenario\n(ratio≈10 expected)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "exp4_main.png")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Plots saved → {out}")


if __name__ == "__main__":
    run_exp4_large_population()
