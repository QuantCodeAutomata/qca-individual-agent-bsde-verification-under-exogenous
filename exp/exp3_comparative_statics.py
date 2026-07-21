"""
Experiment 3: Economic Comparative Statics of Equilibrium Risk Premium.

Studies how the equilibrium premium decomposition responds to:
(i)   Default intensity lambda
(ii)  Jump size beta (sign and magnitude)
(iii) Gamma heterogeneity (ghat)
(iv)  Terminal liability structure
(v)   Horizon T

Each scenario independently solves the full MFG equilibrium.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ModelParams, make_scenario
from src.mfg_solver import solve_mfg_equilibrium
from src.diagnostics import mfg_diagnostics

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Baseline parameters (small T for guaranteed convergence)
BASELINE = dict(
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


def solve_scenario_summary(params: ModelParams) -> dict:
    """Solve MFG and return summary statistics."""
    params.n_gamma_samples = 10   # reduced for speed; sufficient for convergence
    params.max_iter_outer = 15
    params.max_iter_inner = 40
    params.tol_outer_fp = 1e-4
    params.tol_inner_fp = 1e-5

    mfg_sol = solve_mfg_equilibrium(params)
    diag = mfg_diagnostics(mfg_sol, params)

    return {
        "theta_total_mean": float(np.mean(mfg_sol["theta_mfg"])),
        "theta_brownian_mean": float(np.mean(mfg_sol["theta_brownian"])),
        "theta_default_mean": float(np.mean(mfg_sol["theta_default"])),
        "psi_star_mean": float(np.mean(mfg_sol["psi_star"])),
        "clearing_max": diag["clearing_max"],
        "outer_converged": mfg_sol["outer_converged"],
        "outer_iterations": mfg_sol["outer_iterations"],
        "ghat": mfg_sol["ghat"],
        "sign_ok": diag["sign_prediction_ok"],
    }


def run_lambda_cs() -> pd.DataFrame:
    """Comparative statics: vary lambda (intensity level)."""
    print("\n--- CS (i): Lambda (intensity) ---")
    lambda_bars = [0.2, 0.5, 1.0, 2.5]
    rows = []
    for lb in lambda_bars:
        kw = {**BASELINE, "lambda_bar": lb}
        params = make_scenario(**kw)
        row = solve_scenario_summary(params)
        row["lambda_bar"] = lb
        rows.append(row)
        print(f"  lambda_bar={lb:.1f}: theta_default={row['theta_default_mean']:.4f}, "
              f"psi*={row['psi_star_mean']:.4f}, converged={row['outer_converged']}")
    return pd.DataFrame(rows)


def run_beta_cs() -> pd.DataFrame:
    """Comparative statics: vary beta (sign and magnitude)."""
    print("\n--- CS (ii): Beta (jump size) ---")
    beta_vals = [-0.5, -0.2, 0.0, 0.2, 0.5]
    rows = []
    for bv in beta_vals:
        if bv <= -1.0:
            continue
        kw = {**BASELINE, "beta": bv}
        params = make_scenario(**kw)
        row = solve_scenario_summary(params)
        row["beta"] = bv
        row["eta"] = bv / BASELINE["sigma_stock"]
        rows.append(row)
        sign = "+" if row["theta_default_mean"] > 0 else "-"
        print(f"  beta={bv:.1f}: theta_default={row['theta_default_mean']:.4f} "
              f"({sign}), sign_ok={row['sign_ok']}")
    return pd.DataFrame(rows)


def run_gamma_hetero_cs() -> pd.DataFrame:
    """Comparative statics: vary gamma heterogeneity."""
    print("\n--- CS (iii): Gamma heterogeneity ---")
    # Vary width of gamma distribution [1, 1+w] with same lower bound
    gamma_configs = [
        (2.0, 2.0),    # homogeneous: degenerate at 2.0
        (1.0, 3.0),    # moderate heterogeneity
        (0.5, 4.0),    # wide heterogeneity
    ]
    rows = []
    for glo, ghi in gamma_configs:
        if glo == ghi:
            # Degenerate case: homogeneous gamma
            kw = {**BASELINE, "gamma_lo": glo - 0.001, "gamma_hi": ghi + 0.001, "gamma": glo}
        else:
            kw = {**BASELINE, "gamma_lo": glo, "gamma_hi": ghi, "gamma": (glo + ghi) / 2.0}
        params = make_scenario(**kw)
        ghat = params.ghat()
        row = solve_scenario_summary(params)
        row["gamma_lo"] = glo
        row["gamma_hi"] = ghi
        row["ghat"] = ghat
        rows.append(row)
        print(f"  [gamma_lo={glo}, gamma_hi={ghi}]: ghat={ghat:.4f}, "
              f"theta_total={row['theta_total_mean']:.4f}, "
              f"psi*={row['psi_star_mean']:.4f}")
    return pd.DataFrame(rows)


def run_liability_cs() -> pd.DataFrame:
    """Comparative statics: vary terminal liability structure."""
    print("\n--- CS (iv): Terminal liability ---")
    liability_configs = [
        ("no_regime_jump",
         lambda x: 0.1 * np.tanh(x),
         lambda x: 0.1 * np.tanh(x)),         # identical pre/post
        ("baseline",
         lambda x: 0.1 * np.tanh(x),
         lambda x: 0.05 * np.tanh(x) - 0.02), # moderate shift
        ("large_default_shift",
         lambda x: 0.2 * np.tanh(x),
         lambda x: -0.1 * np.tanh(x) - 0.05), # large shift
    ]
    rows = []
    for name, phi0, phi1 in liability_configs:
        kw = {**BASELINE, "phi_0": phi0, "phi_1": phi1}
        params = make_scenario(**kw)
        row = solve_scenario_summary(params)
        row["liability_case"] = name
        rows.append(row)
        print(f"  {name}: theta_total={row['theta_total_mean']:.4f}, "
              f"theta_brownian={row['theta_brownian_mean']:.4f}, "
              f"theta_default={row['theta_default_mean']:.4f}")
    return pd.DataFrame(rows)


def run_horizon_cs() -> pd.DataFrame:
    """Comparative statics: vary horizon T."""
    print("\n--- CS (v): Horizon T ---")
    T_vals = [0.1, 0.2, 0.3, 0.5]
    rows = []
    for Tv in T_vals:
        kw = {**BASELINE, "T": Tv}
        params = make_scenario(**kw)
        row = solve_scenario_summary(params)
        row["T"] = Tv
        rows.append(row)
        print(f"  T={Tv:.1f}: theta_total={row['theta_total_mean']:.4f}, "
              f"converged={row['outer_converged']}, "
              f"clearing_max={row['clearing_max']:.4e}")
    return pd.DataFrame(rows)


def run_exp3_comparative_statics() -> dict:
    """Run all comparative statics scenarios."""
    print("=" * 60)
    print("EXP 3: Economic Comparative Statics")
    print("=" * 60)

    # Run all CS
    df_lambda = run_lambda_cs()
    df_beta = run_beta_cs()
    df_gamma = run_gamma_hetero_cs()
    df_liability = run_liability_cs()
    df_T = run_horizon_cs()

    # Save tables
    df_lambda.to_csv(os.path.join(RESULTS_DIR, "exp3_lambda_cs.csv"), index=False)
    df_beta.to_csv(os.path.join(RESULTS_DIR, "exp3_beta_cs.csv"), index=False)
    df_gamma.to_csv(os.path.join(RESULTS_DIR, "exp3_gamma_cs.csv"), index=False)
    df_liability.to_csv(os.path.join(RESULTS_DIR, "exp3_liability_cs.csv"), index=False)
    df_T.to_csv(os.path.join(RESULTS_DIR, "exp3_T_cs.csv"), index=False)

    # Plots
    _plot_exp3(df_lambda, df_beta, df_gamma, df_liability, df_T)

    print("\nExp 3 complete.")

    results = {
        "lambda_cs": df_lambda.to_dict(orient="records"),
        "beta_cs": df_beta.to_dict(orient="records"),
        "gamma_cs": df_gamma.to_dict(orient="records"),
        "liability_cs": df_liability.to_dict(orient="records"),
        "T_cs": df_T.to_dict(orient="records"),
    }
    return results


def _plot_exp3(df_lambda, df_beta, df_gamma, df_liability, df_T):
    """Generate comparative statics plots for Exp 3."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Exp 3: Comparative Statics of Equilibrium Premium", fontsize=14)

    # (a) Lambda CS: theta_default vs lambda
    ax = axes[0, 0]
    ax.plot(df_lambda["lambda_bar"], df_lambda["theta_default_mean"], "o-", color="red")
    ax.set_xlabel("Intensity lambda_bar")
    ax.set_ylabel("Mean default premium theta_default")
    ax.set_title("(i) Default Premium vs Intensity")
    ax.grid(True, alpha=0.3)

    # (b) Beta CS: theta_default vs beta
    ax = axes[0, 1]
    df_b = df_beta.sort_values("beta")
    ax.plot(df_b["beta"], df_b["theta_default_mean"], "o-")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Beta (jump size)")
    ax.set_ylabel("Mean default premium theta_default")
    ax.set_title("(ii) Default Premium vs Beta Sign")
    ax.grid(True, alpha=0.3)

    # (c) Beta CS: total premium vs |beta|
    ax = axes[0, 2]
    ax.plot(np.abs(df_b["beta"]), np.abs(df_b["theta_default_mean"]), "o-", color="orange")
    ax.set_xlabel("|Beta|")
    ax.set_ylabel("|Mean default premium|")
    ax.set_title("(ii) |Default Premium| vs |Beta|")
    ax.grid(True, alpha=0.3)

    # (d) Gamma CS: total premium vs ghat
    ax = axes[1, 0]
    ax.plot(df_gamma["ghat"], df_gamma["theta_total_mean"], "o-", color="green")
    ax.set_xlabel("ghat = E[1/gamma]^{-1}")
    ax.set_ylabel("Mean total premium")
    ax.set_title("(iii) Total Premium vs ghat")
    ax.grid(True, alpha=0.3)

    # (e) Liability CS: Brownian and default components
    ax = axes[1, 1]
    x = np.arange(len(df_liability))
    width = 0.35
    ax.bar(x - width/2, df_liability["theta_brownian_mean"], width, label="Brownian")
    ax.bar(x + width/2, df_liability["theta_default_mean"], width, label="Default")
    ax.set_xticks(x)
    ax.set_xticklabels(df_liability["liability_case"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Mean premium component")
    ax.set_title("(iv) Premium by Liability Structure")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (f) Horizon CS: total premium vs T
    ax = axes[1, 2]
    ax.plot(df_T["T"], df_T["theta_total_mean"], "o-", label="theta_total")
    ax.plot(df_T["T"], df_T["theta_default_mean"], "s-", label="theta_default")
    ax.plot(df_T["T"], df_T["theta_brownian_mean"], "^-", label="theta_brownian")
    ax.set_xlabel("Horizon T")
    ax.set_ylabel("Mean premium component")
    ax.set_title("(v) Premium vs Horizon T")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_main.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # Heatmap: default premium vs (lambda, beta)
    lambda_grid = [0.5, 1.0, 2.5]
    beta_grid = [-0.5, -0.2, 0.2, 0.5]
    Z = np.zeros((len(lambda_grid), len(beta_grid)))

    print("\n  Computing heatmap (lambda x beta)...")
    for li, lb in enumerate(lambda_grid):
        for bi, bv in enumerate(beta_grid):
            kw = {**BASELINE, "lambda_bar": lb, "beta": bv, "T": 0.2}
            params = make_scenario(**kw)
            try:
                row = solve_scenario_summary(params)
                Z[li, bi] = row["theta_default_mean"]
            except Exception:
                Z[li, bi] = np.nan

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    im = ax2.imshow(Z, aspect="auto", cmap="RdBu_r",
                    extent=[beta_grid[0], beta_grid[-1], lambda_grid[-1], lambda_grid[0]])
    plt.colorbar(im, ax=ax2, label="Mean default premium")
    ax2.set_xlabel("Beta (jump size)")
    ax2.set_ylabel("Lambda_bar (intensity)")
    ax2.set_title("Heatmap: Default Premium vs (Lambda, Beta)")
    ax2.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_heatmap.png"), dpi=100, bbox_inches="tight")
    plt.close()

    print(f"  Plots saved to {RESULTS_DIR}")


if __name__ == "__main__":
    results = run_exp3_comparative_statics()
