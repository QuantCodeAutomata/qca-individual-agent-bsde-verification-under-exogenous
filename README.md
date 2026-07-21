# MFG BSDE Defaultable Market Model

Numerical verification of individual-agent and mean-field-game (MFG) equilibrium
results for an incomplete defaultable-market model with exponential utility agents.
The model contains a single default jump, common and idiosyncratic Brownian noise,
and a bounded risk-aversion distribution across a continuum of heterogeneous agents.

All experiments use **synthetic admissible data** generated from a concrete
low-dimensional Markovian specification; no empirical financial data are required.

---

## Repository Structure

```
repo/
├── data/               # Placeholder (synthetic data generated at runtime)
├── src/                # Core library modules
│   ├── model.py        # Model coefficients: σ, β, λ, η, μ
│   ├── simulation.py   # Brownian paths, default time, wealth process
│   ├── projection.py   # Projection Π_t(z), admissible-exposure decomposition
│   ├── qsolver.py      # Scalar root solver: q − a·exp(−q) = r
│   ├── bsde_individual.py  # Single-agent BSDE solver and control recovery
│   ├── mfg_solver.py   # Nested fixed-point MFG equilibrium solver
│   └── diagnostics.py  # Residuals, optimality checks, convergence plots
├── exp/                # Experiment entry-points
│   ├── exp1_individual_bsde.py   # Exp 1 – individual BSDE verification
│   ├── exp2_mfg_equilibrium.py   # Exp 2 – MFG equilibrium construction
│   ├── exp3_comparative_statics.py  # Exp 3 – comparative statics
│   └── exp4_large_population.py  # Exp 4 – large-N market clearing
├── tests/              # pytest test suite (95 tests)
│   ├── test_model.py
│   ├── test_projection.py
│   ├── test_qsolver.py
│   └── test_experiments.py
├── results/            # Output figures, tables, and RESULTS.md
│   └── RESULTS.md
└── README.md
```

---

## Scientific Background

The model is a mean-field game of portfolio choice in an incomplete market with
one risky asset, a common Brownian factor W⁰, an idiosyncratic factor W¹, and
a single default time τ with stochastic intensity λ_t.

Each agent solves:

> **maximize** E[−exp(−γ (W_T − F))]

subject to a budget constraint driven by the risky-asset price process.
The optimal control satisfies a **quadratic-growth BSDE** with a compensated
default martingale dM_t = dN_t − λ_t dt.  The equilibrium premium θ^{mfg} is
characterised by a **nested scalar fixed point** involving:

- Inner map S(ψ): ψ = E[exp(−q\*(ψ))/γ] where q\* solves q − λ|η|²e^{−q} = r(ψ)
- Outer map Γ(v): PDE/BSDE backward system for pre-/post-default regimes

The equilibrium premium decomposes as:

> θ^{mfg} = −ĝ Ē[Z^{0‖}]ᵀ − ĝ λ η ψ\*

where ĝ = E[1/γ]^{−1} is the harmonic-mean risk-tolerance.

---

## Installation

```bash
pip install numpy scipy pandas matplotlib numba scikit-learn pytest
```

All dependencies are standard and pre-installed in the provided Docker image.

---

## Running Experiments

```bash
# Run individual experiments
python -m exp.exp1_individual_bsde
python -m exp.exp2_mfg_equilibrium
python -m exp.exp3_comparative_statics
python -m exp.exp4_large_population

# Run the full test suite
pytest tests/ -v
```

Outputs (figures and CSV tables) are written to `results/`.

---

## Key Results

| Experiment | Primary finding |
|---|---|
| 1 – Individual BSDE | q-residual < 1e-12; optimality verified; default term vanishes post-τ |
| 2 – MFG equilibrium | Nested FP converges in 3 iters; clearing residual 3e-08; β<0 ⟹ premium > 0 |
| 3 – Comparative statics | Default premium ∝ λ; sign flips with β; heterogeneity via ĝ and ψ\* |
| 4 – Large-population | Centered clearing error slope −1.014 (R²=0.988), confirming 1/N theorem |

See `results/RESULTS.md` for detailed tables and diagnostics.

---

## Citation / Methodology

Custom implementation — Context7 confirmed no library equivalent for this specific
defaultable quadratic BSDE with compensated default martingale and nested MFG
fixed point (Paper Sec. 3–5).

Scalar root solver uses `scipy.optimize.brentq` with Newton initialisation.
