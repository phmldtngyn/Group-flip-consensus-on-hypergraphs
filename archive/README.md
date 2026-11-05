# Archive — superseded material (kept for traceability)

Nothing in this folder is used by the current manuscript. Files were moved
here, not deleted, during the November 2026 revision audit so that every
number that appeared in earlier drafts can still be traced.

## `v1_v2_analysis_superseded/`

| File | What it is | Why superseded |
|---|---|---|
| `bootstrap_v1_iid.csv` | v1 analysis: 500 iid bootstrap replicates (γ, family-norm reduction relative to the 1D reference) | Generated from a data version that is **not** in this repository: neither its point estimates nor its intervals can be reproduced from `data/processed/measurements_v2.csv`. |
| `regression_v1.json` | v1 coefficients (γ̂ = −0.017867, intercept 0.4804, …) | Not reproducible from the released data with WLS, OLS or cell means (see `docs/audit_report.md`, §2.1). |
| `summary_v2.json` | summary of the v1 iid bootstrap | as above |
| `v2_*.csv`, `v2_regression_summary.json` | v2 analysis (two-level cluster bootstrap, three-contrast family norm) produced by `src/analysis/regression_bootstrap.py` | Three defects: (i) the reported γ = −0.0165 is the bootstrap **mean**, not the point estimate, and is for the specification without `C(n)`; (ii) the "reference-invariant" family norm uses only the three contrasts among 2D/WS/config and is neither reference-invariant nor sensitive to the 1D family; (iii) the unstratified cluster bootstrap drops at least one family in ~12–13% of replicates. Replaced by `src/analysis/reanalysis_v2.py`. |
| `newborn_test_v2_DEFECTIVE.csv` | Newborn-identity test on the V2 truncated sets, produced by the original `newborn_extinction_test` | That function declared a "birth" when the state *after* a fire was monochrome (an erasure), so it never recorded a success: 0 successes in all 72 rows, including the 27 cells with c̃ > 0. Fixed in `src/simulation/simulator_numba.py`; regenerated as `data/processed/newborn_test_v2_corrected.csv` by `scripts/rerun_newborn_v2.py`. |

## `figures_superseded/` and `visualization_scripts_superseded/`

| File | Problem |
|---|---|
| `fig3_containment.png` (from `fig3.py`) | **The "brute-force enumeration" markers were the closed-form values plus Gaussian noise** (`np.random.normal`), not an enumeration. Replaced by a figure built from `tests/independent/containment_bruteforce.py` output. |
| `fig4_falsification.png` (from `fig4.py`) | **Panel (a) plotted synthetic data**: `M_mock = 0.4804 + 0.4999 q + family offset` with error bars drawn from `np.random.uniform(0.004, 0.008)`. Panels (b)–(c) showed the v1 iid bootstrap while the caption described the cluster bootstrap. Replaced by a figure built from the measurements and the re-analysis outputs. |
| `fig2_exact_band.png` (from `fig2.py`) | Plotted 3 of the 6 exact cases. Replaced (all six, exact rational values). |
| `fig2_measurements.png`, `fig3_residuals.png`, `bootstrap.png` (from `make_fig23.py`) | Exploratory plots not used in the manuscript; read from a `results_v2/` path that does not exist in the repo. |
| `fig5_fragmentation.png` | Schematic with almost no content; removed from the manuscript (the mechanism it illustrates is not established). |

## `logs_superseded/`

Run logs from the author's machine (UTF-16, Windows paths). Kept as provenance.
