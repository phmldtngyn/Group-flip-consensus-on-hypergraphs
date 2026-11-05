# Data

## `processed/measurements_v2.csv`

Output of `src/simulation/measure_pipeline.py` (V2 protocol, `configs/v2.yaml`).
216 rows, one per simulation run. This file is the input of the regression test
(main text Sec. 6, SM Protocol F.4) and is never overwritten by the pipeline.

| Column | Meaning |
|---|---|
| `family` | `1D` (cycle), `2D` (torus, king adjacency), `config` (configuration model, k = 6, projected to a simple graph), `WS` (Watts–Strogatz, k = 6, p = 0.1) |
| `n` | number of nodes: 100, 144, 400 (one graph per family and `n`, graph seed 42) |
| `q` | **nominal** coverage on the grid 0.55, …, 0.80 |
| `beta` | interaction-set size, `round(q n)`; realised coverage is `beta/n` |
| `cG`, `ctilde` | containment observables, exact enumeration on this run's truncated interaction sets |
| `M` | time-averaged majority fraction over 1.5×10⁵ fires after 5×10⁴ burn-in |
| `se` | batch-means Monte Carlo standard error of `M` (75 batches of 2000 fires) |
| `seed` | run seed (1, 2, 3); used for the dynamics **and** for breaking ties in the last BFS layer of the truncated sets |

Because of the last point, `ctilde` differs between the three runs of a cell in
15 of the 72 cells.

Check on `se`: within a cell, the standard deviation of `M` across the three runs
divided by the mean `se` has median 0.832 over the 72 cells; if the batch-means
standard errors are correct, the expected median for three runs is
√(ln 2) ≈ 0.833.

## `processed/newborn_test_v2_corrected.csv`

Newborn-identity Monte Carlo test (SM Prop. F.1) on the same truncated sets
(interaction sets of run seed 1; 4×10⁵ fires per cell), one row per
(family, n, q): trials, successes, empirical rate, enumerated `ctilde`, and
`z = (empirical − ctilde)/binomial SE`. Produced by `scripts/rerun_newborn_v2.py`
with the corrected `newborn_extinction_test`; the file released earlier
(`newborn_test_v2.csv`, 0 successes in every row) came from a defective version
of that function and is archived.

## `raw/`

Not tracked. `python scripts/run_all.py --with-measurements` writes a fresh
re-run of the V2 simulations to `data/raw/v2_rerun/` for comparison.

## Superseded data

The v1 bootstrap replicates and v1 coefficients were produced from an earlier data
version that is not in this repository; they are kept in
`archive/v1_v2_analysis_superseded/` for traceability only.
