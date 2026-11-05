# Validation ledger

Every numerical check cited in the manuscript, the command that produces it,
and its evidence file. All rows were executed for this release (Python 3.13.18,
package versions in `requirements-lock.txt`). "Independent" means code written
separately from the production code, sharing no functions with it.

| # | Check | Claimed in | Command | Evidence | Outcome |
|---|---|---|---|---|---|
| A | 1D closed forms `c_G`, `c̃` vs exhaustive enumeration, all `n ∈ [12, 40]`, all `r`, incl. boundary (18, 2) | main §4; SM Props E.1–E.2 | `python tests/independent/containment_bruteforce.py` | `results/validation/containment_bruteforce.json` | **Exact (rational) agreement in all 192 cases** |
| A′ | same, production-code path | — | `python tests/gvm_validation_suite.py --task T2` | `results/validation/suite_rerun/T2.json` | pass |
| B | Region-II exact chain, six cases, independent solver (full configurations, no symmetry reduction) | main Table 1; SM Tables S2–S3 | `python tests/independent/exact_chain_independent.py` | `results/validation/exact_independent.json` | reachable sets single irreducible class; L¹ residual ≤ 4.1×10⁻¹⁴ |
| B′ | same values as **exact rationals** (lumped chain, FLINT, solution verified by exact substitution) | main Table 1 | same | same | 43/57, 65/84, 149/201, 265/339, 72/95, 319/433 |
| B″ | repository solvers agree | SM Protocol F.3 | `python src/exact/exact_chain_1d.py --config configs/exact_grid.yaml --out results/tables/exact_b_crosscheck.json`; suite T5 | `results/tables/exact_b_crosscheck.csv`; `results/validation/suite_rerun/T5.json` | agree to ≤ 10⁻¹³ |
| C | Conjecture 1 in the six cases (`E[M] < f(q)`) | main Conj. 1 | as B′ | as B′ | **holds in exact arithmetic**, all six |
| D | Exact chain = band in Region I | SM §C | suite T1 | `results/validation/suite_rerun/T1.json` | pass (`E[M] = f(q)` to 10⁻¹⁴) |
| E | Simulator vs exact chain, Region II, six cases (native balls) | main §7; SM Protocol F.3(iii) | `python tests/independent/sim_independent.py` | `results/validation/sim_independent.json` | max \|Δ\| = 4.6×10⁻⁵, max \|z\| = 1.65 |
| F | Simulator vs band, Region I (1D n = 400, 401; 2D L = 12, 30; R1, R2 verified by BFS) | main §7 | same; suite T3, T4 | same; `suite_rerun/T3.json`, `T4.json` | max \|Δ\| = 2.1×10⁻⁵, max \|z\| = 0.8 |
| G | Cycle law: `E[K] = (3−2q)/(2−q)`, `CV_T² = (7−8q+2q²)/(3−2q)²` | main App. B, §7; SM Thm C.8 | same as E | same | max \|z\| = 1.95 (E[K]), 1.86 (CV²) |
| H | Newborn identity, native 1D balls | SM Remark F.1.1 | suite T6 | `results/validation/suite_rerun/T6.json` | within binomial error |
| H′ | Newborn identity, V2 truncated sets (72 cells) | SM Remark F.1.1 | `python scripts/rerun_newborn_v2.py` | `data/processed/newborn_test_v2_corrected.csv` | 18 interior cells, max \|z\| = 1.85 |
| I | Regression test, primary + sensitivity | main §6, Table 2; SM Protocol F.4, Table S4 | `python src/analysis/reanalysis_v2.py --B 2000` | `results/tables/reanalysis/` | executed; acceptance criteria not met |
| J | Unit tests | — | `pytest -q` | — | 6 passed |
| K | Manuscript numbers generated from outputs | all | `python scripts/make_numbers_tex.py` | `manuscript/numbers.tex`, `*_rows.tex` | generated |
| L | LaTeX build, no undefined references or citations | — | `bash scripts/build_manuscript.sh` | `manuscript/*.log` | clean |

## Not covered

- Validation task T7 of `tests/gvm_validation_suite.py` is an exploratory
  extension with different families; it is not evidence for any manuscript
  claim and was not re-run.
- The V2 measurements themselves (`data/processed/measurements_v2.csv`) were not
  regenerated (several hours). Their internal consistency was checked: the
  between-run spread matches the batch-means standard errors (median ratio
  0.832 vs 0.833 expected), and the 1D runs in Region I agree with `f(β/n)`
  within 1.8×10⁻⁴ (|z| ≤ 2.2).
- Claims B.4 and B.5 of the SM (wrapped torus, sparse configuration model) are
  proof sketches; they are not used by any result and were not checked.

## Superseded evidence

`validation_results/` holds the author's original T1–T7 outputs (provenance).
Items found defective during the audit (synthetic figure data, the defective
newborn test, non-reproducible regression numbers) are described in
`archive/README.md` and `docs/audit_report.md`.
