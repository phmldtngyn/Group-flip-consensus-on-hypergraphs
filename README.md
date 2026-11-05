# Group-flip consensus on hypergraphs — reproducibility package

Code, data and manuscript sources for

> P. L. D. Nguyen, *Group-flip consensus on hypergraphs: exact stationary band at
> high coverage, exact one-dimensional containment, and an exploratory controlled
> test of the one-step containment hypothesis*, submitted to Physica A (2026).

Every number, table and figure in the manuscript and its Supplementary Material
(SM) is produced by a script in this repository. Numbers quoted in the text are
not typed by hand: `scripts/make_numbers_tex.py` writes them into
`manuscript/numbers.tex` and the table-row files from the analysis outputs.

## The model in one paragraph

On a fixed hypergraph with `n` nodes, each node `i` has a quenched interaction
ball `B_i` (all nodes within hypergraph distance `r`). Each node fires at rate 1;
when `i` fires, **every node of `i`'s colour inside `B_i` changes to a common
target colour** chosen uniformly among the other colours. With two colours, the
observable is the stationary majority fraction `M`. The coverage is `q = β/n`,
where `β = |B_i|`.

## Main results and their status

| Result | Status | Where |
|---|---|---|
| `E[M] = (2−q)/(3−2q)` when every ball is a strict majority (R1), every residual is erased by one fire (R2), and balls have equal size | **Theorem** (proof in SM §C) | main Thm 2 |
| Rotation identity; two-point cycle law `E[K] = 1/E[M]`, `CV_T² = (7−8q+2q²)/(3−2q)²` | **Theorem** | main Thm 1, App. B; SM Prop C.1, Thm C.8 |
| Closed forms for the containment observables `c_G`, `c̃` on the 1D hypergrid; three regions | **Theorem** (checked against exhaustive enumeration, 192 cases) | main Thms 3–4; SM §E |
| Stationary `E[M]` below the R2 threshold for six odd `n ≤ 25` | **Exact rationals**, certified by exact arithmetic | main Table 1 |
| `E[M] ≤ f(q)` throughout Region II | **Conjecture** (six cases) | main Conj. 1 |
| Does `c̃` account for the family dependence of `M` at intermediate coverage? | **Exploratory test: no.** γ̂ = −0.0196, 95% CI [−0.0222, −0.0169]; adding `c̃` increases family separation | main §6, SM Protocol F.4 |

## Repository layout

```
manuscript/            main.tex, SM.tex, references.bib, numbers.tex (generated),
                       *_rows.tex (generated), figures/, compiled PDFs
src/
  model/               1D balls, containment (Numba), graph families
  simulation/          measure_pipeline.py (V2 measurements), simulator_numba.py
  exact/               exact_chain_1d.py (repository exact solver)
  analysis/            reanalysis_v2.py (regression + bootstraps, primary analysis)
  visualization/       make_figures.py (Figs 2-4), fig15.py (Fig 1 schematic)
tests/
  unit_tests.py        fast pytest suite
  gvm_validation_suite.py   validation tasks T1-T6 (T7 is exploratory, not used)
  independent/         independent cross-checks written for the revision audit:
                       containment_bruteforce.py, exact_chain_independent.py,
                       sim_independent.py
scripts/               run_all.py (pipeline), make_numbers_tex.py, build_manuscript.sh
configs/               v2.yaml (measurement + analysis), exact_grid.yaml, v1.yaml (provenance)
data/processed/        measurements_v2.csv (216 runs), newborn_test_v2_corrected.csv
results/
  tables/reanalysis/   point estimates, bootstrap replicates and summaries, jackknife
  tables/              exact_b_crosscheck.* (repository solver output)
  validation/          outputs of the independent checks; suite_rerun/ (T1-T6)
  logs/                run logs
validation_results/    T1-T7 outputs from the author's original run (provenance)
docs/                  validation.md (ledger), reproducibility.md, audit_report.md
archive/               superseded analyses, figures and scripts (see archive/README.md)
```

## Installation

```bash
git clone https://github.com/phmldtngyn/Group-flip-consensus-on-hypergraphs.git
cd Group-flip-consensus-on-hypergraphs
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt                  # exact versions used for this release
```

Python ≥ 3.10 (tested on 3.13). Building the PDFs additionally needs `pdflatex`,
`bibtex` and the `elsarticle` class (TeX Live package `texlive-publishers`).

## Reproducing the paper

```bash
python scripts/run_all.py            # everything except the multi-hour V2 simulations (~35 min)
python scripts/run_all.py --quick    # smoke test (bootstrap B = 200)
```

Individual steps:

| Step | Command | Output | Time |
|---|---|---|---|
| Unit tests | `pytest -q` | — | 1 s |
| 1D containment vs enumeration | `python tests/independent/containment_bruteforce.py` | `results/validation/containment_bruteforce.json` | 1 s |
| Exact chain, independent + exact rationals | `python tests/independent/exact_chain_independent.py` | `results/validation/exact_independent.json` | ~1 min |
| Exact chain, repository solver | `python src/exact/exact_chain_1d.py --config configs/exact_grid.yaml --out results/tables/exact_b_crosscheck.json` | `results/tables/exact_b_crosscheck.*` | ~5 min |
| Validation suite T1–T6 | `python tests/gvm_validation_suite.py --task T5 --out results/validation/suite_rerun` (etc.) | `results/validation/suite_rerun/` | ~1 min |
| Simulation checks | `python tests/independent/sim_independent.py` | `results/validation/sim_independent.json` | ~2 min |
| Newborn identity, V2 sets | `python scripts/rerun_newborn_v2.py` | `data/processed/newborn_test_v2_corrected.csv` | ~5 min |
| Regression test | `python src/analysis/reanalysis_v2.py --B 2000` | `results/tables/reanalysis/` | ~25 min |
| Figures 2–4 | `python src/visualization/make_figures.py --outdir manuscript/figures` | `manuscript/figures/` | 10 s |
| Text numbers and tables | `python scripts/make_numbers_tex.py` | `manuscript/numbers.tex`, `*_rows.tex` | 1 s |
| PDFs | `bash scripts/build_manuscript.sh` | `manuscript/main.pdf`, `SM.pdf` | 30 s |
| V2 simulations (optional) | `python src/simulation/measure_pipeline.py --out data/raw/v2_rerun` | new CSV (does not overwrite the released one) | hours |

## Paper-to-code map

| Manuscript item | Produced by | Data |
|---|---|---|
| Table 1, SM Tables S2–S3, Fig. 2 | `tests/independent/exact_chain_independent.py` | — |
| Fig. 3, Sec. 4 validation | `tests/independent/containment_bruteforce.py` | — |
| Table 2, SM Table S4, Fig. 4, Sec. 6 numbers | `src/analysis/reanalysis_v2.py` | `data/processed/measurements_v2.csv` |
| Sec. 7 simulation checks | `tests/independent/sim_independent.py`; `validation_results/T3,T4,T6` | — |
| Newborn identity on the V2 interaction sets (SM Remark F.1.1) | `scripts/rerun_newborn_v2.py` | `data/processed/newborn_test_v2_corrected.csv` |
| Fig. 1 | `src/visualization/fig15.py` (schematic drawing) | — |

## The data

`data/processed/measurements_v2.csv` — 216 rows, **one per run**: 4 families
(1D cycle, 2D king torus, configuration model k = 6, Watts–Strogatz k = 6,
p = 0.1) × 3 sizes (100, 144, 400) × 6 coverages (0.55–0.80) × 3 seeds.
Columns: `family, n, q` (nominal), `beta = round(q n)`, `cG`, `ctilde` (exact
enumeration on the truncated interaction sets of that run), `M` (mean majority
fraction, 1.5×10⁵ fires after 5×10⁴ burn-in), `se` (batch-means standard error
of `M`, 75 batches), `seed`. One graph per (family, n); the seed also breaks ties
in the last BFS layer of the truncated sets. See `data/README.md`.

## Validation status

See `docs/validation.md` for the ledger (every check, its command and its
evidence file) and `docs/audit_report.md` for the revision audit, including the
problems found in the previous release.

## Known limitations

- Band and Region-II results are for two colours and equal ball sizes.
- Region II is solved exactly only for six small odd `n`.
- The regression test uses one graph realisation per (family, n), truncated
  rather than native interaction sets, and only 12 clusters; its intervals are
  conditional on these graphs. The coefficient of `c̃` is identified almost
  entirely by the 1D family.

## Citation

See `CITATION.cff`. Until the article is published:

```bibtex
@misc{nguyen2026groupflip,
  author = {Nguyen, Pham Le Dat},
  title  = {Group-flip consensus on hypergraphs: exact stationary band at high
            coverage, exact one-dimensional containment, and an exploratory
            controlled test of the one-step containment hypothesis},
  year   = {2026},
  note   = {Submitted to Physica A},
  url    = {https://github.com/phmldtngyn/Group-flip-consensus-on-hypergraphs}
}
```

## Licence

MIT (see `LICENSE`).
