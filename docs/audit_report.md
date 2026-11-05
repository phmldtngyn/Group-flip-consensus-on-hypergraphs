# Revision audit — November 2026

Manuscript: *Group-flip consensus on hypergraphs: exact stationary band at high
coverage, exact one-dimensional containment, and an exploratory controlled test
of the one-step containment hypothesis* (target: Physica A).

This report covers: (1) the inputs and their consistency, (2) the problems found,
by severity, (3) items that need the author's decision, (4) what was verified
and how, (5) the change log, and (6) a readiness assessment. All new or changed
files are on branch `revision` of the repository; nothing was pushed.

---

## 1. Inputs and version consistency

| Input | Status |
|---|---|
| SM source (`SM.txt`, received as `message (2).txt`) | Compiles; word-for-word identical to the SM PDF (`draft_5.pdf`) apart from hyphenation and font encoding. **Authoritative for the SM.** |
| Main-text source (`main.txt`) | **Not received**, and the repository contains no LaTeX. `manuscript/main.tex` was therefore **reconstructed from the main PDF (`draft_7.pdf`)** and then revised. If your own source differs in layout or macros, port the changes from `main.tex` rather than replacing your file wholesale. |
| Main PDF (`draft_7.pdf`, 27 pp.) | Internally inconsistent with the SM in several places (Sec. 2 below). All figures were floated to the end and numbered D.1–D.5; Table 2 was cut off at the right margin; the title page was duplicated. |
| Repository `main` (commit `9c3e806`) | One version of the data; no earlier manuscript versions. Several outputs cited by the manuscript cannot be regenerated from the repository (Sec. 2). |
| Title | The PDF title contains "exploratory"; the brief you supplied omits it. I kept "exploratory", which matches the content. |

---

## 2. Findings, by severity

### Critical (could invalidate a result or the credibility of the paper)

**C1. Figure 3 showed formula values with added noise, labelled as brute-force enumeration.**
`src/visualization/fig3.py` computed Eqs. (12)–(13) and plotted them twice: as the
curve and, after adding Gaussian noise (`np.random.normal(0, 0.008)`), as the
"Brute-force enum." markers. A comment in the script says the noise was added to
look realistic. *Resolution:* real exhaustive enumeration
(`tests/independent/containment_bruteforce.py`) agrees **exactly** with the closed
forms in all 192 cases, so the theorem stands; the figure was rebuilt from the
enumeration output.

**C2. Figure 4(a) showed synthetic data labelled as simulation means with SE bars.**
`src/visualization/fig4.py` plotted `0.4804 + 0.4999 q + family offset` with error
bars drawn from `np.random.uniform(0.004, 0.008)`. Panels (b)–(c) showed the v1
iid bootstrap while the caption described the cluster bootstrap. *Resolution:*
rebuilt from `measurements_v2.csv` and the re-analysis outputs. The real error
bars (standard error of the mean of three runs, about 8×10⁻⁵) are 50–100
times smaller than the synthetic ones.

**C3. The headline statistical conclusion is reversed once the analysis is implemented correctly.**
The draft reported γ̂ = −0.0165, 95% CI [−0.0230, +0.0045] ("indistinguishable
from zero"), and an unchanged family separation (reduction −0.031), and it
attributed the v1 results to two artifacts. The audit found:

| Value in draft | What it actually was |
|---|---|
| γ̂ = −0.0165 | **Mean of the bootstrap distribution**, not the point estimate; and for the specification **without** `C(n)`, although the protocol states `C(n)`. |
| −0.017867 (Table S4 "point estimate") and the coefficients in `regression.json` | **Not reproducible** from the released data with WLS, OLS or cell means; evidently from an earlier data version that is not in the repository. |
| −0.0179 | Median of the v1 iid bootstrap — also from the unavailable data version. |
| CI [−0.0230, +0.0045] | Unstratified cluster bootstrap in which 61/500 replicates lack a whole family. **All** positive-γ replicates come from such incomplete replicates; in the same scheme applied to the stated specification (B = 2000), the replicates that contain all four families give [−0.0238, −0.0138]. |
| Family-norm reduction −0.031 | Statistic built from the 3 contrasts among 2D/WS/config only. It is **not** reference-invariant and ignores 1D, the only family where c̃ varies. |

With the specification as stated and a cluster bootstrap that keeps every family
and every coverage stratum (fixed as the primary analysis **before** the new
results were seen): **γ̂ = −0.0196, 95% CI [−0.0222, −0.0169]**; six-contrast
reduction **−0.98 [−1.13, −0.85]**, meaning adding c̃ roughly doubles the family
separation. This holds with categorical q (−0.0185 [−0.0208, −0.0161]), full
(n,q)-cell fixed effects (−0.0186 [−0.0215, −0.0159]), without `C(n)`, without
weights, on cell means, with q = β/n, and under leave-one-cluster-out
(all 12 estimates in [−0.0208, −0.0186]). The v1 "−0.865" was therefore not an
artifact: the inflation of the family separation is real.

*Consequence:* both acceptance criteria still fail, more decisively (wrong sign,
interval excluding zero), so "no evidence that c̃ mediates the family separation"
survives. But the paper's narrative (null at the cluster level; iid significance
and norm inflation are artifacts; S6.5's general methodological warning) is not
supported and has been rewritten. **This needs your approval** (Sec. 3, item A1).

**C4. The newborn-identity test released with the data was the defective version.**
`data/processed/newborn_test_v2.csv` has **0 successes in all 72 rows**, including
the 27 cells with c̃ > 0. It was produced by `simulator_numba.newborn_extinction_test`,
which treated a fire whose *result* is monochrome as a "birth" (it is an erasure).
The fix described in SM Remark F.1.1 was applied only in the validation suite (T6).
*Resolution:* function corrected; on known 1D cases it reproduces c̃ (|z| ≤ 1.65);
re-run on all 72 V2 interaction sets: all 18 cells with 0 < c̃ < 1 agree (max |z| = 1.85).

### Major

**M1. Model positioning misdescribed Ref. [7].** The draft called the rule "the
ball-constrained member of the GVM family" and "unanimity-free". In Kim et al.
(PRE 111, L052301, 2025) only the focal node updates, and only after *unanimous*
observations. The rule studied here is related but distinct; rewritten. Kim et
al.'s nonlinearity parameter is also called `q`, clashing with coverage `q`; the
text no longer uses `q` for their parameter.

**M2. Seeds do not vary "only the dynamics randomness".** The truncation code
breaks last-layer ties with the run seed, so c̃ differs between the runs of a cell
in 15 of 72 cells. Now stated in Protocol 1 and SM F.4 (data unchanged).

**M3. Identification.** 96.6% of the within-family variation in c̃ is in the 1D
family; c̃ ≡ 0 in all 2D cells. γ̂ records how the 1D advantage shrinks as 1D
enters Region I. Stated explicitly as a caveat.

**M4. SE-aggregation statement in the SM was wrong in its premise.** The SM said
the released file stores cell averages of seed SEs and that with three seeds this
differs from the correct cell SE "by a global factor". The file is **seed-level**
(one row per run, its own batch-means SE). The batch-means SEs are accurate:
between-run SD / SE has median 0.832, against √(ln 2) = 0.833 expected.

**M5. Placeholder band formula and vacuous H3 check in the repository solver.**
`src/model/balls_1d.py::_M_band_formula` returned `0.5(1−c_G) + 0.5(1−c̃/2)`,
labelled "placeholder, NOT the proven formula". In Region II it returned NaN, so
`exact_chain_1d.py` always reported "no H3 violation" regardless of the data, and
labelled Region II as "high_coverage". Fixed; re-run gives all six differences negative.

**M6. Claims without evidence.** The main text said the cycle statistics (E[K],
CV²_T) and simulator-vs-exact Region-II agreement had been checked; no run
existed in the repository. Both are now checked by an independent simulator
(all |z| < 2).

**M7. Table S3 not traceable.** Its iteration counts and residuals did not match any
output, and it said n = 23, 25 were infeasible, while the repository log shows
all six solved. Replaced by the independent solver's results.

**M8. Literature coverage is thin** (10 references). One verified reference added
(Neuhäuser, Mellor & Lambiotte, PRE 101, 032310, 2020) and one arXiv entry
(Noonan & Lambiotte, arXiv:2101.03632 — please check for a journal version).
More positioning against higher-order voter and majority models would help.

### Moderate

- Remark 2 cited Thm C.6/Remark C.6.1 (which assume R2) for a statement claimed
  under R1 only; now cites Prop C.1. The PASTA sentence was ambiguous (see A2).
- Overclaims removed: "elimination of this natural alternative redirects theory
  toward multi-step fragmentation", "exact finite-size solutions show the band is
  an upper bound", "the first exact finite-size stationary solutions", "matched (n,q)".
- Physical-interpretation section now separates what is shown (sign of the
  deviation) from what is hypothesised (multi-step survival).
- The regressor `q` is the nominal grid value, not β/n (differs ≤ 0.5/n); now
  documented, with the β/n result as sensitivity (γ̂ = −0.0201).
- Figure 2 plotted 3 of the 6 exact cases; now all six.
- Fig. D.4 caption contained an internal to-do ("a definitive check requires
  varying β…"); removed. Fig. D.5 (near-empty schematic of an unestablished
  mechanism) removed from the main text (restorable; see A5).
- SM internal-notebook material removed or rephrased: "normative" remarks,
  editing instructions ("must, at submission, match this ledger"), references to
  earlier drafts, the "S1 (A)" double numbering and crosswalk.
- Repository: `README` called E[M] "the expected consensus time", gave the wrong
  author name, described the regression without `C(n)` and the bootstrap as
  "on aggregated cells"; `run_all.py` called functions with wrong arguments;
  three wrapper scripts were empty files; `unit_tests.py` imported a function
  that does not exist; `requirements.txt` and `pyproject.toml` pinned conflicting
  versions; `LICENSE` was declared but missing; `__pycache__` was committed.

### Minor

- Notation: Eq. (6) equated an expectation with a random quantity; Sec. 3.5 wrote
  an expectation without `E`; Remark 1 called the 2-section graph "the r-th power
  of C_n(±1,±2)" (it is C_n(±1,±2); the ball is the neighbourhood in its r-th power).
- SM Prop. E.3 said "R1 strict for β > (n+1)/2", which would exclude
  β = (n+1)/2 — the n = 9 case of Table 1. Corrected (Phase II item 5.7).
- "Appendix Appendix A" duplication, broken "SM Protocol VI.4" pointer,
  duplicated title page, truncated Table 2, Theorem A.5 statement printed twice.

---

## 3. Items that need your decision

Following your rules, these are marked rather than silently adopted. Items A1–A5
are implemented in the revised files (so the package is coherent) but are
yours to accept or revert; items B1–B4 are flagged only.

| # | Item | What I did |
|---|---|---|
| A1 | **Primary statistical analysis** (stated spec + family-stratified cluster bootstrap, B = 2000) and the resulting reversal of the Sec. 6 narrative | Implemented; all earlier variants reproduced in SM Table S4 / Note F.5. |
| A2 | Remark 2: "the PASTA property is not applicable here" | Clarified to: PASTA applies to the firing epochs (Poisson, rate n, independent of the state), not to rotation epochs. The conclusion E[M] = 1/E[K] is unchanged. |
| A3 | Positioning relative to Kim et al. (M1) | Rewritten as "related but distinct". |
| A4 | SM Props B.4 and B.5 relabelled **Claim** (proof sketch only) | Weakening only; content unchanged. |
| A5 | Fig. D.5 removed; Fig. 2 and Table 1 now report certified rationals | Restore D.5 if you want it. |
| B1 | **SM Prop B.2 proof gap**: the proof uses β > n/2 to exclude wrap-around, but the equivalence R2 ⟺ 3β ≥ 2n−1 is stated without that assumption. I checked numerically that the equivalence holds for all n ≤ 60, so the statement looks right; the β ≤ n/2 case needs a sentence. | Not changed. |
| B2 | **SM Thm C.6 / Lemma C.0(iv)**: the ratio-limit argument starts from a stationary epoch chain; independence of the initial law also needs the (standard) remark that the finite transient does not affect the long-run average. | Not changed. |
| B3 | SM Corollary C.7 mentioned "the value 7/9 is a numerical observation" with no definition of what 7/9 refers to. | Removed; restore with an explanation if meaningful. |
| B4 | SM Prop D.3 gave c_G = (2β−n+1)/n "in 1D" under R2; at β = n this exceeds 1. | Replaced by a pointer to Prop E.1 (which has the cap). |

Changes I treated as unambiguous notation or typography: Eq. (6) and Sec. 3.5
notation, Remark 1, Prop E.3 boundary wording, explicit use of ball symmetry in
Lemma C.3, H_{L0−1} in Thm A.3, f ∈ [4/5, 1] (was [4/5, 1)).

---

## 4. Scientific validation report

| Question | Answer |
|---|---|
| **What was checked** | Every numerical claim in the main text and SM; the full computational pipeline; all mathematical statements for internal consistency (not re-derived from scratch). |
| **Reproduced** | Exact Region-II values (two independent solvers + exact rational certificates); 1D containment formulas (exact, 192 cases); band in Region I (exact chain and simulation, 1D and 2D); simulator vs exact chain in Region II; cycle law; newborn identity (native and truncated sets); v2 bootstrap interval (reproduced exactly from code and data). |
| **Corrected** | γ point estimate and interval; family-effect statistic; bootstrap design; SE-aggregation statement; newborn test data; Figures 2–4; Table S3; region labels and band comparator in code. |
| **Not independently verified** | The proofs themselves (checked for internal consistency only); SM Claims B.4–B.5; the V2 measurements were not re-simulated (internally consistent, see `docs/validation.md`). |
| **Mathematically unresolved** | Conjecture 1 (six cases only); items B1–B2 above; no closed form in Region II. |
| **Statistically uncertain** | Only 12 clusters (3 per family), one graph per cluster; γ identified almost only by 1D; bootstrap with 3 clusters per family may understate realisation-level variability; the analysis was revised after the data were first analysed (now twice). |
| **Softened or reclassified** | Props B.4–B.5 → Claims; fragmentation/multi-step survival → hypothesis; "matched (n,q)" removed; band "shown to be an upper bound" → conjecture; "redirects theory" removed. |
| **Internally consistent?** | Yes, as far as checked: main ↔ SM cross-references are resolved by LaTeX (`xr-hyper`), and every quoted computational number comes from a generated macro file. |

### Computing environment

Python 3.13.18; numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, statsmodels 0.15.0,
numba 0.68.1, matplotlib 3.11.2, networkx 3.7, python-flint 0.9.0; pdfTeX
1.40.25 (TeX Live 2023), elsarticle. Seeds: bootstrap 9999; simulations as in
each script. Run times are in `README.md`.

---

## 5. Change log

Type: E = editorial, S = structural, C = computational, St = statistical,
R = reproducibility, **M = mathematically substantive (needs approval)**.

| File | Change | Reason | Type |
|---|---|---|---|
| `manuscript/main.tex` (new) | Reconstructed from PDF; abstract, introduction, Sec. 6–9 rewritten; figures moved into the body; tables regenerated | No source supplied; findings C1–C4, M1–M8 | S, E, St |
| `manuscript/main.tex` | Remark 2 PASTA wording; positioning vs Ref. [7] | A2, A3 | **M** (approval) |
| `manuscript/SM.tex` | Sec. labels A–H; notebook remarks removed; Props B.4–B.5 → Claims; Prop E.3 wording; Lemma C.3 symmetry; Protocol F.3–F.4, Note F.5, ledger, Tables S2–S4 rewritten from outputs | Findings above | E, S, St, **M** (A4) |
| `manuscript/references.bib` (new) | Verified entries; Neuhäuser et al. added; Noonan & Lambiotte (arXiv) added | M8 | E |
| `manuscript/numbers.tex`, `*_rows.tex` (generated) | Every quoted computational number | Traceability | R |
| `manuscript/figures/fig2–4` | Rebuilt from computed outputs | C1, C2 | C |
| `src/analysis/reanalysis_v2.py` (new) | Corrected analysis, all variants, diagnostics, jackknife | C3 | St |
| `src/simulation/simulator_numba.py` | Newborn test fixed | C4 | C |
| `src/model/balls_1d.py` | Real band formula; correct region labels; `band_applies` flag | M5 | C |
| `src/simulation/measure_pipeline.py`, `src/model/graph_families.py` | Import paths fixed; CLI added; tie-breaking documented (logic unchanged) | Could not run as documented | R |
| `src/visualization/make_figures.py` (new) | Figures 2–4 from outputs only | C1, C2 | C |
| `tests/independent/*` (new) | Independent exact solver (+ exact rationals), brute-force containment, independent simulator | Independent verification | C |
| `tests/unit_tests.py` | Rewritten (old file could not run) | — | R |
| `scripts/run_all.py` | Rewritten as a working pipeline | Old one called functions with wrong arguments | R |
| `scripts/make_numbers_tex.py`, `build_manuscript.sh`, `rerun_newborn_v2.py` (new) | Generation, build, newborn re-run | R | R |
| `data/processed/newborn_test_v2_corrected.csv` (new) | Corrected newborn test | C4 | C |
| `results/tables/reanalysis/`, `results/validation/` (new) | All outputs of the audit | Evidence | R |
| `results/tables/exact_b_crosscheck.*` | Re-run with corrected comparator | M5 | C |
| `README.md`, `data/README.md`, `docs/validation.md`, `docs/audit_report.md`, `archive/README.md` | Rewritten / new | Accuracy | E |
| `configs/*.yaml` | Comments corrected (data are run-level; tie-breaking; v1 provenance); Vietnamese comments translated | Accuracy | E |
| `requirements*.txt`, `environment.yml`, `pyproject.toml`, `CITATION.cff`, `LICENSE`, `.gitignore` | Consistent, tested versions; correct author; MIT text as declared in README | R | R |
| moved to `archive/` | v1/v2 analysis outputs, old figures and figure scripts, defective newborn CSV, superseded drivers, old logs | Superseded; kept for traceability | S |
| removed | Committed `__pycache__`; three empty wrapper scripts; two empty log files | Build artifacts / empty | S |

---

## 6. Submission-readiness assessment (internal; not a prediction of acceptance)

| Category | Score | Justification and remaining weaknesses |
|---|---|---|
| Originality and contribution | 6 | The exact band for a quenched group-flip rule via an affine epoch identity is a clean result. The model is narrow (Q = 2, equal balls) and reduces to a pairwise 2-section graph, so the "hypergraph" angle is modest. |
| Scientific relevance to Physica A | 7 | Exact stationary results for an interacting particle system fit the journal well. |
| Mathematical exposition | 7 | Proofs are clear after cleanup; items B1–B2 remain; many proofs are compressed. |
| Theoretical consistency | 8 | No contradictions found after fixes; exact computations agree with every theorem tested. |
| Physical interpretation | 5 | Region II is characterised only by sign and six numbers; no mechanism is established; the family dependence remains unexplained. |
| Numerical methodology | 8 | Exact rational certificates, two independent solvers, independent simulator; the V2 truncated-set design departs from the theory's assumptions. |
| Statistical validity | 5 | Correct now, but the test is weakly identified (1D only), has 12 clusters, and was revised post hoc twice. Its conclusion is modest. |
| Reproducibility | 8 | One command regenerates everything except the multi-hour simulations; all numbers are generated. The original measurement run is not re-executed here. |
| Literature positioning | 5 | Ten references; needs fuller engagement with higher-order voter/majority/consensus literature. |
| Academic writing | 7 | Clear and restrained after revision; the main text is long relative to its content and Sec. 6 could be shortened. |
| LaTeX quality | 7 | Compiles cleanly, cross-references checked automatically; main.tex is a reconstruction you must reconcile with your source. |
| Repository quality | 7 | Clear structure, documented, tested; legacy modules (`containment_numba.py`, `graph_families.py`, validation suite) are still somewhat ad hoc. |

**Recommendation: Requires further revision.**

The mathematical core and the exact computations hold up under independent
checking, which is a good position. Before submission:

1. Decide on items A1–A5 and B1–B4.
2. Reconcile `manuscript/main.tex` with your own source.
3. Because two figures and one data file contained synthetic or defective
   content that the text presented as evidence, re-check any material produced
   the same way (in particular anything generated with AI assistance) against
   real outputs before submission; the AI-use declaration should reflect this.
4. Consider whether Sec. 6 earns its place in the main text. With c̃ varying
   only within 1D, it answers a narrow question; moving it to the SM, with a
   short summary in the main text, would sharpen the paper.
5. Expand the literature review.
