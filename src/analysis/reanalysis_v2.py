"""
reanalysis_v2.py -- audit and corrected re-analysis of the V2 controlled
regression test (main-text Protocol 1, SM Protocol F.4).

Input
-----
data/processed/measurements_v2.csv : 216 SEED-LEVEL rows
    (4 families x 3 sizes x 6 coverages x 3 seeds). Column `se` is the
    batch-means Monte Carlo standard error of that run (75 batches of
    2000 events after 5e4 burn-in). The file is NOT cell-aggregated.

What this script does (and why)
-------------------------------
A. Traces the three gamma values that appear in earlier drafts
   (-0.0165, -0.017867, -0.0179) to the estimator that produced each.
B. Re-defines the family-effect norm so that it is genuinely invariant
   to the choice of reference family (all 6 pairwise contrasts among
   the 4 families), and reports the original 3-contrast statistic for
   comparison.
C. Re-runs the bootstrap three ways:
     orig  : two-level cluster bootstrap exactly as in
             src/analysis/regression_bootstrap.py (reproduces the draft);
     strat : family-stratified cluster bootstrap -- the 3 (family, n)
             clusters of each family are resampled within that family,
             and inside each selected cluster the 3 seeds are resampled
             within each of the 6 coverage strata. Every replicate keeps
             all 4 families and all 6 coverage values per cluster;
     iid   : rows resampled independently (v1 sensitivity, anti-
             conservative; reproduced for traceability).
D. Specification sensitivity on the SAME data: linear q (as stated),
   categorical q, full (n, q)-cell fixed effects, cell means (72 rows),
   OLS vs WLS.
E. Diagnostics: where the c-tilde variation lives; leave-one-cluster-out
   jackknife; between-seed SD vs batch-means SE.

Primary analysis (fixed before inspecting the new outputs)
----------------------------------------------------------
Specification stated in the manuscript, M ~ q + ctilde + C(family) + C(n),
WLS with weights 1/se^2, family-stratified cluster bootstrap ("strat"),
B = 2000, seed = 9999, percentile 95% intervals.

Usage
-----
    python src/analysis/reanalysis_v2.py --data data/processed/measurements_v2.csv \
        --out results/tables/reanalysis --B 2000
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

FAMILIES = ["1D", "2D", "WS", "config"]

SPECS = {
    # name: (reduced, full)
    "main_noN":   ("M ~ q + C(family)",               "M ~ q + ctilde + C(family)"),
    "stated":     ("M ~ q + C(family) + C(n)",        "M ~ q + ctilde + C(family) + C(n)"),
    "q_cat":      ("M ~ C(q) + C(family) + C(n)",     "M ~ C(q) + ctilde + C(family) + C(n)"),
    "nq_cellFE":  ("M ~ C(n):C(q) + C(family)",       "M ~ C(n):C(q) + ctilde + C(family)"),
    # realised coverage beta/n instead of the nominal grid value q
    "stated_qe":  ("M ~ qe + C(family) + C(n)",       "M ~ qe + ctilde + C(family) + C(n)"),
}


# ----------------------------------------------------------------- helpers

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"family", "n", "q", "beta", "cG", "ctilde", "M", "se", "seed"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"missing columns {miss}")
    if len(df) != 216:
        warnings.warn(f"expected 216 rows, found {len(df)}")
    df["family"] = pd.Categorical(df["family"], categories=FAMILIES)
    df["q"] = df["q"].round(4)
    df["qe"] = df["beta"] / df["n"]   # realised coverage
    df["w"] = 1.0 / df["se"] ** 2
    df["w"] = df["w"] / df["w"].mean()
    df["cluster"] = df["family"].astype(str) + "_" + df["n"].astype(str)
    return df


def fit(formula: str, d: pd.DataFrame, weighted: bool = True):
    if weighted:
        return smf.wls(formula, data=d, weights=d["w"]).fit()
    return smf.ols(formula, data=d).fit()


def family_effects(res) -> np.ndarray:
    """Effects of the 4 families on a common scale (1D = 0 reference).
    Returns NaN entries for families absent from the fit."""
    out = np.zeros(4)
    for k, f in enumerate(FAMILIES[1:], start=1):
        t = f"C(family)[T.{f}]"
        out[k] = res.params[t] if t in res.params.index else np.nan
    return out


def norm_all6(a: np.ndarray) -> float:
    """Reference-invariant: L2 norm of all 6 pairwise differences."""
    return float(np.sqrt(sum((a[i] - a[j]) ** 2 for i, j in combinations(range(4), 2))))


def norm_orig3(a: np.ndarray) -> float:
    """Statistic used in the draft: the 3 contrasts among the NON-reference
    families (2D-WS, 2D-config, WS-config). Not reference-invariant."""
    b = a[1:]
    return float(np.sqrt(sum((b[i] - b[j]) ** 2 for i, j in combinations(range(3), 2))))


def norm_v1(a: np.ndarray) -> float:
    """v1 statistic: L2 norm of the coefficients relative to the 1D reference."""
    return float(np.sqrt(np.nansum(a[1:] ** 2)))


def reduction(nr: float, nf: float) -> float:
    return (nr - nf) / nr if nr > 0 else np.nan


def stats_for(d: pd.DataFrame, spec: str, weighted: bool = True) -> dict:
    fr, ff = SPECS[spec]
    rr, rf = fit(fr, d, weighted), fit(ff, d, weighted)
    ar, af = family_effects(rr), family_effects(rf)
    out = {"gamma": float(rf.params.get("ctilde", np.nan)),
           "families_present": int(d["family"].astype(str).nunique())}
    for name, fn in (("all6", norm_all6), ("orig3", norm_orig3), ("v1", norm_v1)):
        out[f"norm_red_{name}"] = fn(ar)
        out[f"norm_full_{name}"] = fn(af)
        out[f"reduction_{name}"] = reduction(fn(ar), fn(af))
    return out


# ------------------------------------------------------------ resamplers

def resample_orig(d, rng):
    """Two-level cluster bootstrap as implemented in the draft code."""
    clusters = d["cluster"].unique()
    pick = rng.integers(0, len(clusters), size=len(clusters))
    parts = []
    for k, c in enumerate(pick):
        g = d[d["cluster"] == clusters[c]]
        g = g.iloc[rng.integers(0, len(g), size=len(g))].copy()
        g["bcluster"] = k
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def resample_strat(d, rng):
    """Family-stratified cluster bootstrap; seeds resampled within q strata."""
    parts = []
    k = 0
    for f in FAMILIES:
        sub = d[d["family"] == f]
        sizes = np.sort(sub["n"].unique())
        for c in rng.integers(0, len(sizes), size=len(sizes)):
            g = sub[sub["n"] == sizes[c]]
            for _, cell in g.groupby("q"):
                parts.append(cell.iloc[rng.integers(0, len(cell), size=len(cell))]
                             .assign(bcluster=k))
            k += 1
    return pd.concat(parts, ignore_index=True)


def resample_iid(d, rng):
    return d.iloc[rng.integers(0, len(d), size=len(d))].reset_index(drop=True)


def resample_mc_only(d, rng):
    """Seeds resampled within each (family, n, q) cell; clusters fixed.
    Captures Monte Carlo (dynamics) noise only."""
    parts = [cell.iloc[rng.integers(0, len(cell), size=len(cell))]
             for _, cell in d.groupby(["family", "n", "q"], observed=True)]
    return pd.concat(parts, ignore_index=True)


RESAMPLERS = {"orig": resample_orig, "strat": resample_strat,
              "iid": resample_iid, "mc_only": resample_mc_only}


def bootstrap(d, spec, scheme, B, seed, weighted=True):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(B):
        db = RESAMPLERS[scheme](d, rng)
        try:
            s = stats_for(db, spec, weighted)
        except Exception as exc:  # pragma: no cover
            s = {"error": str(exc)}
        s["b"] = b
        rows.append(s)
    return pd.DataFrame(rows)


def summarise(boot: pd.DataFrame, point: dict) -> dict:
    g = boot["gamma"].dropna()
    out = {"B_requested": int(len(boot)), "B_valid_gamma": int(len(g)),
           "replicates_missing_a_family": int((boot["families_present"] < 4).sum()),
           "gamma_point": point["gamma"],
           "gamma_boot_mean": float(g.mean()), "gamma_boot_median": float(g.median()),
           "gamma_boot_sd": float(g.std(ddof=1)),
           "gamma_ci_lo": float(np.percentile(g, 2.5)),
           "gamma_ci_hi": float(np.percentile(g, 97.5)),
           "frac_gamma_gt0": float((g > 0).mean())}
    for name in ("all6", "orig3", "v1"):
        v = boot[f"reduction_{name}"].replace([np.inf, -np.inf], np.nan).dropna()
        out[f"reduction_{name}_point"] = point[f"reduction_{name}"]
        out[f"reduction_{name}_median"] = float(v.median())
        out[f"reduction_{name}_ci_lo"] = float(np.percentile(v, 2.5))
        out[f"reduction_{name}_ci_hi"] = float(np.percentile(v, 97.5))
        out[f"reduction_{name}_frac_ge_0.30"] = float((v >= 0.30).mean())
        out[f"reduction_{name}_B_valid"] = int(len(v))
    return out


# -------------------------------------------------------------- diagnostics

def diagnostics(d: pd.DataFrame) -> dict:
    cells = d.groupby(["family", "n", "q"], observed=True).agg(
        ct_min=("ctilde", "min"), ct_max=("ctilde", "max"),
        M_sd=("M", "std"), se_mean=("se", "mean"), M=("M", "mean"))
    tot = ((d["ctilde"] - d["ctilde"].mean()) ** 2).sum()
    share = {f: float(((d.loc[d.family == f, "ctilde"] - d["ctilde"].mean()) ** 2).sum() / tot)
             for f in FAMILIES}
    # within-family variation of ctilde (what identifies gamma once family FE are in)
    wf = d.groupby("family", observed=True)["ctilde"].transform("mean")
    within = ((d["ctilde"] - wf) ** 2)
    within_share = {f: float(within[d.family == f].sum() / within.sum()) for f in FAMILIES}
    ratio = (cells["M_sd"] / cells["se_mean"]).dropna()
    return {
        "rows": int(len(d)), "cells": int(len(cells)),
        "cells_ctilde_varies_across_seeds": int(((cells.ct_max - cells.ct_min) > 1e-12).sum()),
        "cells_with_ctilde_gt0_by_family": {f: int((cells.loc[f, "ct_max"] > 0).sum()) for f in FAMILIES},
        "ctilde_max_by_family": {f: float(cells.loc[f, "ct_max"].max()) for f in FAMILIES},
        "share_total_SS_ctilde_by_family": share,
        "share_within_family_SS_ctilde_by_family": within_share,
        "between_seed_sd_over_mean_batch_se": {
            "median": float(ratio.median()), "mean": float(ratio.mean()),
            "theory_median_if_se_correct": float(np.sqrt(np.log(2))),
            "note": "for R=3 replicates, s/sigma has median sqrt(ln 2)=0.8326"},
        "batch_se_range": [float(d.se.min()), float(d.se.max())],
    }


def jackknife(d, spec):
    out = []
    for c in d["cluster"].unique():
        s = stats_for(d[d.cluster != c], spec)
        out.append({"dropped_cluster": c, "gamma": s["gamma"],
                    "reduction_all6": s["reduction_all6"]})
    j = pd.DataFrame(out)
    G = len(j)
    gbar = j["gamma"].mean()
    se = float(np.sqrt((G - 1) / G * ((j["gamma"] - gbar) ** 2).sum()))
    return j, se


def cell_means(d):
    c = d.groupby(["family", "n", "q"], observed=True).agg(
        M=("M", "mean"), ctilde=("ctilde", "mean"), se=("se", "mean"),
        M_sd=("M", "std"), R=("M", "size")).reset_index()
    # cell-mean SE: combines within-run MC error and between-seed spread
    c["se"] = np.sqrt((c["se"] ** 2 + c["M_sd"] ** 2) / (2 * c["R"]))
    c["w"] = 1.0 / c["se"] ** 2
    c["w"] = c["w"] / c["w"].mean()
    c["cluster"] = c["family"].astype(str) + "_" + c["n"].astype(str)
    c["seed"] = 0
    return c


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/measurements_v2.csv")
    ap.add_argument("--out", default="results/tables/reanalysis")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=9999)
    ap.add_argument("--points-only", action="store_true",
                    help="recompute point estimates/diagnostics only and merge into an existing report")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = load(args.data)

    report = {"config": vars(args), "diagnostics": diagnostics(d)}

    # A. point estimates for every spec
    pts = {}
    for spec in SPECS:
        for wt in (True, False):
            pts[f"{spec}|{'wls' if wt else 'ols'}"] = stats_for(d, spec, wt)
    cm = cell_means(d)
    cm["qe"] = cm["q"]  # cell means carry the nominal q only
    for spec in ("main_noN", "stated"):
        pts[f"{spec}|cellmeans_wls"] = stats_for(cm, spec, True)
    report["point_estimates"] = pts
    pd.DataFrame(pts).T.to_csv(f"{args.out}/point_estimates.csv")
    if args.points_only:
        path = f"{args.out}/reanalysis_report.json"
        old = json.load(open(path))
        old["point_estimates"] = pts
        old["diagnostics"] = report["diagnostics"]
        json.dump(old, open(path, "w"), indent=2, default=float)
        print({k: round(v["gamma"], 5) for k, v in pts.items()})
        return

    # full coefficient table for the stated spec
    res = fit(SPECS["stated"][1], d)
    coef = pd.DataFrame({"estimate": res.params, "analytic_se_wls": res.bse})
    coef.to_csv(f"{args.out}/coefficients_stated_full.csv")
    res0 = fit(SPECS["main_noN"][1], d)
    pd.DataFrame({"estimate": res0.params}).to_csv(f"{args.out}/coefficients_main_noN_full.csv")

    # B/C. bootstraps
    runs = [  # (spec, scheme, B, seed)
        ("main_noN", "orig", 500, 9999),     # reproduce the draft
        ("main_noN", "iid", 500, 9999),      # reproduce v1-style iid
        ("stated", "orig", args.B, args.seed),
        ("stated", "strat", args.B, args.seed),   # PRIMARY (revised)
        ("stated", "iid", args.B, args.seed),
        ("stated", "mc_only", args.B, args.seed),
        ("main_noN", "strat", args.B, args.seed),
        ("q_cat", "strat", args.B, args.seed),
        ("nq_cellFE", "strat", args.B, args.seed),
    ]
    summ = {}
    for spec, scheme, B, seed in runs:
        key = f"{spec}|{scheme}|B{B}"
        bt = bootstrap(d, spec, scheme, B, seed)
        bt.to_csv(f"{args.out}/boot_{spec}_{scheme}_B{B}.csv", index=False)
        summ[key] = summarise(bt, pts[f"{spec}|wls"])
        print(key, {k: round(v, 5) if isinstance(v, float) else v
                    for k, v in summ[key].items() if k.startswith(("gamma", "reduction_all6", "replic"))},
              flush=True)
    # cell-means (72 rows) with stratified cluster bootstrap over clusters only
    report["bootstrap"] = summ
    pd.DataFrame(summ).T.to_csv(f"{args.out}/bootstrap_summary.csv")

    # E. jackknife
    jk, jse = jackknife(d, "stated")
    jk.to_csv(f"{args.out}/jackknife_stated.csv", index=False)
    report["jackknife_stated"] = {"se_gamma": jse,
                                  "gamma_range": [float(jk.gamma.min()), float(jk.gamma.max())]}

    with open(f"{args.out}/reanalysis_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps(report["diagnostics"], indent=1, default=float))
    print(json.dumps(report["jackknife_stated"], indent=1))


if __name__ == "__main__":
    main()
