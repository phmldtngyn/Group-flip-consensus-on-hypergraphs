"""
regression_bootstrap.py — Regression + bootstrap for V2 analysis.

Model specifications (must match Main Text and SM F.4)
------------------------------------------------------
Matched pairs (reduced baseline → full model with ctilde):

    reduced          : M ~ q + C(family)
    full             : M ~ q + ctilde + C(family)

    reduced_size     : M ~ q + C(family) + C(n)
    full_size        : M ~ q + ctilde + C(family) + C(n)

    reduced_interact : M ~ q + C(family) * C(n)
    full_interact    : M ~ q + ctilde + C(family) * C(n)

The 'size' and 'interact' pairs implement the size controls requested by
the reviewer. They are still NOT full (n, q)-cell fixed-effects models:
n has only three levels, and q enters continuously. Results are
EXPLORATORY; the paper must not describe them as a matched-(n,q)
comparison.

Cell-level standard error
-------------------------
If the aggregated DataFrame contains 'M_std' (across-run standard
deviation of M) and 'n_runs' (number of runs per cell), the proper
cell-level SE is

    SE_cell^2 = (1/R) * mean_r(s_r^2)  +  (1/R) * M_std^2

where s_r are the within-run Monte Carlo SEs. This combines within-run
Monte Carlo uncertainty with between-run variability.

If 'M_std' and 'n_runs' are missing, we fall back to the stored 'se'
column (arithmetic mean of seed-level SEs) and emit a UserWarning. The
fallback underestimates between-run variability; regenerate the
aggregation with M_std and n_runs for the proper formula.

Bootstrap scheme
----------------
- Input is the *aggregated* DataFrame (one row per cell).
- Resampling unit: (family, n) groups; within each resampled group, rows
  are resampled with replacement (two-level cluster bootstrap).
- B = 500 (default, exploratory). B >= 2000 is prescribed for any
  confirmatory expansion.
- Seed: 9999 (default).

Uncertainty interpretation
--------------------------
- The bootstrap produces regression-estimator uncertainty (CIs on coefs).
- Simulation-run uncertainty enters only through the 1/SE^2 weights.
- Confidence intervals do NOT include graph-ensemble uncertainty, because
  only one graph realization is used per (family, n) pair.

Author: [author]
Last updated: 2026-11
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------------
# Regression specifications
# ---------------------------------------------------------------------------

FORMULA_REDUCED = "M ~ q + C(family)"
FORMULA_FULL = "M ~ q + ctilde + C(family)"

FORMULA_REDUCED_SIZE = "M ~ q + C(family) + C(n)"
FORMULA_FULL_SIZE = "M ~ q + ctilde + C(family) + C(n)"

FORMULA_REDUCED_INTERACT = "M ~ q + C(family) * C(n)"
FORMULA_FULL_INTERACT = "M ~ q + ctilde + C(family) * C(n)"

FORMULAS = {
    "reduced": FORMULA_REDUCED,
    "full": FORMULA_FULL,
    "reduced_size": FORMULA_REDUCED_SIZE,
    "full_size": FORMULA_FULL_SIZE,
    "reduced_interact": FORMULA_REDUCED_INTERACT,
    "full_interact": FORMULA_FULL_INTERACT,
}

# Matched (reduced, full) pairs used for the family-norm reduction test.
SPEC_PAIRS = {
    "main":        ("reduced",          "full"),
    "with_size":   ("reduced_size",     "full_size"),
    "interaction": ("reduced_interact", "full_interact"),
}


# ---------------------------------------------------------------------------
# Cell-level standard error
# ---------------------------------------------------------------------------

def compute_cell_se(df: pd.DataFrame) -> pd.Series:
    """
    Compute the proper cell-level standard error.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated data with column 'se'. If columns 'M_std' and 'n_runs'
        are present, use the proper combined formula; otherwise fall back
        to 'se' with a UserWarning.

    Returns
    -------
    pd.Series
        Cell-level standard errors, indexed like df.
    """
    if "se" not in df.columns:
        raise KeyError("Column 'se' is required.")

    has_m_std = "M_std" in df.columns
    has_n_runs = "n_runs" in df.columns

    if has_m_std and has_n_runs:
        R = df["n_runs"].astype(float).clip(lower=1.0)
        within_var = (df["se"].astype(float) ** 2) / R
        # For R=1, M_std is undefined; treat between-run variance as 0.
        between_var = (df["M_std"].astype(float).fillna(0.0) ** 2) / R
        cell_se = np.sqrt(within_var + between_var)
    else:
        warnings.warn(
            "Columns 'M_std' and 'n_runs' not found; falling back to "
            "stored 'se' (arithmetic mean of seed-level SEs). This "
            "underestimates between-run variability. Regenerate the "
            "aggregation with M_std and n_runs for the proper formula.",
            UserWarning,
            stacklevel=2,
        )
        cell_se = df["se"].astype(float)

    # Guard against zero or NaN SE
    positive = cell_se[cell_se > 0]
    fallback = float(positive.median()) if len(positive) > 0 else 1.0
    cell_se = cell_se.replace(0, np.nan).fillna(fallback)
    return cell_se


def _attach_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Attach normalized 1/SE^2 weights as column '__w'."""
    se = compute_cell_se(df)
    df = df.copy()
    w = 1.0 / (se ** 2)
    df["__w"] = w / w.mean()
    return df


# ---------------------------------------------------------------------------
# Fit models
# ---------------------------------------------------------------------------

def fit_models(
    df: pd.DataFrame,
    weighted: bool = True,
    formulas: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """
    Fit regression models on aggregated data.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated data with at least columns: M, q, ctilde, family.
    weighted : bool
        If True, use WLS with weights 1/SE_cell^2; otherwise OLS.
    formulas : dict, optional
        Mapping from name to formula. Defaults to FORMULAS.

    Returns
    -------
    dict mapping model name to fitted statsmodels results object.
    """
    required = {"M", "q", "family"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if formulas is None:
        formulas = FORMULAS

    work = _attach_weights(df) if weighted else df.copy()

    results = {}
    for name, formula in formulas.items():
        if ("_size" in name or "_interact" in name) and "n" not in work.columns:
            warnings.warn(f"Skipping '{name}': column 'n' not found.")
            continue
        try:
            if weighted:
                res = smf.wls(formula, data=work, weights=work["__w"]).fit()
            else:
                res = smf.ols(formula, data=work).fit()
            results[name] = res
        except Exception as exc:
            warnings.warn(f"Could not fit '{name}': {exc}")

    return results


def summarize_models(models: Dict[str, object]) -> pd.DataFrame:
    """Extract coefficients, standard errors, and p-values into a tidy table."""
    rows: List[Dict] = []
    for name, res in models.items():
        for term in res.params.index:
            rows.append({
                "model": name,
                "term": term,
                "estimate": float(res.params[term]),
                "std_err": float(res.bse[term]),
                "t_value": float(res.tvalues[term]),
                "p_value": float(res.pvalues[term]),
                "n_obs": int(res.nobs),
                "r_squared": float(res.rsquared),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _resample_groups_and_rows(
    df: pd.DataFrame,
    group_cols: Tuple[str, ...],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Cluster bootstrap: resample groups, then rows within each group."""
    group_keys = df[list(group_cols)].drop_duplicates().reset_index(drop=True)
    group_keys["__gid"] = np.arange(len(group_keys))
    df = df.merge(group_keys, on=list(group_cols), how="left")

    groups = [g for _, g in df.groupby("__gid", sort=False)]
    n_groups = len(groups)

    sampled_gids = rng.integers(0, n_groups, size=n_groups)

    out_frames = []
    for new_gid, gid in enumerate(sampled_gids):
        g = groups[gid]
        idx = rng.integers(0, len(g), size=len(g))
        g_resampled = g.iloc[idx].copy()
        g_resampled["__gid"] = new_gid
        out_frames.append(g_resampled)

    result = pd.concat(out_frames, ignore_index=True)
    return result.drop(columns=["__gid"])


def _fit_boot(formula: str, df_boot: pd.DataFrame, weighted: bool):
    if weighted:
        return smf.wls(formula, data=df_boot, weights=df_boot["__w"]).fit()
    return smf.ols(formula, data=df_boot).fit()


def bootstrap_regression(
    df: pd.DataFrame,
    B: int = 500,
    seed: int = 9999,
    group_cols: Tuple[str, ...] = ("family", "n"),
    weighted: bool = True,
    formulas: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Bootstrap regression coefficients on aggregated data.

    Returns
    -------
    DataFrame with columns:
        model, term, estimate_mean, estimate_sd,
        ci_lower_2.5, ci_upper_97.5, B
    """
    if "M" not in df.columns or "q" not in df.columns:
        raise KeyError("Aggregated DataFrame must contain 'M' and 'q'.")

    if formulas is None:
        formulas = FORMULAS

    work = _attach_weights(df) if weighted else df.copy()
    rng = np.random.default_rng(seed)
    samples: Dict[Tuple[str, str], List[float]] = {}

    for _ in range(B):
        df_boot = _resample_groups_and_rows(work, group_cols, rng)
        for name, formula in formulas.items():
            if ("_size" in name or "_interact" in name) and "n" not in df_boot.columns:
                continue
            try:
                res = _fit_boot(formula, df_boot, weighted)
            except Exception:
                continue
            for term in res.params.index:
                samples.setdefault((name, term), []).append(float(res.params[term]))

    rows: List[Dict] = []
    for (name, term), vals in samples.items():
        arr = np.asarray(vals, dtype=float)
        rows.append({
            "model": name,
            "term": term,
            "estimate_mean": float(arr.mean()),
            "estimate_sd": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
            "ci_lower_2.5": float(np.percentile(arr, 2.5)),
            "ci_upper_97.5": float(np.percentile(arr, 97.5)),
            "B": int(arr.size),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Family-effect norm reduction (reference-invariant)
# ---------------------------------------------------------------------------

def _pairwise_family_contrasts(res) -> np.ndarray:
    """Return all pairwise contrasts of family main effects."""
    terms = [t for t in res.params.index if t.startswith("C(family)[T.")]
    coefs = np.array([res.params[t] for t in terms])
    contrasts = []
    for i in range(len(coefs)):
        for j in range(i + 1, len(coefs)):
            contrasts.append(coefs[i] - coefs[j])
    return np.array(contrasts) if contrasts else np.array([0.0])


def family_norm_reduction(
    models: Dict[str, object],
    full_name: str,
    reduced_name: str,
) -> float:
    """
    Reference-invariant L2-norm reduction of family effects:

        (||contrasts_reduced|| - ||contrasts_full||) / ||contrasts_reduced||

    Negative values indicate inflation (the full model spreads family
    effects further apart than the reduced model).
    """
    if reduced_name not in models or full_name not in models:
        return float("nan")

    t_red = _pairwise_family_contrasts(models[reduced_name])
    t_full = _pairwise_family_contrasts(models[full_name])

    norm_red = float(np.linalg.norm(t_red))
    norm_full = float(np.linalg.norm(t_full))
    if norm_red == 0:
        return float("nan")
    return (norm_red - norm_full) / norm_red


def bootstrap_family_norm_reduction(
    df: pd.DataFrame,
    B: int = 500,
    seed: int = 9999,
    group_cols: Tuple[str, ...] = ("family", "n"),
    weighted: bool = True,
    pair: Tuple[str, str] = ("reduced", "full"),
) -> np.ndarray:
    """
    Bootstrap the family-effect norm reduction statistic.

    Parameters
    ----------
    pair : (reduced_name, full_name)
        Names of the matched specifications in FORMULAS.
    """
    reduced_name, full_name = pair
    if reduced_name not in FORMULAS or full_name not in FORMULAS:
        raise KeyError(f"Unknown spec pair: {pair}")

    work = _attach_weights(df) if weighted else df.copy()
    rng = np.random.default_rng(seed)
    vals: List[float] = []

    for _ in range(B):
        df_boot = _resample_groups_and_rows(work, group_cols, rng)
        models = {}
        ok = True
        for name in (reduced_name, full_name):
            try:
                models[name] = _fit_boot(FORMULAS[name], df_boot, weighted)
            except Exception:
                ok = False
                break
        if not ok:
            continue
        v = family_norm_reduction(models, full_name, reduced_name)
        if not np.isnan(v):
            vals.append(v)

    return np.asarray(vals)


# ---------------------------------------------------------------------------
# Convenience: run full analysis
# ---------------------------------------------------------------------------

def run_analysis(
    df: pd.DataFrame,
    B: int = 500,
    seed: int = 9999,
    weighted: bool = True,
    formulas: Optional[Dict[str, str]] = None,
    pairs: Optional[Dict[str, Tuple[str, str]]] = None,
) -> Dict[str, object]:
    """
    Fit models + bootstrap + family-norm bootstrap.

    Returns
    -------
    dict with keys:
        'coefficients'          : tidy coefficient table (analytic SE)
        'bootstrap'             : bootstrap distributions (percentile CIs)
        'family_norm_bootstrap' : {pair_name: np.ndarray of reductions}
    """
    if formulas is None:
        formulas = FORMULAS
    if pairs is None:
        pairs = SPEC_PAIRS

    models = fit_models(df, weighted=weighted, formulas=formulas)
    coef_table = summarize_models(models)
    boot_table = bootstrap_regression(
        df, B=B, seed=seed, weighted=weighted, formulas=formulas
    )

    family_norm_bootstrap: Dict[str, np.ndarray] = {}
    for name, pair in pairs.items():
        if pair[0] not in formulas or pair[1] not in formulas:
            continue
        family_norm_bootstrap[name] = bootstrap_family_norm_reduction(
            df, B=B, seed=seed, weighted=weighted, pair=pair
        )

    return {
        "coefficients": coef_table,
        "bootstrap": boot_table,
        "family_norm_bootstrap": family_norm_bootstrap,
    }


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def run_sensitivity(
    df: pd.DataFrame,
    B: int = 500,
    seed: int = 9999,
) -> pd.DataFrame:
    """
    Run sensitivity analysis across specifications and weighting schemes.

    For each matched pair (main, with_size, interaction) × (WLS, OLS),
    report the gamma estimate, its bootstrap CI, and the family-norm
    reduction statistic with its bootstrap CI.

    Returns
    -------
    DataFrame with columns:
        pair, weighting, se_method, gamma_estimate, gamma_ci_lo,
        gamma_ci_hi, norm_reduction_median, norm_reduction_ci_lo,
        norm_reduction_ci_hi, n_boot
    """
    rows: List[Dict] = []
    se_method = (
        "cell"
        if ("M_std" in df.columns and "n_runs" in df.columns)
        else "fallback"
    )

    for pair_name, (red, full) in SPEC_PAIRS.items():
        if ("_size" in red or "_interact" in red) and "n" not in df.columns:
            continue
        for weighted in (True, False):
            formulas = {red: FORMULAS[red], full: FORMULAS[full]}
            try:
                boot = bootstrap_regression(
                    df, B=B, seed=seed, weighted=weighted, formulas=formulas
                )
                boot_norm = bootstrap_family_norm_reduction(
                    df, B=B, seed=seed, weighted=weighted, pair=(red, full)
                )
            except Exception as exc:
                warnings.warn(f"Sensitivity spec '{pair_name}' failed: {exc}")
                continue

            gamma_rows = boot[(boot["model"] == full) &
                              (boot["term"] == "ctilde")]
            if gamma_rows.empty:
                continue
            gr = gamma_rows.iloc[0]

            rows.append({
                "pair": pair_name,
                "weighting": "wls" if weighted else "ols",
                "se_method": se_method,
                "gamma_estimate": float(gr["estimate_mean"]),
                "gamma_ci_lo": float(gr["ci_lower_2.5"]),
                "gamma_ci_hi": float(gr["ci_upper_97.5"]),
                "norm_reduction_median": (
                    float(np.median(boot_norm)) if boot_norm.size else float("nan")
                ),
                "norm_reduction_ci_lo": (
                    float(np.percentile(boot_norm, 2.5))
                    if boot_norm.size else float("nan")
                ),
                "norm_reduction_ci_hi": (
                    float(np.percentile(boot_norm, 97.5))
                    if boot_norm.size else float("nan")
                ),
                "n_boot": int(boot_norm.size),
            })

    return pd.DataFrame(rows)