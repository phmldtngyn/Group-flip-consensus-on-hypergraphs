"""
run_regression.py — Full regression + bootstrap pipeline for V2 analysis.

This script:
  1. Reads config from configs/v2.yaml (bootstrap B, seed, group columns).
  2. Loads seed-level simulation data from data/processed/measurements_raw.csv.
  3. Aggregates to cell level: one row per (family, n, q, beta, ctilde),
     keeping M, se, M_std, n_runs (needed for the proper cell SE formula).
  4. Writes the aggregated data to data/processed/measurements_v2.csv.
  5. Fits reduced/full regressions and runs the cluster bootstrap
     (main, with_size, interaction specifications).
  6. Optionally runs the sensitivity sweep (spec x weighting).
  7. Writes results to results/tables/:
       - measurements_v2.csv     (aggregated cells)
       - coefficients.csv        (analytic SEs, all specs)
       - bootstrap.csv           (bootstrap percentile CIs, all specs)
       - family_norm.csv         (family-effect norm reduction, all pairs)
       - sensitivity.csv         (optional; with --sensitivity)
       - regression_summary.json (compact summary for the paper)

Outputs match SM Protocol F.4. Do NOT describe results as a
matched-(n,q) comparison: the specification includes C(n) and an
optional family:n interaction, but n has only three levels and q enters
continuously. Results are exploratory unless B >= 2000 is used.

Usage
-----
    python scripts/run_regression.py --config configs/v2.yaml
    python scripts/run_regression.py --config configs/v2.yaml --sensitivity --B 2000

Author: [author]
Last updated: 2026-11
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Make src/ importable when invoked from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.regression_bootstrap import (  # noqa: E402
    FORMULAS,
    SPEC_PAIRS,
    bootstrap_family_norm_reduction,
    bootstrap_regression,
    compute_cell_se,
    fit_models,
    summarize_models,
    run_sensitivity,
)

# Acceptance threshold on the family-norm reduction statistic (SM F.4).
FAMILY_NORM_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Config and IO
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_paths(cfg: dict, args: argparse.Namespace) -> dict:
    """Resolve input/output paths with CLI overrides taking precedence."""
    data_cfg = cfg.get("data", {})
    reg_cfg = cfg.get("regression", {})

    raw = args.raw or data_cfg.get("raw", "data/processed/measurements_raw.csv")
    aggregated = (
        args.aggregated
        or data_cfg.get("aggregated", "data/processed/measurements_v2.csv")
    )
    results = args.results_dir or reg_cfg.get("results_dir", "results/tables")

    return {
        "raw": REPO_ROOT / raw,
        "aggregated": REPO_ROOT / aggregated,
        "results": REPO_ROOT / results,
    }


def resolve_bootstrap(cfg: dict, args: argparse.Namespace) -> dict:
    boot_cfg = cfg.get("regression", {}).get("bootstrap", {})
    B = args.B if args.B is not None else int(boot_cfg.get("B", 500))
    seed = int(boot_cfg.get("seed", 9999))
    group_cols = tuple(boot_cfg.get("group_cols", ["family", "n"]))
    return {"B": B, "seed": seed, "group_cols": group_cols}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

RAW_REQUIRED = {"family", "n", "q", "beta", "ctilde", "M", "se"}
CELL_KEYS = ["family", "n", "q", "beta", "ctilde"]


def aggregate_cells(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate seed-level runs to cells.

    Each output row corresponds to one (family, n, q, beta, ctilde) cell
    and carries:
      - M        : mean over seeds
      - se       : arithmetic mean of seed-level SEs (kept for backward compat)
      - M_std    : between-seed standard deviation of M
      - n_runs   : number of seeds in the cell
    """
    missing = RAW_REQUIRED - set(raw_df.columns)
    if missing:
        raise SystemExit(
            f"Raw data is missing required columns: {sorted(missing)}. "
            "Expected at least: family, n, q, beta, ctilde, M, se."
        )

    # ddof=1 std gives NaN for n_runs=1; compute_cell_se handles NaN as 0.
    agg = (
        raw_df.groupby(CELL_KEYS, dropna=False)
        .agg(
            M=("M", "mean"),
            se=("se", "mean"),
            M_std=("M", "std"),
            n_runs=("M", "size"),
        )
        .reset_index()
    )

    # Sanity: no zero or NaN SE in the fallback column
    bad = agg["se"].isna() | (agg["se"] <= 0)
    if bad.any():
        warnings.warn(
            f"{int(bad.sum())} cell(s) have non-positive or NaN 'se'; "
            "compute_cell_se will fall back to the median SE for these.",
            UserWarning,
        )

    return agg


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def fit_and_bootstrap(
    df: pd.DataFrame,
    B: int,
    seed: int,
    group_cols: tuple,
    weighted: bool = True,
) -> dict:
    """Fit all specs, bootstrap coefficients, and bootstrap family-norm stats."""
    models = fit_models(df, weighted=weighted)
    coef_table = summarize_models(models)

    boot_table = bootstrap_regression(
        df, B=B, seed=seed, weighted=weighted, group_cols=group_cols
    )

    fn_rows = []
    for pair_name, (red, full) in SPEC_PAIRS.items():
        if red not in FORMULAS or full not in FORMULAS:
            continue
        if ("_size" in red or "_interact" in red) and "n" not in df.columns:
            continue
        arr = bootstrap_family_norm_reduction(
            df, B=B, seed=seed, weighted=weighted,
            group_cols=group_cols, pair=(red, full),
        )
        if arr.size:
            fn_rows.append({
                "pair": pair_name,
                "reduced_spec": red,
                "full_spec": full,
                "median": float(np.median(arr)),
                "ci_lo": float(np.percentile(arr, 2.5)),
                "ci_hi": float(np.percentile(arr, 97.5)),
                "frac_above_threshold": float((arr >= FAMILY_NORM_THRESHOLD).mean()),
                "n_boot": int(arr.size),
            })
        else:
            fn_rows.append({
                "pair": pair_name, "reduced_spec": red, "full_spec": full,
                "median": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "frac_above_threshold": float("nan"),
                "n_boot": 0,
            })

    return {
        "coefficients": coef_table,
        "bootstrap": boot_table,
        "family_norm": pd.DataFrame(fn_rows),
    }


def make_summary(
    df: pd.DataFrame,
    results: dict,
    B: int,
    seed: int,
    cfg_path: str,
) -> dict:
    """Compact summary used by the paper and figures."""
    boot = results["bootstrap"]
    gamma = boot[(boot["model"] == "full") & (boot["term"] == "ctilde")]
    fn = results["family_norm"]

    se_method = (
        "cell" if {"M_std", "n_runs"}.issubset(df.columns) else "fallback"
    )

    summary = {
        "config": cfg_path,
        "n_cells": int(len(df)),
        "n_runs_total": int(df["n_runs"].sum()) if "n_runs" in df.columns else None,
        "B": B,
        "seed": seed,
        "se_method": se_method,
        "gamma": (
            {
                "mean": float(gamma["estimate_mean"].iloc[0]),
                "ci_lo": float(gamma["ci_lower_2.5"].iloc[0]),
                "ci_hi": float(gamma["ci_upper_97.5"].iloc[0]),
                "ci_excludes_zero": bool(
                    gamma["ci_upper_97.5"].iloc[0] < 0
                    or gamma["ci_lower_2.5"].iloc[0] > 0
                ),
            }
            if not gamma.empty else None
        ),
        "family_norm": {
            row["pair"]: {
                "median": row["median"],
                "ci_lo": row["ci_lo"],
                "ci_hi": row["ci_hi"],
                "frac_above_threshold": row["frac_above_threshold"],
            }
            for _, row in fn.iterrows()
        },
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/v2.yaml",
        help="YAML config (default: configs/v2.yaml).",
    )
    parser.add_argument(
        "--raw", default=None,
        help="Override path to seed-level measurements CSV.",
    )
    parser.add_argument(
        "--aggregated", default=None,
        help="Override path for aggregated CSV output.",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Override results directory (default: results/tables).",
    )
    parser.add_argument(
        "--B", type=int, default=None,
        help="Bootstrap replications (overrides config).",
    )
    parser.add_argument(
        "--sensitivity", action="store_true",
        help="Run the spec x weighting sensitivity sweep.",
    )
    parser.add_argument(
        "--no-aggregate", action="store_true",
        help="Skip aggregation; use the existing aggregated CSV directly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(REPO_ROOT / args.config)
    paths = resolve_paths(cfg, args)
    boot = resolve_bootstrap(cfg, args)

    paths["results"].mkdir(parents=True, exist_ok=True)
    paths["aggregated"].parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Aggregation
    # ------------------------------------------------------------------
    if args.no_aggregate:
        print(f"[1/5] Using existing aggregated file: {paths['aggregated']}")
        if not paths["aggregated"].exists():
            raise SystemExit(f"Aggregated file not found: {paths['aggregated']}")
        agg_df = pd.read_csv(paths["aggregated"])
    else:
        print(f"[1/5] Loading raw data: {paths['raw']}")
        raw_df = pd.read_csv(paths["raw"])
        print(f"      {len(raw_df):,} seed-level rows")
        agg_df = aggregate_cells(raw_df)
        agg_df.to_csv(paths["aggregated"], index=False)
        print(
            f"[2/5] Wrote {len(agg_df)} cells to {paths['aggregated']} "
            f"(with M_std, n_runs)"
        )

    if {"M_std", "n_runs"}.issubset(agg_df.columns):
        print("      Cell SE: proper formula (within + between).")
    else:
        print(
            "      WARNING: aggregated file lacks M_std/n_runs; "
            "compute_cell_se will fall back to 'se'."
        )

    # ------------------------------------------------------------------
    # 2. Main analysis
    # ------------------------------------------------------------------
    print(
        f"[3/5] Fit + bootstrap (B={boot['B']}, seed={boot['seed']}, "
        f"group_cols={boot['group_cols']})"
    )
    results = fit_and_bootstrap(
        agg_df,
        B=boot["B"],
        seed=boot["seed"],
        group_cols=boot["group_cols"],
        weighted=True,
    )

    results["coefficients"].to_csv(
        paths["results"] / "coefficients.csv", index=False
    )
    results["bootstrap"].to_csv(
        paths["results"] / "bootstrap.csv", index=False
    )
    results["family_norm"].to_csv(
        paths["results"] / "family_norm.csv", index=False
    )
    print(f"      → coefficients.csv, bootstrap.csv, family_norm.csv")

    # ------------------------------------------------------------------
    # 3. Sensitivity (optional)
    # ------------------------------------------------------------------
    if args.sensitivity:
        print(f"[4/5] Sensitivity sweep (B={boot['B']})")
        sens = run_sensitivity(
            agg_df, B=boot["B"], seed=boot["seed"],
        )
        sens.to_csv(paths["results"] / "sensitivity.csv", index=False)
        print(f"      → sensitivity.csv ({len(sens)} rows)")
    else:
        print("[4/5] Skipping sensitivity (pass --sensitivity to enable).")

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print("[5/5] Writing regression_summary.json")
    summary = make_summary(
        agg_df, results, boot["B"], boot["seed"], cfg_path=args.config,
    )
    with open(paths["results"] / "regression_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Console recap
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    g = summary["gamma"]
    if g is not None:
        print(
            f"  gamma (ctilde):  {g['mean']:+.6f}  "
            f"95% CI [{g['ci_lo']:+.6f}, {g['ci_hi']:+.6f}]"
        )
    for pair, stats in summary["family_norm"].items():
        if np.isnan(stats["median"]):
            continue
        print(
            f"  norm reduction [{pair}]: median {stats['median']:+.4f}  "
            f"95% CI [{stats['ci_lo']:+.4f}, {stats['ci_hi']:+.4f}]  "
            f"frac>0.30 = {stats['frac_above_threshold']:.3f}"
        )
    print("=" * 64)
    print(f"Results saved in: {paths['results']}")


if __name__ == "__main__":
    main()