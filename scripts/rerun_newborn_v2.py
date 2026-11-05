"""
rerun_newborn_v2.py -- regenerate the newborn-identity test on the V2
truncated interaction sets with the corrected newborn_extinction_test.

Uses the same graph builders, truncation and seeds as
src/simulation/measure_pipeline.py (interaction sets of run seed 1; test
seed 555; 4e5 fires per cell). Writes
data/processed/newborn_test_v2_corrected.csv.

Usage:  python scripts/rerun_newborn_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "simulation"))
import measure_pipeline as mp  # noqa: E402
from simulator_numba import newborn_extinction_test  # noqa: E402


def main():
    fams = {"1D": lambda n: mp.adj_1d(n), "2D": lambda n: mp.adj_2d_king(n),
            "config": lambda n: mp.adj_config(n, 6, 42), "WS": lambda n: mp.adj_ws(n, 6, 0.1, 42)}
    rows = []
    for fam, build in fams.items():
        for n in (100, 144, 400):
            adj = build(n)
            for q in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
                beta = int(round(q * n))
                balls, fullw = mp.truncated_balls_words(adj, n, beta, 1)
                _, ct = mp.containment_words(balls, n, balls.shape[1], fullw)
                tr, sc = newborn_extinction_test(balls, n, 400000, 555)
                p = sc / tr if tr else np.nan
                se = np.sqrt(ct * (1 - ct) / tr) if tr else np.nan
                rows.append(dict(family=fam, n=n, q=q, beta=beta, trials=int(tr), successes=int(sc),
                                 empirical=p, ctilde_formula=float(ct),
                                 z=(p - ct) / se if se and se > 0 else (0.0 if p == ct else np.nan)))
            print(fam, n, flush=True)
    d = pd.DataFrame(rows)
    out = ROOT / "data/processed/newborn_test_v2_corrected.csv"
    d.to_csv(out, index=False)
    z = d["z"].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"wrote {out}; rows {len(d)}; max |empirical - ctilde| = "
          f"{(d.empirical - d.ctilde_formula).abs().max():.4f}; max |z| = {z.abs().max():.2f}; "
          f"cells with ctilde>0: {(d.ctilde_formula > 0).sum()}")


if __name__ == "__main__":
    main()
