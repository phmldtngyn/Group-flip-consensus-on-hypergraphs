#!/usr/bin/env python3
"""SM Heuristic: large-n annealed random map; stationary mean of 1-x with batch-means SE."""
import numpy as np
rng=np.random.default_rng(2)
for q in [0.70,0.75,0.80,0.90]:
    T=4_000_000; burn=10_000; u=rng.random(T); x=0.1; vals=np.empty(T-burn)
    for t in range(T):
        x=(1-q)*x if u[t]<x else (1-q)*(1-x)
        if t>=burn: vals[t-burn]=1-x
    B=vals.reshape(200,-1).mean(1)
    print(q, f"{vals.mean():.5f} +- {B.std(ddof=1)/np.sqrt(len(B)):.5f}",flush=True)
