"""
v2 measurement protocol (produces data/processed/measurements_v2.csv):

NOTE (revision audit): the truncation tie-break in the last BFS layer uses
the run seed (1, 2, 3), so the three runs of a cell may use slightly
different interaction sets; this is documented in SM Protocol F.4 and is
left unchanged so that the released data remain reproducible.

Usage:  python src/simulation/measure_pipeline.py --out data/raw/v2_rerun
(runtime: several hours on a laptop).

Original description:
 - target-q grid (same q for all families)
 - equal-size truncated balls of size beta = round(q*n) via BFS layering
 - true c_G, c_tilde on truncated balls (word-bitmask, Numba)
 - simulator M with batch-mean SE on the SAME truncated balls
 - newborn extinction unit test against c_tilde
"""
import numpy as np
import pandas as pd
from collections import deque
from numba import njit, prange
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # sibling modules
from simulator_numba import simulate_M, newborn_extinction_test, popcnt
import os
import json


# ---------- words helpers ----------
def to_words(mask_bool, n):
    W = (n + 63) // 64
    words = np.zeros(W, dtype=np.uint64)
    idx = np.flatnonzero(mask_bool)
    for v in idx:
        words[v // 64] |= np.uint64(1) << np.uint64(v % 64)
    return words


@njit(cache=True, parallel=True)
def containment_words(balls_words, n, W, fullw):
    cG = np.int64(0)
    ctn = np.int64(0)
    ctd = np.int64(0)
    for i in prange(n):
        res = np.empty(W, dtype=np.uint64)
        rsize = np.int64(0)
        for w in range(W):
            res[w] = fullw[w] & ~balls_words[i, w]
            rsize += np.int64(popcnt(res[w]))
        if rsize == 0:
            cG += n
            ctn += 1
            ctd += 1
            continue
        ca = np.int64(0)
        cr = np.int64(0)
        for j in range(n):
            ok = True
            for w in range(W):
                if (res[w] & ~balls_words[j, w] & fullw[w]) != 0:
                    ok = False
                    break
            if ok:
                ca += 1
                wj = j // 64
                bj = j % 64
                if (res[wj] >> np.uint64(bj)) & np.uint64(1):
                    cr += 1
        cG += ca
        ctn += cr
        ctd += rsize
    return cG / (n * n), (ctn / ctd if ctd > 0 else 1.0)


# ---------- adjacency builders (true geometries) ----------
def adj_1d(n):
    return [{(i - 1) % n, (i + 1) % n} for i in range(n)]


def adj_2d_king(n):
    L = int(round(n ** 0.5))
    assert L * L == n, f"n={n} is not a perfect square, required for 2D grid"
    adj = []
    for i in range(L):
        for j in range(L):
            s = set()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        s.add(((i + di) % L) * L + (j + dj) % L)
            adj.append(s)
    return adj


def adj_config(n, k, seed):
    import networkx as nx
    G = nx.configuration_model([k] * n, seed=seed)
    G = nx.Graph(G)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    adj = [set() for _ in range(n)]
    for u, v in G.edges():
        adj[u].add(v)
        adj[v].add(u)
    return adj


def adj_ws(n, k, p, seed):
    import networkx as nx
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    adj = [set() for _ in range(n)]
    for u, v in G.edges():
        adj[u].add(v)
        adj[v].add(u)
    return adj


# ---------- equal-size truncated balls ----------
def truncated_balls_words(adj, n, beta, seed):
    rng = np.random.default_rng(seed)
    W = (n + 63) // 64
    fullw = np.full(W, np.uint64(0xFFFFFFFFFFFFFFFF))
    rem = n % 64
    if rem:
        fullw[W - 1] = (np.uint64(1) << np.uint64(rem)) - np.uint64(1)
    balls = np.empty((n, W), dtype=np.uint64)

    for s in range(n):
        dist = [-1] * n
        order = []
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            order.append(u)
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        sel = [order[0]]
        pos, d = 1, 1
        while len(sel) < beta and pos < len(order):
            layer = []
            while pos < len(order) and dist[order[pos]] == d:
                layer.append(order[pos])
                pos += 1
            need = beta - len(sel)
            if len(layer) <= need:
                sel.extend(layer)
            else:
                sel.extend(rng.choice(layer, size=need, replace=False).tolist())
            d += 1
        m = np.zeros(n, dtype=bool)
        m[sel] = True
        balls[s] = to_words(m, n)
    return balls, fullw


# ---------- one measurement row ----------
def measure_row(family, n, q_target, seed, adj,
                T=200000, burn=50000, batch_len=2000):
    beta = int(round(q_target * n))
    beta = max(beta, 2)
    balls, fullw = truncated_balls_words(adj, n, beta, seed)
    W = balls.shape[1]
    cG, ctilde = containment_words(balls, n, W, fullw)
    batches = simulate_M(balls, n, T, burn, batch_len, seed + 7777)
    M = float(np.mean(batches))
    se = float(np.std(batches, ddof=1) / np.sqrt(len(batches))) if len(batches) > 1 else 0.0
    return {
        "family": family, "n": n, "q": q_target, "beta": beta,
        "cG": float(cG), "ctilde": float(ctilde),
        "M": M, "se": se, "seed": seed,
    }


def run_v2(output_dir="results_v2"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)

    families = {
        "1D":     lambda n: adj_1d(n),
        "2D":     lambda n: adj_2d_king(n),
        "config": lambda n: adj_config(n, 6, 42),
        "WS":     lambda n: adj_ws(n, 6, 0.1, 42),
    }
    
    # Use perfect squares to ensure 2D grid works seamlessly
    n_list = [100, 144, 400]
    q_list = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    seeds = [1, 2, 3]

    rows = []
    newborn_rows = []
    
    for fname, builder in families.items():
        for n in n_list:
            adj = builder(n)
            for q in q_list:
                for sd in seeds:
                    rows.append(measure_row(fname, n, q, sd, adj))
                
                # newborn unit test at one seed per (family,n,q)
                beta = int(round(q * n))
                balls, fullw = truncated_balls_words(adj, n, beta, 1)
                tr, sc = newborn_extinction_test(balls, n, 400000, 555)
                _, ct = containment_words(balls, n, balls.shape[1], fullw)
                newborn_rows.append({
                    "family": fname, "n": n, "q": q,
                    "trials": int(tr), "successes": int(sc),
                    "empirical": (sc / tr) if tr else np.nan,
                    "ctilde_formula": float(ct),
                })
            print(f"[v2] done {fname} n={n}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/measurements_v2.csv", index=False)
    
    ndf = pd.DataFrame(newborn_rows)
    ndf.to_csv(f"{output_dir}/newborn_test_v2.csv", index=False)
    
    print("v2 measurement complete.")
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Re-run the V2 measurement protocol.")
    ap.add_argument("--out", default="data/raw/v2_rerun", help="output directory")
    run_v2(ap.parse_args().out)