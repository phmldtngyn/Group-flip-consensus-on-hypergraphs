"""
sim_independent.py -- independent Monte Carlo checks for the revision audit.

Shares no code with src/simulation/simulator_numba.py or
tests/gvm_validation_suite.py. Continuous-time Gillespie simulation of the
Q = 2 ball-constrained group-flip dynamics on NATIVE metric balls of the
hypergraph (balls from BFS on the hypergraph, not truncated).

Checks
------
S1  Region II, 1D hypergrid, all six exact cases: time-averaged majority
    fraction vs the independent exact-chain value.
S2  Region I (R1 and R2 verified by BFS): 1D (n=400, r=75; n=401, r=70)
    and 2D (L=30, r=10; L=12, r=4): M vs band f(q) = (2-q)/(3-2q).
S3  Dwell law (Theorem C.8 / Appendix B) in Region I: mean number of
    events per rotation cycle E[K] = (3-2q)/(2-q) and squared coefficient
    of variation of the cycle time CV_T^2 = (7-8q+2q^2)/(3-2q)^2.

Uncertainty: 10 independent replicas per case; reported SE is the
standard error across replicas (no batch-means assumption).

Usage
-----
    python tests/independent/sim_independent.py --out results/validation/sim_independent.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque

import numpy as np
from numba import njit


# ---------------------------------------------------------------- geometry

def hypergrid_adjacency(L: int, d: int):
    """2-section graph of the d-dim periodic hypergrid whose hyperedges are
    3 consecutive nodes along one axis."""
    n = L ** d
    adj = [set() for _ in range(n)]
    for v in range(n):
        coords = list(np.unravel_index(v, (L,) * d))
        for ax in range(d):
            for start in (-2, -1, 0):
                edge = []
                for k in range(3):
                    c = coords.copy()
                    c[ax] = (c[ax] + start + k) % L
                    edge.append(int(np.ravel_multi_index(c, (L,) * d)))
                for u in edge:
                    if u != v:
                        adj[v].add(u)
    return adj


def bfs_dist(adj, s):
    n = len(adj)
    dist = np.full(n, -1)
    dist[s] = 0
    dq = deque([s])
    while dq:
        u = dq.popleft()
        for w in adj[u]:
            if dist[w] < 0:
                dist[w] = dist[u] + 1
                dq.append(w)
    return dist


def native_balls(adj, r):
    n = len(adj)
    D = np.array([bfs_dist(adj, s) for s in range(n)])
    balls = [np.flatnonzero(D[i] <= r) for i in range(n)]
    return balls, D


def check_R1_R2(balls, D, r):
    n = len(balls)
    r1 = all(len(b) * 2 > n for b in balls)
    r2 = True
    for b in balls:
        res = np.setdiff1d(np.arange(n), b)
        if len(res) and D[np.ix_(res, res)].max() > r:
            r2 = False
            break
    sizes = {len(b) for b in balls}
    return r1, r2, sizes


def to_csr(balls):
    ptr = np.zeros(len(balls) + 1, dtype=np.int64)
    for i, b in enumerate(balls):
        ptr[i + 1] = ptr[i] + len(b)
    idx = np.concatenate(balls).astype(np.int64)
    return ptr, idx


# ---------------------------------------------------------------- dynamics

@njit(cache=True)
def run(ptr, idx, n, T_events, burn, seed):
    np.random.seed(seed)
    col = np.zeros(n, dtype=np.int8)
    n1 = 0
    t = 0.0
    tw_M = 0.0
    tw_T = 0.0
    label = 0                    # current strict-majority colour (start mono 0)
    last_rot_t = -1.0
    last_rot_e = -1
    ncyc = 0
    sK = 0.0
    sK2 = 0.0
    sT = 0.0
    sT2 = 0.0
    for e in range(T_events):
        dt = np.random.exponential(1.0 / n)
        if e >= burn:
            maj = max(n1, n - n1) / n
            tw_M += maj * dt
            tw_T += dt
        t += dt
        i = np.random.randint(n)
        c = col[i]
        for p in range(ptr[i], ptr[i + 1]):
            j = idx[p]
            if col[j] == c:
                col[j] = 1 - c
                if c == 0:
                    n1 += 1
                else:
                    n1 -= 1
        # strict-majority label (ties keep the old label; ties are transient)
        new = label
        if 2 * n1 > n:
            new = 1
        elif 2 * n1 < n:
            new = 0
        if new != label:
            label = new
            if e >= burn and last_rot_e >= 0:
                K = e - last_rot_e
                Tc = t - last_rot_t
                ncyc += 1
                sK += K
                sK2 += K * K
                sT += Tc
                sT2 += Tc * Tc
            last_rot_e = e
            last_rot_t = t
    return tw_M / tw_T, ncyc, sK, sK2, sT, sT2


def replicas(balls, T, burn, R, seed0):
    ptr, idx = to_csr(balls)
    n = len(balls)
    Ms, EK, CV2 = [], [], []
    for k in range(R):
        M, nc, sK, sK2, sT, sT2 = run(ptr, idx, n, T, burn, seed0 + 1009 * k)
        Ms.append(M)
        if nc > 0:
            mT = sT / nc
            EK.append(sK / nc)
            CV2.append((sT2 / nc - mT ** 2) / mT ** 2)
    f = lambda a: (float(np.mean(a)), float(np.std(a, ddof=1) / np.sqrt(len(a))))
    return f(Ms), f(EK), f(CV2)


def band(q):
    return (2 - q) / (3 - 2 * q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/validation/sim_independent.json")
    ap.add_argument("--exact", default="results/validation/exact_independent.json")
    ap.add_argument("--R", type=int, default=10)
    ap.add_argument("--T", type=int, default=2_000_000)
    args = ap.parse_args()
    exact = {}
    if os.path.exists(args.exact):
        for rec in json.load(open(args.exact)):
            exact[(rec["n"], rec["r"])] = rec["M_float"]
    out = {"S1_regionII": [], "S2_S3_regionI": []}
    t0 = time.time()
    # S1
    for n, r in [(9, 1), (15, 2), (17, 2), (21, 3), (23, 3), (25, 3)]:
        balls, D = native_balls(hypergrid_adjacency(n, 1), r)
        r1, r2, sizes = check_R1_R2(balls, D, r)
        (M, se), _, _ = replicas(balls, args.T, args.T // 20, args.R, 11)
        Mx = exact.get((n, r))
        rec = dict(n=n, r=r, beta=sizes.pop(), R1=r1, R2=r2, M_sim=M, se=se,
                   M_exact=Mx, z=(M - Mx) / se if Mx else None)
        print(rec, flush=True)
        out["S1_regionII"].append(rec)
    # S2/S3
    for name, L, d, r in [("1D", 400, 1, 75), ("1D", 401, 1, 70), ("2D", 12, 2, 4), ("2D", 30, 2, 10)]:
        balls, D = native_balls(hypergrid_adjacency(L, d), r)
        r1, r2, sizes = check_R1_R2(balls, D, r)
        n = len(balls)
        beta = sizes.pop() if len(sizes) == 1 else None
        q = beta / n
        (M, se), (K, seK), (C, seC) = replicas(balls, args.T, args.T // 20, args.R, 23)
        rec = dict(family=name, L=L, n=n, r=r, beta=beta, q=q, R1=r1, R2=r2,
                   M_sim=M, M_se=se, M_band=band(q), z_M=(M - band(q)) / se,
                   EK_sim=K, EK_se=seK, EK_theory=(3 - 2 * q) / (2 - q),
                   z_EK=(K - (3 - 2 * q) / (2 - q)) / seK,
                   CV2_sim=C, CV2_se=seC, CV2_theory=(7 - 8 * q + 2 * q * q) / (3 - 2 * q) ** 2,
                   z_CV2=(C - (7 - 8 * q + 2 * q * q) / (3 - 2 * q) ** 2) / seC)
        print(rec, flush=True)
        out["S2_S3_regionI"].append(rec)
    out["config"] = vars(args)
    out["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
