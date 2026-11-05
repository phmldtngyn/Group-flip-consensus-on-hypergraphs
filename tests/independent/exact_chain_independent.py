"""
exact_chain_independent.py -- independent cross-check of the exact
Region-II stationary values (main text Table 1, SM Table S2).

Written for the revision audit. It deliberately shares no code with
src/exact/exact_chain_1d.py or tests/gvm_validation_suite.py:

  * State space: full two-colour configurations (n-bit integers), no
    colour-complement canonicalisation, no dihedral quotient.
  * Transition rule implemented literally from the model definition:
    a fire at i recolours every node of colour sigma_i inside B_i.
  * Irreducibility of the reachable set is checked explicitly
    (strongly connected components).
  * Stationary law: sparse direct solve of pi (P - I) = 0, sum pi = 1.
  * Exact rational certificate: the chain is equivariant under the
    dihedral group D_n and the colour swap, so it is strongly lumpable
    onto orbits. The lumped chain is solved over Q with FLINT (exact
    arithmetic), giving E[M] as an exact rational number.

Usage
-----
    python tests/independent/exact_chain_independent.py \
        --out results/validation/exact_independent.json
"""
from __future__ import annotations

import argparse
import json
import time
from fractions import Fraction

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

CASES = [(9, 1), (15, 2), (17, 2), (21, 3), (23, 3), (25, 3)]


def ball_masks(n: int, r: int) -> list[int]:
    """1D hypergrid ball: d(i,j) = ceil(delta/2) <= r  <=>  delta <= 2r."""
    masks = []
    for i in range(n):
        m = 0
        for j in range(n):
            delta = min((i - j) % n, (j - i) % n)
            if (delta + 1) // 2 <= r:
                m |= 1 << j
        masks.append(m)
    return masks


def fire(x: int, i: int, masks: list[int], full: int) -> int:
    """Colour-1 nodes are the set bits of x. Fire at i flips i's colour class in B_i."""
    b = masks[i]
    if (x >> i) & 1:          # i has colour 1: its colour-1 ball-mates become 0
        return x & ~b
    return x | (b & full)     # i has colour 0: its colour-0 ball-mates become 1


def reachable_chain(n: int, r: int):
    masks = ball_masks(n, r)
    full = (1 << n) - 1
    index = {0: 0}
    states = [0]
    rows, cols = [], []
    k = 0
    while k < len(states):
        x = states[k]
        for i in range(n):
            y = fire(x, i, masks, full)
            if y not in index:
                index[y] = len(states)
                states.append(y)
            rows.append(k)
            cols.append(index[y])
        k += 1
    N = len(states)
    P = sp.csr_matrix((np.full(len(rows), 1.0 / n), (rows, cols)), shape=(N, N))
    P.sum_duplicates()
    return states, P


def majority_fraction(states, n):
    pc = np.array([bin(s).count("1") for s in states])
    return np.maximum(pc, n - pc) / n


def stationary_power(P: sp.csr_matrix, tol: float = 1e-16, max_iter: int = 200000):
    """Lazy power iteration pi <- pi (I+P)/2 from the uniform vector."""
    N = P.shape[0]
    PT = P.T.tocsr()
    pi = np.full(N, 1.0 / N)
    for it in range(max_iter):
        new = 0.5 * (pi + PT @ pi)
        new /= new.sum()
        if np.abs(new - pi).sum() < tol:
            return new, it + 1
        pi = new
    return pi, max_iter


def stationary_direct(P: sp.csr_matrix) -> np.ndarray:
    N = P.shape[0]
    A = (P.T - sp.identity(N, format="csr")).tolil()
    A[0, :] = np.ones(N)
    b = np.zeros(N)
    b[0] = 1.0
    pi = spla.spsolve(A.tocsc(), b)
    return pi


# ---------------- exact rational via symmetry lumping ----------------

def canon(x: int, n: int) -> int:
    """Minimal representative under D_n x colour swap."""
    full = (1 << n) - 1
    best = None
    rev = int(format(x, f"0{n}b")[::-1], 2)
    for base in (x, rev):
        for c in (base, base ^ full):
            for s in range(n):
                y = ((c << s) | (c >> (n - s))) & full
                if best is None or y < best:
                    best = y
    return best


def lumped_exact(n: int, r: int):
    import flint

    masks = ball_masks(n, r)
    full = (1 << n) - 1
    start = canon(0, n)
    index = {start: 0}
    reps = [start]
    trans = []
    k = 0
    while k < len(reps):
        x = reps[k]
        out = {}
        for i in range(n):
            y = canon(fire(x, i, masks, full), n)
            if y not in index:
                index[y] = len(reps)
                reps.append(y)
            out[index[y]] = out.get(index[y], 0) + 1
        trans.append(out)
        k += 1
    N = len(reps)
    # Solve pi (P - I) = 0 with sum pi = 1, i.e. A pi^T = e0, with
    # A = (P^T - I) and first row replaced by ones. Integer form: n*P.
    A = flint.fmpq_mat(N, N)
    for a, out in enumerate(trans):
        for bidx, cnt in out.items():
            A[bidx, a] += flint.fmpq(cnt, n)
        A[a, a] -= 1
    for j in range(N):
        A[0, j] = 1
    rhs = flint.fmpq_mat(N, 1)
    rhs[0, 0] = 1
    pi = A.solve(rhs, algorithm="dixon")
    assert A * pi == rhs  # exact verification of the solution
    EM = flint.fmpq(0)
    for a, x in enumerate(reps):
        pc = bin(x).count("1")
        EM += pi[a, 0] * flint.fmpq(max(pc, n - pc), n)
    return N, Fraction(int(EM.p), int(EM.q))


def band(beta: int, n: int) -> Fraction:
    q = Fraction(beta, n)
    return (2 - q) / (3 - 2 * q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/validation/exact_independent.json")
    ap.add_argument("--no-exact", action="store_true")
    ap.add_argument("--cases", default=None, help="comma list of n to run, e.g. 23,25")
    ap.add_argument("--append", action="store_true", help="merge into existing --out file")
    args = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = []
    if args.append and os.path.exists(args.out):
        out = json.load(open(args.out))
    cases = CASES
    if args.cases:
        keep = {int(x) for x in args.cases.split(",")}
        cases = [c for c in CASES if c[0] in keep]
        out = [o for o in out if o["n"] not in keep]
    for n, r in cases:
        t0 = time.time()
        beta = min(n, 4 * r + 1)
        states, P = reachable_chain(n, r)
        ncomp, lab = connected_components(P, directed=True, connection="strong")
        if P.shape[0] <= 6000:
            pi, method = stationary_direct(P), "sparse direct solve"
        else:
            pi, its = stationary_power(P)
            method = f"lazy power iteration ({its} iterations)"
        resid = float(np.abs(pi @ P - pi).sum())
        EM = float(pi @ majority_fraction(states, n))
        rec = dict(n=n, r=r, beta=beta, q=beta / n,
                   region_II=bool(2 * beta > n and 3 * beta < 2 * n - 1), method=method,
                   reachable_full_colourings=len(states),
                   reachable_mod_colour_swap=len(states) // 2,
                   strongly_connected_components=int(ncomp),
                   min_pi=float(pi.min()), l1_residual=resid,
                   M_float=EM, M_band=float(band(beta, n)),
                   diff=EM - float(band(beta, n)))
        if not args.no_exact:
            norb, Mq = lumped_exact(n, r)
            rec.update(orbits_Dn_x_swap=norb, M_exact_rational=f"{Mq.numerator}/{Mq.denominator}",
                       float_minus_rational=EM - Mq.numerator / Mq.denominator,
                       rational_minus_band=str(Mq - band(beta, n)),
                       below_band_exact=bool(Mq < band(beta, n)))
        rec["seconds"] = round(time.time() - t0, 2)
        print(rec, flush=True)
        out.append(rec)
        out.sort(key=lambda o: o["n"])
        with open(args.out, "w") as f:   # write after every case
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
