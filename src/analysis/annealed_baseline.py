r"""Annealed-ball baseline for two-colour group-flip dynamics.
At each fire of node i (rate 1 per node) the interaction set is redrawn:
B = {i} plus beta-1 nodes drawn uniformly without replacement from V\{i}.
All nodes of i's colour in B adopt the other colour. The dynamics then
depends only on the count k of colour-1 nodes; we solve the stationary law of
that birth-death-type chain exactly (rational arithmetic for small n,
double precision otherwise) and report E[M] = sum_k pi_k max(k,n-k)/n.
Usage: python3 annealed_baseline.py"""
from fractions import Fraction
from math import comb
import numpy as np, sys

def generator(n, beta, exact):
    one = Fraction(1) if exact else 1.0
    Q = [[0*one for _ in range(n+1)] for _ in range(n+1)]
    D = comb(n-1, beta-1)
    for k in range(n+1):
        for colour, m in ((1, k), (0, n-k)):
            if m == 0: continue
            # node of this colour fires: total rate m; moved = 1 + h
            for h in range(0, min(m-1, beta-1)+1):
                w = comb(m-1, h)*comb(n-m, beta-1-h)
                if w == 0: continue
                rate = one*m*w/D if not exact else Fraction(m*w, D)
                moved = 1+h
                k2 = k-moved if colour == 1 else k+moved
                Q[k][k2] += rate; Q[k][k] -= rate
    return Q

def stationary(n, beta, exact=False):
    Q = generator(n, beta, exact)
    if exact:
        import sympy
        M = sympy.Matrix(Q).T
        M = M.row_insert(0, sympy.ones(1, n+1))
        b = sympy.zeros(n+2, 1); b[0] = 1
        pi = M.solve_least_squares(b) if False else None
        # solve exactly: replace last equation by normalisation
        A = sympy.Matrix(Q).T
        A[n, :] = sympy.ones(1, n+1)
        rhs = sympy.zeros(n+1, 1); rhs[n] = 1
        pi = list(A.LUsolve(rhs))
        EM = sum(p*Fraction(max(k, n-k), n) for k, p in enumerate(pi))
        return sympy.nsimplify(EM), float(EM)
    A = np.array(Q, dtype=float).T
    A[n, :] = 1.0
    rhs = np.zeros(n+1); rhs[n] = 1.0
    pi = np.linalg.solve(A, rhs)
    EM = sum(p*max(k, n-k)/n for k, p in enumerate(pi))
    res = np.abs(np.array(Q, dtype=float).T @ pi).sum()
    return EM, res

f = lambda q: (2-q)/(3-2*q)
if __name__ == "__main__":
    print("# Region II cases (same n, beta as the quenched exact table)")
    for n, r in [(9,1),(15,2),(16,2),(17,2),(21,3),(22,3),(23,3),(24,3),(25,3),(27,4),(28,4),(29,4),(30,4),(31,4),(32,4),(33,4),(33,5)]:
        b = 4*r+1; EM, res = stationary(n, b)
        print(f"II n={n} r={r} beta={b} q={b/n:.4f} annealed={EM:.10f} f={f(b/n):.10f} diff={EM-f(b/n):+.6f} res={res:.1e}")
    print("# Region I, native 1D sizes used in the validation (n=400, r=75; n=401, r=70)")
    for n, b in [(400, 301), (401, 281), (13, 9), (11, 9)]:
        EM, res = stationary(n, b)
        print(f"I n={n} beta={b} q={b/n:.4f} annealed={EM:.10f} f={f(b/n):.10f} diff={EM-f(b/n):+.6f} res={res:.1e}")
    print("# coverage grid at n=400")
    for q in [0.55,0.60,0.65,0.70,0.75,0.80,0.90]:
        b = round(q*400); EM, res = stationary(400, b)
        print(f"G n=400 beta={b} q={b/400:.4f} annealed={EM:.10f} f={f(b/400):.10f} diff={EM-f(b/400):+.6f} res={res:.1e}")
