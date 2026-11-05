"""
containment_bruteforce.py -- brute-force check of the exact 1D containment
formulas (main text Eqs. (12)-(13); SM Props. E.1-E.2) and data source for
Figure 3.

For every n in [12, 40] and every radius r >= 1 with 4r+1 <= n (plus the
full-coverage radius), balls of the 1D hypergrid are built from the
hypergraph distance d(i,j) = ceil(delta(i,j)/2), and

    c_G     = (1/n^2) * #{(i,j) : R_i subset of B_j}
    c_tilde = sum_i #{j in R_i : R_i subset of B_j} / sum_i |R_i|

are computed by explicit set inclusion (no formula is used). Results are
compared with the closed forms. Output JSON contains every (n, r) record.

Usage
-----
    python tests/independent/containment_bruteforce.py \
        --out results/validation/containment_bruteforce.json
"""
from __future__ import annotations

import argparse
import json
import os
from fractions import Fraction


def balls(n, r):
    out = []
    for i in range(n):
        out.append(frozenset(j for j in range(n)
                             if (min((i - j) % n, (j - i) % n) + 1) // 2 <= r))
    return out


def enumerate_containment(n, r):
    B = balls(n, r)
    V = frozenset(range(n))
    R = [V - b for b in B]
    cg = sum(1 for i in range(n) for j in range(n) if R[i] <= B[j])
    num = sum(1 for i in range(n) for j in R[i] if R[i] <= B[j])
    den = sum(len(x) for x in R)
    ct = Fraction(num, den) if den else Fraction(1)
    return len(B[0]), Fraction(cg, n * n), ct


def closed_forms(n, beta):
    cg = min(Fraction(1), Fraction(max(0, 2 * beta - n + 1), n))
    if 2 * beta < n:
        ct = Fraction(0)
    elif 3 * beta >= 2 * n - 1:
        ct = Fraction(1)
    else:
        ct = Fraction(2 * beta - n + 1, n - beta)
    return cg, ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/validation/containment_bruteforce.json")
    ap.add_argument("--nmin", type=int, default=12)
    ap.add_argument("--nmax", type=int, default=40)
    a = ap.parse_args()
    recs, maxerr = [], 0.0
    for n in range(a.nmin, a.nmax + 1):
        r = 1
        while True:
            beta, cg, ct = enumerate_containment(n, r)
            fcg, fct = closed_forms(n, beta)
            err = max(abs(float(cg - fcg)), abs(float(ct - fct)))
            maxerr = max(maxerr, err)
            recs.append(dict(n=n, r=r, beta=beta, q=beta / n,
                             cG_enum=str(cg), ctilde_enum=str(ct),
                             cG_formula=str(fcg), ctilde_formula=str(fct),
                             exact_match=bool(cg == fcg and ct == fct)))
            if beta == n:
                break
            r += 1
    bnd = next(x for x in recs if x["n"] == 18 and x["r"] == 2)
    summary = dict(cases=len(recs), all_exact_match=all(x["exact_match"] for x in recs),
                   max_abs_error=maxerr, boundary_case_18_2=bnd)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(summary=summary, records=recs), open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
