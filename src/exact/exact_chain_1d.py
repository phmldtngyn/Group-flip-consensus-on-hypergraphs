"""
exact_chain_1d.py — Exact solver for 1D CYCLE group-flip dynamics.

This is a corrected version that fixes seven independent bugs in the
previous implementation (see docs/validation.md for the audit):

    1. _load_config had a nested function with no return -> None
    2. beta was n//2+1 instead of 4r+1
    3. Ball masks were on a LINE instead of a CYCLE
    4. M was computed as minority/n instead of (n-bit_count)/n
    5. No BFS reachable set; eigs on reducible chain gave invalid pi
    6. r_values / beta_values were not supported in config
    7. Transition rule was XOR (symmetric) instead of case-based
       (i in minority: A -> A \\ B; i in majority: A -> A ∪ B)

The XOR bug (7) made the chain reducible in a way that broke the
stationary solve; it is the direct cause of the
"Stationary distribution degenerate" RuntimeError.

Algorithm outline
-----------------
1. State: minority bitmask on a CYCLE of n nodes. Canonicalized by
   color-complement symmetry (bit_count > n//2 -> complement).
2. BFS from monochrome (integer 0) enumerates the reachable set.
3. Transition: fire at node i (i = 0..n-1):
       if i in minority:  s' = s & ~B_i     (A -> A \\ B)
       else:              s' = s | B_i      (A -> A ∪ B)
   then canonicalize.
4. Stationary distribution: eigs on reachable set; fallback to power
   iteration if eigs returns a non-positive vector (reducible
   fallback safety).
5. Observable M = (n - bit_count(s)) / n = majority fraction.

Classification of numerical output
----------------------------------
- Analytical exact:       M_band (from proof).
- Numerical stationary:   st["M"] (exact Markov chain solve up to tol).
- Rational reconstruction: M_rational (limit_denominator). SUGGESTED
                          form only, not an algebraic certificate.

Feasibility
-----------
- n = 9..21: fast on a laptop.
- n = 23, 25: reachable set is smaller than 2^(n-1); sparse matrix
  on reachable set is manageable.

Author: [author]
Last updated: 2026-11
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import deque
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

try:
    from src.model.balls_1d import exact_1d_quantities, classify_region_1d
except ImportError:  # pragma: no cover
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.abspath(os.path.join(_here, "..", ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from src.model.balls_1d import exact_1d_quantities, classify_region_1d


# ---------------------------------------------------------------------------
# Ball masks on a CYCLE
# ---------------------------------------------------------------------------

def build_ball_masks(n: int, beta: int) -> List[int]:
    """
    Ball masks on a CYCLE, centered at each node.

    For 1D hypergrid with radius r, beta = 4r+1 and
    B_i(r) = {i-2r, ..., i, ..., i+2r}.

    This is the CENTERED ball (firing node at center), required by the
    manuscript's dynamics. A shifted ball (starting at i) would give a
    different transition matrix and different stationary M.
    """
    if beta < 1 or beta > n:
        raise ValueError(f"Invalid beta={beta} for n={n}")
    half = (beta - 1) // 2  # 2r
    masks = []
    for i in range(n):
        mask = 0
        for k in range(beta):
            mask |= 1 << ((i - half + k) % n)
        masks.append(mask)
    return masks


# ---------------------------------------------------------------------------
# Canonicalization (color-complement symmetry)
# ---------------------------------------------------------------------------

def canonicalize(s: int, n: int) -> int:
    """Map state to canonical minority bitmask."""
    full = (1 << n) - 1
    half = n // 2
    bc = s.bit_count()
    if bc > half:
        return s ^ full
    if bc == half and n % 2 == 0 and (s ^ full) < s:
        return s ^ full
    return s


# ---------------------------------------------------------------------------
# Transition rule (case-based, NOT XOR)
# ---------------------------------------------------------------------------

def apply_fire(s: int, i: int, n: int, masks: List[int]) -> int:
    """
    Given minority bitmask s and firing node i, return new canonical state.

    If i is in minority: all minority nodes in B_i flip to majority,
        so A' = A \\ B_i.
    If i is in majority: all majority nodes in B_i flip to minority,
        so A' = A ∪ B_i.

    Then canonicalize (complement if |A'| > n/2).
    """
    mask = masks[i]
    if (s >> i) & 1:
        new_s = s & ~mask          # i in minority: A -> A \ B
    else:
        new_s = s | mask           # i in majority: A -> A ∪ B
    return canonicalize(new_s, n)


# ---------------------------------------------------------------------------
# BFS reachable set from monochrome
# ---------------------------------------------------------------------------

def bfs_reachable(n: int, beta: int) -> List[int]:
    """BFS from monochrome; returns sorted list of reachable canonical states."""
    masks = build_ball_masks(n, beta)
    start = 0
    visited = {start}
    queue = deque([start])
    while queue:
        s = queue.popleft()
        for i in range(n):
            target = apply_fire(s, i, n, masks)
            if target not in visited:
                visited.add(target)
                queue.append(target)
    return sorted(visited)


# ---------------------------------------------------------------------------
# Transition matrix on reachable set
# ---------------------------------------------------------------------------

def build_transition_matrix(
    reachable: List[int], n: int, beta: int
) -> csr_matrix:
    """Build sparse row-stochastic matrix on reachable states (row = current)."""
    masks = build_ball_masks(n, beta)
    state_to_idx = {int(s): i for i, s in enumerate(reachable)}
    prob = 1.0 / n

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for idx, s in enumerate(reachable):
        s_int = int(s)
        for i in range(n):
            target = apply_fire(s_int, i, n, masks)
            j = state_to_idx.get(target)
            if j is None:
                raise RuntimeError(
                    f"BFS incomplete: transition {s_int} -> {target} not "
                    f"in reachable set (n={n}, beta={beta}). This indicates "
                    f"a bug in bfs_reachable or apply_fire."
                )
            rows.append(idx)
            cols.append(j)
            data.append(prob)

    return csr_matrix(
        (data, (rows, cols)), shape=(len(reachable), len(reachable))
    )


# ---------------------------------------------------------------------------
# Stationary distribution: eigs + power iteration fallback
# ---------------------------------------------------------------------------

def stationary_via_eigs(Q: csr_matrix, tol: float = 1e-12, max_iter: int = 100_000):
    """Return (pi, residual) or None if eigs gives a non-positive vector."""
    try:
        vals, vecs = eigs(
            Q.T.astype(np.float64), k=1, which="LR",
            maxiter=max_iter, tol=tol,
        )
        pi = np.real(vecs[:, 0])
        pi = np.clip(pi, 0, None)
        s = pi.sum()
        if s <= 1e-10:
            return None
        pi /= s
        res = float(np.linalg.norm(Q.T @ pi - pi))
        return pi, res
    except Exception as e:
        print(f"  [WARN] eigs failed: {e}", file=sys.stderr)
        return None


def stationary_via_power(
    Q: csr_matrix, tol: float = 1e-13, max_iter: int = 2_000_000,
):
    """Power iteration fallback. Returns (pi, residual, iters)."""
    m = Q.shape[0]
    pi = np.ones(m) / m
    for it in range(max_iter):
        pi = np.asarray(pi @ Q).flatten()
        pi = np.clip(pi, 0, None)
        s = pi.sum()
        if s <= 1e-15:
            raise RuntimeError("Power iteration collapsed to zero vector")
        pi /= s
        if it % 500 == 0 or it == max_iter - 1:
            res = float(np.linalg.norm(Q.T @ pi - pi))
            if res < tol:
                return pi, res, it + 1
    return pi, float(np.linalg.norm(Q.T @ pi - pi)), max_iter


def stationary_majority_1d(
    n: int, beta: int, tol: float = 1e-12, max_iter: int = 100_000,
) -> Dict:
    """Compute stationary distribution and expected majority fraction M."""
    reachable = bfs_reachable(n, beta)
    Q = build_transition_matrix(reachable, n, beta)
    M_vals = np.array(
        [(n - int(s).bit_count()) / n for s in reachable],
        dtype=np.float64,
    )

    method = None
    pi: Optional[np.ndarray] = None
    residual: Optional[float] = None

    result = stationary_via_eigs(Q, tol=tol, max_iter=max_iter)
    if result is not None:
        pi_candidate, residual_candidate = result
        if residual_candidate < 1e-6:
            pi, residual, method = pi_candidate, residual_candidate, "eigs"

    if pi is None:
        pi, residual, iters = stationary_via_power(Q, tol=tol * 100)
        method = f"power({iters})"

    assert pi is not None and residual is not None
    M = float(np.dot(pi, M_vals))

    return {
        "M": M,
        "pi": pi,
        "residual": residual,
        "method": method,
        "n_states": (1 << n) >> 1,
        "n_states_reachable": len(reachable),
    }


# ---------------------------------------------------------------------------
# Beta resolution
# ---------------------------------------------------------------------------

def resolve_beta_from_r(n: int, r: int) -> int:
    """1D hypergrid: beta = min(n, 4r+1)."""
    return min(n, 4 * r + 1)


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

def run_grid(
    n_list: Tuple[int, ...],
    r_list: Optional[Tuple[int, ...]] = None,
    beta_list: Optional[Tuple[int, ...]] = None,
    tol: float = 1e-12,
    max_iter: int = 100_000,
) -> List[Dict]:
    if r_list is None and beta_list is None:
        raise ValueError("Must provide r_list or beta_list")
    if r_list is not None and len(r_list) != len(n_list):
        raise ValueError(f"len mismatch: n={len(n_list)} r={len(r_list)}")
    if beta_list is not None and len(beta_list) != len(n_list):
        raise ValueError(f"len mismatch: n={len(n_list)} beta={len(beta_list)}")

    recs: List[Dict] = []
    for idx, n in enumerate(n_list):
        if beta_list is not None:
            beta = int(beta_list[idx])
        else:
            beta = resolve_beta_from_r(n, int(r_list[idx]))

        print(f"  n={n:3d}  beta={beta:3d}  ...", end=" ", flush=True)

        try:
            ex = exact_1d_quantities(n, beta)
            M_band = float(ex["M_band"])
        except Exception as e:  # pragma: no cover
            M_band = float("nan")
            print(f"[band failed: {e}]", end=" ", file=sys.stderr)

        st = stationary_majority_1d(n, beta, tol=tol, max_iter=max_iter)
        frac = Fraction(st["M"]).limit_denominator(10 ** 6)
        diff = st["M"] - M_band
        H3 = bool(st["M"] > M_band + 1e-10)

        try:
            region = classify_region_1d(n, beta)
        except Exception:  # pragma: no cover
            region = "unknown"

        rec = {
            "n": n,
            "r": (beta - 1) // 4 if beta % 4 == 1 else None,
            "beta": beta,
            "q": beta / n,
            "M": st["M"],
            "M_band": M_band,
            "M_rational": f"{frac.numerator}/{frac.denominator}",
            "M_rational_method": "numerical_reconstruction",
            "diff": diff,
            "H3_violation": H3,
            "region": region,
            "residual": st["residual"],
            "method": st["method"],
            "n_states": st["n_states"],
            "n_states_reachable": st["n_states_reachable"],
        }
        recs.append(rec)
        print(
            f"M={st['M']:.10f}  M_band={M_band:.10f}  "
            f"diff={diff:+.3e}  reach={st['n_states_reachable']}  "
            f"[{st['method']}]  res={st['residual']:.2e}"
        )

    return recs


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_results(recs: List[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    csv_path = out_path.rsplit(".", 1)[0] + ".csv"
    if recs:
        keys = list(recs[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in recs:
                w.writerow(r)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=2)
    print(f"Saved {csv_path} and {out_path}")


def _load_config(path: str) -> Dict:
    """Load YAML config. Requires pyyaml."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "pyyaml is required for --config. Install with `pip install pyyaml`."
        ) from e
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(
            f"Config '{path}' contains no YAML data (empty or comment-only)."
        )
    return cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Exact solver for 1D cycle group-flip dynamics."
    )
    parser.add_argument("--n", type=int, nargs="+", default=None)
    parser.add_argument("--r", type=int, nargs="+", default=None)
    parser.add_argument("--beta", type=int, nargs="+", default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--out", type=str, default="results/tables/exact_b_crosscheck.json"
    )
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max-iter", type=int, default=100_000)
    args = parser.parse_args(argv)

    n_list: Optional[Tuple[int, ...]] = None
    r_list: Optional[Tuple[int, ...]] = None
    beta_list: Optional[Tuple[int, ...]] = None
    tol = args.tol
    max_iter = args.max_iter

    if args.config:
        cfg = _load_config(args.config)
        if "n_values" in cfg:
            n_list = tuple(cfg["n_values"])
        if "r_values" in cfg:
            r_list = tuple(cfg["r_values"])
        if "beta_values" in cfg:
            beta_list = tuple(cfg["beta_values"])
        tol = float(cfg.get("solver_tolerance", tol))
        max_iter = int(cfg.get("max_iter", max_iter))
    else:
        if args.n:
            n_list = tuple(args.n)
        if args.r:
            r_list = tuple(args.r)
        if args.beta:
            beta_list = tuple(args.beta)

    if n_list is None:
        n_list = (9, 15, 17, 21, 23, 25)

    if r_list is None and beta_list is None:
        if n_list == (9, 15, 17, 21, 23, 25):
            r_list = (1, 2, 2, 3, 3, 3)
        else:
            raise ValueError(
                f"No r or beta for n_list={n_list}. Use --r, --beta, or --config."
            )

    print(f"Running exact solver: n = {n_list}")
    if r_list is not None:
        print(f"  r = {r_list}")
        preview = tuple(resolve_beta_from_r(n, r) for n, r in zip(n_list, r_list))
        print(f"  beta = {preview}")
    elif beta_list is not None:
        print(f"  beta = {beta_list}")
    print(f"  tol = {tol:g}, max_iter = {max_iter}")
    print()

    recs = run_grid(
        n_list=n_list, r_list=r_list, beta_list=beta_list,
        tol=tol, max_iter=max_iter,
    )
    save_results(recs, args.out)


if __name__ == "__main__":
    main()