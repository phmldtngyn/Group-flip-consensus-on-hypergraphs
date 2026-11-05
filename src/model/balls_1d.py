"""
balls_1d.py — Analytical quantities for the 1D chain group-flip model.

Definitions
-----------
For a 1D chain of n nodes and contiguous balls of size beta:

    R1  : 2 * beta > n            (strict high-coverage regime)
    cG  : (2 * beta - n + 1) / n  (Eq. 11 in Main Text)
    c~  : (2 * beta - n + 1) / (n - beta)
    R2  : c~ >= 1                 (second assumption needed for the proof)

Regions
-------
    low_coverage   :  2 * beta <= n         (R1 fails; no analytic M_band)
    high_coverage  :  2 * beta > n, beta < n (R1 holds; R2 may or may not)
    full_coverage  :  beta >= n             (degenerate; ball covers whole chain)

Notes on cG and c~
------------------
- cG can exceed 1 when beta = n (e.g. n=12, beta=12 gives cG = 13/12 > 1).
  We clip cG at 1.0 to keep it in [0, 1], but keep the raw value in
  `cG_raw` for diagnostics.
- When beta >= n, the denominator (n - beta) is zero or negative.
  In that case, c~ is set to 1.0 by convention (full-coverage limit),
  and `ctilde_raw` is set to +inf.

Placeholder for M_band
----------------------
The exact formula for M_band must come from the proof in the paper
(Section 3 / Appendix B). The current expression is a *structural placeholder*
so the solver can run end-to-end. Replace `_M_band_formula` below with the
formula from the manuscript before submission.

Author: [author]
Last updated: 2026-11
"""

from __future__ import annotations

from typing import Dict

import math


# ---------------------------------------------------------------------------
# Region classification
# ---------------------------------------------------------------------------

def classify_region_1d(n: int, beta: int) -> str:
    """
    Region of the 1D hypergrid (main text Theorem 4; SM Prop. E.3).

    Returns one of:
        'full'      : beta >= n (empty residuals)
        'I'         : 3*beta >= 2n-1      (R1 and R2 hold; band applies)
        'II'        : n/2 < beta < (2n-1)/3   (R1 holds, R2 fails)
        'boundary'  : 2*beta == n
        'III'       : 2*beta < n          (R1 fails)

    (Revision audit: the previous labels 'low/high_coverage' put Region II
    under 'high_coverage', which is wrong.)
    """
    if beta >= n:
        return "full"
    if 3 * beta >= 2 * n - 1:
        return "I"
    if 2 * beta > n:
        return "II"
    if 2 * beta == n:
        return "boundary"
    return "III"


# ---------------------------------------------------------------------------
# Band function
# ---------------------------------------------------------------------------

def band_value(n: int, beta: int) -> float:
    """
    Band function f(q) = (2 - q) / (3 - 2 q) with q = beta / n
    (main text Theorem 2). It equals E[M] only when R1 and R2 hold with
    equal balls; elsewhere it is a comparator (Conjecture 1).

    (Revision audit: this replaces a placeholder expression
    0.5(1-cG) + 0.5(1 - ctilde/2) that was not the proven formula.)
    """
    q = min(beta, n) / n
    return (2.0 - q) / (3.0 - 2.0 * q)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def exact_1d_quantities(n: int, beta: int) -> Dict[str, object]:
    """
    Compute all analytical quantities for the 1D chain at (n, beta).

    Parameters
    ----------
    n : int
        Number of nodes (>= 1).
    beta : int
        Ball size (>= 1). May exceed n (full coverage).

    Returns
    -------
    dict with keys:
        M_band       : float, f(beta/n); exact only if band_applies
        band_applies : bool  (R1 and R2)
        cG           : float in [0, 1] (clipped)
        cG_raw       : float (unclipped; can exceed 1)
        ctilde       : float in (0, 1]  (clipped)
        ctilde_raw   : float (unclipped; can be > 1 or inf)
        R1           : bool   (2 * beta > n)
        R2           : bool   (ctilde_raw >= 1)
        region       : str    ('I', 'II', 'boundary', 'III', 'full')
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if beta < 1:
        raise ValueError(f"beta must be >= 1, got {beta}")

    region = classify_region_1d(n, beta)

    # --- R1 (strict high coverage) -----------------------------------------
    # Equivalent to beta > n/2. For odd n, equivalent to beta >= (n+1)/2.
    # The plan explicitly warns against stating "beta > (n+1)/2" — that is
    # incorrect for even n. Use R1 = 2 * beta > n.
    R1 = (2 * beta > n)

    # --- cG, c~ -------------------------------------------------------------
    num = 2 * beta - n + 1
    cG_raw = num / n

    denom_beta = n - beta
    if denom_beta > 0:
        ctilde_raw = num / denom_beta
    else:
        # Full coverage (beta >= n): denominator is zero or negative.
        # Convention: ctilde = 1.0 in the full-coverage limit.
        ctilde_raw = math.inf

    # Clipping: keep cG, ctilde in [0, 1] for the analytical formulas that
    # require them to be probabilities. Raw values are preserved for
    # diagnostics and for the R2 test.
    cG = min(1.0, max(0.0, cG_raw))
    ctilde = 1.0 if math.isinf(ctilde_raw) else min(1.0, max(0.0, ctilde_raw))

    # --- R2 (second assumption; c~ >= 1 in the unclipped sense) -----------
    R2 = (ctilde_raw >= 1.0)

    # --- band function: exact if R1 and R2, comparator otherwise ---------
    M_band = band_value(n, beta)

    return {
        "M_band": M_band,
        "band_applies": bool(R1 and R2),
        "cG": cG,
        "cG_raw": cG_raw,
        "ctilde": ctilde,
        "ctilde_raw": ctilde_raw,
        "R1": R1,
        "R2": R2,
        "region": region,
    }