"""
Fast unit and regression tests (run with:  pytest -q tests/unit_tests.py).

Revision audit: the previous version imported a function that does not
exist (balls_1d_bitmask) and passed r where beta was expected, so it could
not run. These tests cover the closed forms, the region labels, a small
exact chain with a certified rational value, and the headline statistics.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "model"))
sys.path.insert(0, str(ROOT / "tests" / "independent"))

from balls_1d import exact_1d_quantities, classify_region_1d, band_value  # noqa: E402
import containment_bruteforce as cb  # noqa: E402
import exact_chain_independent as ec  # noqa: E402


def test_containment_closed_forms_match_enumeration():
    for n in range(12, 31):
        r = 1
        while True:
            beta, cg, ct = cb.enumerate_containment(n, r)
            fcg, fct = cb.closed_forms(n, beta)
            assert cg == fcg and ct == fct, (n, r)
            q = exact_1d_quantities(n, beta)
            assert abs(q["cG"] - float(fcg)) < 1e-12
            assert abs(q["ctilde"] - float(fct)) < 1e-12
            if beta == n:
                break
            r += 1


def test_boundary_case_18_2():
    beta, cg, ct = cb.enumerate_containment(18, 2)
    assert beta == 9 and cg == Fraction(1, 18) and ct == Fraction(1, 9)
    assert classify_region_1d(18, 9) == "boundary"


def test_region_labels():
    assert classify_region_1d(25, 13) == "II"
    assert classify_region_1d(9, 5) == "II"       # beta = (n+1)/2 lies in Region II
    assert classify_region_1d(13, 9) == "I"
    assert classify_region_1d(10, 3) == "III"


def test_band_function():
    assert abs(band_value(2, 1) - 0.75) < 1e-15      # f(1/2) = 3/4
    assert abs(band_value(3, 2) - 0.8) < 1e-15       # f(2/3) = 4/5


def test_exact_chain_n9_rational():
    norb, M = ec.lumped_exact(9, 1)
    assert norb == 8 and M == Fraction(43, 57)
    assert M < Fraction(13, 17)                      # below f(5/9) = 13/17


def test_reanalysis_headline_if_present():
    """Regression test on the released outputs (skipped if not generated)."""
    p = ROOT / "results/tables/reanalysis/reanalysis_report.json"
    if not p.exists():
        return
    r = json.load(open(p))
    prim = r["bootstrap"]["stated|strat|B2000"]
    assert abs(prim["gamma_point"] - (-0.01959)) < 1e-4
    assert prim["gamma_ci_hi"] < 0
    assert prim["reduction_all6_point"] < 0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
