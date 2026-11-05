"""
run_all.py -- reproduce every number, table and figure in the manuscript.

Steps (each writes its own outputs; see README.md, "Paper-to-code map"):

  1. unit       pytest tests/unit_tests.py                                  (~1 s)
  2. contain    tests/independent/containment_bruteforce.py                 (~1 s)
  3. exact      tests/independent/exact_chain_independent.py               (~1 min)
  4. exact_b    src/exact/exact_chain_1d.py (repository solver, cross-check) (~5 min)
  5. suite      tests/gvm_validation_suite.py --task T1..T6                 (~1 min)
  6. sim        tests/independent/sim_independent.py                        (~2 min)
     newborn    scripts/rerun_newborn_v2.py (newborn identity, truncated sets) (~5 min)
  7. reanalysis src/analysis/reanalysis_v2.py  (B = 2000)                   (~25 min)
  8. figures    src/visualization/make_figures.py
  9. numbers    scripts/make_numbers_tex.py
 10. latex      scripts/build_manuscript.sh  (needs pdflatex + elsarticle)

The V2 simulations themselves (src/simulation/measure_pipeline.py, several
hours) are NOT re-run by default; pass --with-measurements to regenerate
them into data/raw/v2_rerun/ (the released data/processed/measurements_v2.csv
is never overwritten).

Usage
-----
    python scripts/run_all.py                 # all steps except measurements
    python scripts/run_all.py --only reanalysis figures numbers
    python scripts/run_all.py --quick         # smoke test: reanalysis with B = 200, written to
                                              # results/tables/reanalysis_quick/ (released
                                              # results/tables/reanalysis/ is not modified)

A step is reported PASS only if its command exits with status 0.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def steps(quick: bool, with_meas: bool):
    B = "200" if quick else "2000"
    s = [
        ("unit", [PY, "-m", "pytest", "-q", "tests/unit_tests.py"]),
        ("contain", [PY, "tests/independent/containment_bruteforce.py"]),
        ("exact", [PY, "tests/independent/exact_chain_independent.py"]),
        ("exact_b", [PY, "src/exact/exact_chain_1d.py", "--config", "configs/exact_grid.yaml",
                     "--out", "results/tables/exact_b_crosscheck.json"]),
    ]
    s += [(f"suite_{t}", [PY, "tests/gvm_validation_suite.py", "--task", t,
                          "--out", "results/validation/suite_rerun"])
          for t in ("T1", "T2", "T3", "T4", "T5", "T6")]
    s += [
        ("sim", [PY, "tests/independent/sim_independent.py"]),
        ("newborn", [PY, "scripts/rerun_newborn_v2.py"]),
        # --quick writes to a scratch directory so released outputs are untouched
        ("reanalysis", [PY, "src/analysis/reanalysis_v2.py", "--B", B] +
         (["--out", "results/tables/reanalysis_quick"] if quick else [])),
        ("figures", [PY, "src/visualization/make_figures.py", "--outdir", "manuscript/figures"]),
        ("numbers", [PY, "scripts/make_numbers_tex.py"]),
        ("latex", ["bash", "scripts/build_manuscript.sh"]),
    ]
    if with_meas:
        s.insert(0, ("measurements", [PY, "src/simulation/measure_pipeline.py",
                                      "--out", "data/raw/v2_rerun"]))
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="run only these step names")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--with-measurements", action="store_true")
    a = ap.parse_args()
    (ROOT / "results/logs").mkdir(parents=True, exist_ok=True)
    status = {}
    for name, cmd in steps(a.quick, a.with_measurements):
        if a.only and name not in a.only and not any(name.startswith(o + "_") for o in a.only):
            continue
        t0 = time.time()
        log = ROOT / "results/logs" / f"run_all_{name}.log"
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        status[name] = "PASS" if rc == 0 else f"FAILED (exit {rc}; see {log.relative_to(ROOT)})"
        print(f"[{status[name]}] {name}  ({time.time() - t0:.0f} s)", flush=True)
    bad = [k for k, v in status.items() if v != "PASS"]
    print("\nAll steps passed." if not bad else f"\nFailed steps: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
