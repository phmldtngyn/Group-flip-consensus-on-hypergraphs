"""
make_figures.py -- regenerate manuscript Figures 2, 3 and 4 from computed
outputs only (no synthetic or illustrative data).

    Fig. 2  results/validation/exact_independent.json        (exact chain)
    Fig. 3  results/validation/containment_bruteforce.json   (enumeration)
    Fig. 4  data/processed/measurements_v2.csv               (simulations)
            results/tables/reanalysis/boot_stated_strat_B2000.csv (primary)
            results/tables/reanalysis/boot_stated_iid_B2000.csv   (sensitivity)

Figure 1 (model schematic) is produced by src/visualization/fig15.py and is
a drawing, not data.

Usage
-----
    python src/visualization/make_figures.py --outdir manuscript/figures
"""
from __future__ import annotations

import argparse
import json
import os
from fractions import Fraction

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": 0.8,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "savefig.dpi": 300, "mathtext.fontset": "cm",
})

INK = "#222222"
GREY = "#8a8a8a"
FAM_STYLE = {"1D": ("#1b1b1b", "o"), "2D": ("#c0392b", "s"),
             "WS": ("#2e6db4", "^"), "config": ("#2a8a4a", "D")}


def band(q):
    return (2 - q) / (3 - 2 * q)


def frac(s):
    return float(Fraction(s))


def fig2(exact_json, out):
    recs = sorted(json.load(open(exact_json)), key=lambda x: x["q"])
    q = np.array([r["q"] for r in recs])
    M = np.array([frac(r["M_exact_rational"]) if "M_exact_rational" in r else r["M_float"] for r in recs])
    lab = [f"({r['n']},{r['r']})" for r in recs]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    qq = np.linspace(0.5, 0.72, 200)
    a = ax[0]
    a.axvspan(2 / 3, 0.72, color="#ececec", lw=0)
    a.plot(qq, band(qq), color=INK, lw=1.2, label=r"$f(q)=(2-q)/(3-2q)$")
    a.axhline(0.75, color=GREY, ls="--", lw=0.7)
    a.axhline(0.80, color=GREY, ls=":", lw=0.7)
    a.plot(q, M, "o", ms=4.5, mfc="white", mec=INK, mew=1.0, label="exact chain (Table 1)")
    off = {"(9,1)": (-24, 4), "(23,3)": (4, -9)}
    for x, y, t in zip(q, M, lab):
        a.annotate(t, (x, y), xytext=off.get(t, (3, -9)), textcoords="offset points",
                   fontsize=6.5, color=GREY)
    a.text(0.693, 0.725, "Thm. 2\ndomain\n(asympt.)", fontsize=6.2, color=GREY,
           va="bottom", ha="center")
    a.set_xlim(0.5, 0.72)
    a.set_ylim(0.72, 0.82)
    a.set_xlabel(r"coverage $q=\beta/n$")
    a.set_ylabel(r"stationary majority $E[M]$")
    a.legend(loc="upper left", frameon=False)
    a.set_title("(a)", loc="left", fontsize=9)
    b = ax[1]
    b.axhline(0, color=INK, lw=0.8)
    b.plot(q, M - band(q), "o", ms=4.5, mfc="white", mec=INK, mew=1.0)
    offb = {"(9,1)": (-26, -3)}
    for x, y, t in zip(q, M - band(q), lab):
        b.annotate(t, (x, y), xytext=offb.get(t, (4, -3)), textcoords="offset points",
                   fontsize=6.5, color=GREY)
    b.set_xlabel(r"coverage $q=\beta/n$")
    b.set_ylabel(r"$M_{\rm exact}-f(q)$")
    b.set_xlim(0.5, 0.65)
    b.set_title("(b)", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png")
    plt.close(fig)


def fig3(cont_json, out, n=40):
    recs = [r for r in json.load(open(cont_json))["records"] if r["n"] == n]
    q = np.array([r["q"] for r in recs])
    cg = np.array([frac(r["cG_enum"]) for r in recs])
    ct = np.array([frac(r["ctilde_enum"]) for r in recs])
    # closed forms as continuous curves over beta (odd beta only are realised)
    betas = np.arange(1, n + 1)
    qb = betas / n
    cg_f = np.minimum(1, np.maximum(0, 2 * betas - n + 1) / n)
    ct_f = np.where(2 * betas < n, 0.0,
                    np.where(3 * betas >= 2 * n - 1, 1.0, (2 * betas - n + 1) / np.maximum(n - betas, 1)))
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for a, f, e, ylab, eq in ((ax[0], cg_f, cg, r"$c_G$", "Eq. (12)"),
                              (ax[1], ct_f, ct, r"$\tilde c$", "Eq. (13)")):
        a.step(qb, f, where="post", color=GREY, lw=0.9, label=f"{eq} (all integer $\\beta$)")
        a.plot(q, e, "o", ms=4.2, mfc="white", mec=INK, mew=1.0,
               label=r"enumeration, $\beta=\min(n,4r+1)$")
        a.set_xlabel(r"$q=\beta/n$")
        a.set_ylabel(ylab)
        a.set_ylim(-0.05, 1.08)
        a.legend(frameon=False, loc="upper left")
    for x in (0.5, (2 * n - 1) / (3 * n)):
        ax[1].axvline(x, color=GREY, ls="--", lw=0.6)
    for x, t in ((0.3, "III"), (0.58, "II"), (0.85, "I")):
        ax[1].text(x, 0.45, t, color=GREY, ha="center")
    ax[0].set_title(f"(a)  $n={n}$", loc="left", fontsize=9)
    ax[1].set_title(f"(b)  $n={n}$", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png")
    plt.close(fig)


def fig4(meas_csv, boot_primary, boot_iid, out, n_show=400):
    d = pd.read_csv(meas_csv)
    bp = pd.read_csv(boot_primary)
    bi = pd.read_csv(boot_iid)
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.45))
    a = ax[0]
    qq = np.linspace(0.54, 0.81, 100)
    a.plot(qq, band(qq), color=GREY, lw=0.9, ls="--", label=r"$f(q)$")
    sub = d[d.n == n_show]
    for fam, (col, mk) in FAM_STYLE.items():
        g = sub[sub.family == fam].groupby("q")
        m = g.M.mean()
        # between-seed standard error of the mean over 3 seeds
        se = g.M.std(ddof=1) / np.sqrt(g.M.size())
        off = {"1D": -0.006, "2D": -0.002, "WS": 0.002, "config": 0.006}[fam]
        a.errorbar(m.index + off, m.values, yerr=se.values, fmt=mk, ms=3.2, color=col,
                   mfc="white", mew=0.9, elinewidth=0.8, capsize=0, label=fam)
    a.set_xlabel(r"$q$")
    a.set_ylabel(r"$M$")
    a.set_title(f"(a)  $n={n_show}$", loc="left", fontsize=9)
    a.legend(frameon=False, loc="upper left", ncol=2, fontsize=6.2, handletextpad=0.3,
             columnspacing=0.8)
    b = ax[1]
    bins = np.linspace(min(bp.gamma.min(), bi.gamma.min()), max(bp.gamma.max(), bi.gamma.max()), 45)
    b.hist(bi.gamma, bins=bins, color="#c9d6e8", edgecolor="none", label="runs iid")
    b.hist(bp.gamma, bins=bins, histtype="step", color=INK, lw=1.0, label="stratified\ncluster\n(primary)")
    lo, hi = np.percentile(bp.gamma, [2.5, 97.5])
    for x in (lo, hi):
        b.axvline(x, color=INK, ls=":", lw=0.8)
    b.axvline(0, color="#c0392b", lw=0.8)
    b.set_xlabel(r"bootstrap $\hat\gamma$")
    b.set_ylabel("count")
    b.set_title("(b)", loc="left", fontsize=9)
    b.set_xlim(bins[0], 0.003)
    b.legend(frameon=False, loc="center right", fontsize=6)
    c = ax[2]
    v = bp["reduction_all6"].replace([np.inf, -np.inf], np.nan).dropna()
    c.hist(v, bins=45, histtype="step", color=INK, lw=1.0)
    c.set_xlim(min(-1.5, v.min()), 0.6)
    c.axvspan(0.30, 0.6, color="#f3e3a6", lw=0, label="acceptance\n" + r"($\geq 0.30$)")
    c.axvline(0, color=GREY, lw=0.8)
    c.axvline(np.median(v), color=INK, ls=":", lw=0.8)
    c.set_xlabel("family-norm reduction")
    c.set_ylabel("count")
    c.set_title("(c)", loc="left", fontsize=9)
    c.legend(frameon=False, loc="upper right", fontsize=6)
    fig.tight_layout()
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="manuscript/figures")
    ap.add_argument("--exact", default="results/validation/exact_independent.json")
    ap.add_argument("--containment", default="results/validation/containment_bruteforce.json")
    ap.add_argument("--meas", default="data/processed/measurements_v2.csv")
    ap.add_argument("--boot-primary", default="results/tables/reanalysis/boot_stated_strat_B2000.csv")
    ap.add_argument("--boot-iid", default="results/tables/reanalysis/boot_stated_iid_B2000.csv")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    fig2(a.exact, os.path.join(a.outdir, "fig2_exact_band"))
    fig3(a.containment, os.path.join(a.outdir, "fig3_containment"))
    fig4(a.meas, a.boot_primary, a.boot_iid, os.path.join(a.outdir, "fig4_controlled_test"))
    print("figures written to", a.outdir)


if __name__ == "__main__":
    main()
