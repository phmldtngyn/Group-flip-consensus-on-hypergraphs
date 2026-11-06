"""
make_figures.py — Generate all figures of the paper from the JSON data.

Outputs (PDF, vector): figures/fig1..fig7
Style: two-column PRE, ~3.35 in column width, 8-9 pt labels.

Data inputs (data/ directory):
  crossover_{config,2d,3d}.json     crossover scans (radii tables)
  artifact_{config_n999,nscan,2d_L40}.json   m-dimming data
  verification.json                 hypergeometric benchmark
  quasi_consensus.json              parameter-point series
  partial_conversion.json           partial conversion + extinction
  regen_e2c3_overlay.json           coverage-dynamics overlay (14 systems)
  table_qdyn.json                   rebuilt Table VII (curve crossings;
                                    produced by scripts/analysis_tables.py)

Changes vs the previous release:
  * fig2 now shows the ten systems of the rebuilt Table VII (curve
    crossings, bootstrap CIs, family means) instead of the six legacy
    integer-interpolation values.
  * fig3 writes fig3_mdimming.pdf (the filename the main text includes).
  * new fig7_overlay draws the exact band curve
    E[M] = (2 - beta/n)/(3 - 2 beta/n) on top of the collapse: solid on
    its 2D validity domain beta/n >= 7/9 (shaded), dotted as an
    extrapolation below (Theorem 6; data x-axes converted from
    q_H = (beta-1)/(n-1) to beta/n, a difference <= 2e-3).
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE_DIR, "data")
FIG = os.path.join(BASE_DIR, "figures")

os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "lines.linewidth": 1.2, "mathtext.fontset": "cm",
})

def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def fit_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ [a, b]
    serr = float(np.sqrt(resid @ resid / (len(x) - 2))
                 / np.sqrt(((x - x.mean()) ** 2).sum()))
    return a, b, serr


# =====================================================================
# Fig 1: crossover radius vs system size (3 panels)
# =====================================================================
def fig1():
    cfg = load("crossover_config.json")["sizes"]
    d2 = load("crossover_2d.json")["sizes"]
    d3 = load("crossover_3d.json")["sizes"]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), constrained_layout=True)

    ns = sorted(int(s) for s in cfg)
    rt = [cfg[str(n)]["r_tilde_dyn"] for n in ns]
    lnns = [np.log(n) for n in ns]
    a, b, sa = fit_slope(lnns, rt)
    ax = axes[0]
    ax.errorbar(lnns, rt, yerr=0.08, fmt="o", ms=3.5, color="C0", capsize=2)
    xs = np.linspace(min(lnns), max(lnns), 10)
    ax.plot(xs, a * xs + b, "-", color="C0",
            label=f"fit {a:.2f}$\\,\\ln n{{-}}${abs(b):.2f}")
    ax.plot(xs, xs / np.log(6), "--", color="k", lw=0.9,
            label=r"$1/\ln b=0.558$ (cond.)")
    ax.set_xlabel(r"$\ln n$")
    ax.set_ylabel(r"$\tilde r_{\rm dyn}$")
    ax.set_title("(a) config model")
    ax.legend(frameon=False, loc="upper left")

    for ax, d, lab, th, pw in ((axes[1], d2, "(b) 2D hypergrid", 0.5, 2),
                               (axes[2], d3, "(c) 3D hypergrid", 1 / 3, 3)):
        pairs = sorted(((int(s) ** pw, d[s]["r_dyn"], d[s]["r_tilde_dyn"])
                        for s in d))               # (n = L^d, r_dyn, r~)
        ns = [p[0] for p in pairs]
        rint = [p[1] for p in pairs]
        rt = [p[2] for p in pairs]
        ax.loglog(ns, rint, "s", ms=3.2, mfc="none", color="C1", label=r"$r_{\rm dyn}$ (integer)")
        ax.loglog(ns, rt, "o", ms=3.5, color="C0", label=r"$\tilde r_{\rm dyn}$")
        xa = np.array([np.log(n) for n in ns])
        ya = np.log(rt)
        al, be, se = fit_slope(xa, ya)
        xs = np.logspace(np.log10(min(ns)), np.log10(max(ns)), 20)
        ax.loglog(xs, np.exp(be) * xs ** al, "-", color="C0",
                  label=f"fit $\\alpha={al:.3f}\\pm{se:.3f}$")
        ax.loglog(xs, np.exp(be) * xs ** th, "--", color="k", lw=0.9,
                  label=rf"$n^{{{th:.2f}}}$")
        ax.set_xlabel(r"$n$")
        ax.set_title(lab)
    axes[1].set_ylabel(r"$r$")
    axes[1].legend(frameon=False, loc="upper left", handlelength=1.4)
    fig.savefig(os.path.join(FIG, "fig1_crossover_radius.pdf"))
    plt.close(fig)
    print("fig1 done")


# =====================================================================
# Fig 2: q_dyn at m=2 across the ten systems of Table VII
# =====================================================================
def fig2():
    t = load("table_qdyn.json")
    rows = t["rows"]
    fam = {}
    for r in rows:
        fam.setdefault(r["kind"], []).append(r)

    def mstd(rs):
        v = np.array([r["q_dyn"] for r in rs])
        return float(v.mean()), float(v.std(ddof=1))

    m2, s2 = mstd(fam["2D"])
    mc, sc = mstd(fam["Config"])
    m3, s3 = mstd(fam["3D"])

    labels, vals, errs, cols = [], [], [], []
    for r in rows:
        if r["kind"] == "Config":
            labels.append("cfg\n%d" % r["n"]); cols.append("#d62728")
        elif r["kind"] == "2D":
            L = {400: 20, 1600: 40, 6400: 80}[r["n"]]
            labels.append("2D\nL=%d" % L); cols.append("#1f77b4")
        else:
            L = {512: 8, 1000: 10, 1728: 12}[r["n"]]
            labels.append("3D\nL=%d" % L); cols.append("#2ca02c")
        vals.append(r["q_dyn"])
        errs.append(max(r["q_dyn"] - r["ci_lo"], r["ci_hi"] - r["q_dyn"]))

    fig, ax = plt.subplots(figsize=(3.35, 2.3), constrained_layout=True)
    ax.axhspan(m2 - s2, m2 + s2, color="#1f77b4", alpha=0.12, lw=0)
    ax.axhline(m2, color="#1f77b4", lw=1.0,
               label=f"2D family {m2:.4f}$\\pm${s2:.4f}")
    ax.axhline(mc, color="#d62728", lw=1.0, ls="--",
               label=f"config family {mc:.4f}$\\pm${sc:.4f}")
    ax.errorbar(range(len(vals)), vals, yerr=errs, fmt="o", ms=4.5,
                color="k", ecolor="gray", capsize=2, lw=1)
    for i, (v, c) in enumerate(zip(vals, cols)):
        ax.plot(i, v, "o", ms=4.5, color=c)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=6.3)
    ax.set_ylabel(r"$q_{\rm dyn}(m{=}2)$")
    ax.set_ylim(0.52, 0.66)
    ax.legend(frameon=False, loc="upper left", fontsize=6.3)
    fig.savefig(os.path.join(FIG, "fig2_qdyn_geometry.pdf"))
    plt.close(fig)
    print("fig2 done; 2D %.4f+-%.4f  3D %.4f+-%.4f  cfg %.4f+-%.4f"
          % (m2, s2, m3, s3, mc, sc))


# =====================================================================
# Fig 3: finite-window artifact  (main text includes fig3_mdimming.pdf)
# =====================================================================
def fig3():
    ac = load("artifact_config_n999.json")["m"]
    an = load("artifact_nscan.json")["sizes"]
    a2 = load("artifact_2d_L40.json")["m"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), constrained_layout=True)
    ax = axes[0]
    ms = sorted(int(m) for m in ac)
    ax.plot(ms, [ac[str(m)]["1_minus_q_sqrt_m"] for m in ms], "o-", color="C0",
            label=r"$n{=}999$ (config)")
    for nsize, col in (("501", "C2"), ("2001", "C3")):
        ms2 = sorted(int(m) for m in an[nsize]["m"])
        ax.plot(ms2, [an[nsize]["m"][str(m)]["1_minus_q_sqrt_m"] for m in ms2],
                "s--", color=col, ms=3.5, label=rf"$n{{=}}{nsize}$ (config)")
    ax.axhline(0.63, color="k", ls=":", lw=1.0)
    ax.annotate("earlier conjecture 0.63", (2.1, 0.645), fontsize=7)
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"$(1-\widetilde q_{\rm dyn})\sqrt{m}$")
    ax.set_title("(a) config model")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    ms3 = sorted(int(m) for m in a2)
    ax.plot(ms3, [a2[str(m)]["1_minus_q_sqrt_m"] for m in ms3], "o-", color="C1",
            label="2D hypergrid $L{=}40$")
    ax.axhline(0.63, color="k", ls=":", lw=1.0)
    ax.set_xlabel(r"$m$")
    ax.set_title("(b) 2D hypergrid, $m$ up to 20")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, "fig3_mdimming.pdf"))
    plt.close(fig)
    print("fig3 done -> fig3_mdimming.pdf")


# =====================================================================
# Fig 4: hypergeometric one-shot extinction vs mean field
# =====================================================================
def fig4():
    v = load("verification.json")
    rows = v["hypergeom_benchmark"]["rows"]
    q = v["hypergeom_benchmark"]["q"]
    n = v["hypergeom_benchmark"]["n"]
    ks = [r["k"] for r in rows if r["k"] <= 370]
    ex = [max(r["hypergeom_exact"], 1e-60) for r in rows if r["k"] <= 370]
    mf = [r["meanfield_q^k-1"] for r in rows if r["k"] <= 370]
    fig, ax = plt.subplots(figsize=(3.35, 2.5), constrained_layout=True)
    kk = np.linspace(2, 367, 300)
    ax.semilogy(kk, q ** (kk - 1), "-", color="gray", lw=1.0,
                label=r"mean field $q^{N_c-1}$")
    ax.semilogy(ks, ex, "o", ms=4, color="C0",
                label=r"exact $\binom{s_i}{N_c-1}/\binom{n-1}{N_c-1}$")
    def h(x):
        return -x * np.log(x) - (1 - x) * np.log(1 - x)
    al = kk / n
    rate = q * h(np.clip(al / q, 1e-12, 1)) - h(np.clip(al, 1e-12, 1))
    ax.semilogy(kk, np.exp(rate * n), "-", color="C3", lw=1.0,
                label=r"entropy law $e^{n[qh(\alpha/q)-h(\alpha)]}$")
    ax.axvline(367, color="k", ls=":", lw=0.9)
    ax.annotate(r"$N_c>s_i{+}1$: $P=0$" + "\n" + r"but $q^{N_c-1}>0$",
                (368, 3e-6), fontsize=7)
    ax.set_xlabel(r"class size $N_c$")
    ax.set_ylabel(r"$P($class dies at one firing$)$")
    ax.set_ylim(1e-52, 2)
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    fig.savefig(os.path.join(FIG, "fig4_hypergeometric.pdf"))
    plt.close(fig)
    print("fig4 done")


# =====================================================================
# Fig 5: quasi-consensus (time series, aliasing, dwell)
# =====================================================================
def fig5():
    qd = load("quasi_consensus.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), constrained_layout=True)
    M = np.array(qd["long_run"]["M_series_downsampled"])
    t = np.array(qd["long_run"]["event_times_downsampled"])
    sel = (t > 20) & (t < 80)
    ax = axes[0]
    ax.plot(t[sel], M[sel], lw=0.5, color="C0")
    ax.axhspan(0.90, 0.95, color="C1", alpha=0.18, lw=0)
    ax.set_xlabel("time")
    ax.set_ylabel(r"$M(t)$")
    ax.set_title("(a) majority fraction ($L{=}20$, $r{=}8$)")
    ax.set_ylim(0.86, 1.01)
    al = qd["aliasing"]
    dts = [float(k) for k in al]
    nus = [al[k]["nu"] for k in al]
    cvs = [al[k]["CV"] for k in al]
    ax = axes[1]
    xs = np.logspace(np.log10(0.0025), np.log10(0.5), 50)
    ax.loglog(dts, nus, "o-", color="C0", label=r"measured $\nu(\Delta t)$")
    ax.loglog(xs, 1 / (2 * xs), "--", color="k", lw=0.9, label=r"$1/(2\Delta t)$")
    ax.set_xlabel(r"sampling interval $\Delta t$")
    ax.set_ylabel(r"$\nu$ (per time unit)")
    ax2 = ax.twinx()
    ax2.semilogx(dts, cvs, "s-", color="C3", ms=3.5, label=r"CV$(\Delta t)$")
    ax2.axhline(1 / np.sqrt(2), color="C3", ls=":", lw=0.9)
    ax2.set_ylabel(r"CV", color="C3")
    ax2.tick_params(axis="y", colors="C3")
    ax2.set_ylim(0.1, 1.05)
    ax.set_title("(b) temporal aliasing")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.5, loc="lower left")
    Kh = {int(k): c for k, c in qd["long_run"]["K_histogram"].items()}
    tot = sum(Kh.values())
    ax = axes[2]
    ks = sorted(Kh)
    ax.bar([k + 0.0 for k in ks], [Kh[k] / tot for k in ks], width=0.35,
           color="C0", label=r"event-quantized $K$")
    p = qd["pooled"]
    ax.set_xlabel(r"dwell length")
    ax.set_ylabel("probability")
    ax.set_title("(c) dwell statistics")
    ax.text(0.55, 0.90, f"$\\nu={p['nu_per_time']:.1f}$/time\n"
                        f"$E[K]={p['E_K']:.3f}$\n"
                        f"CV$_K={p['CV_K_event']:.3f}$\n"
                        f"CV$_T={p['CV_T_continuous']:.3f}$",
            transform=ax.transAxes, fontsize=7, va="top")
    pg = p["nu_per_time"] / 400.0
    ax.bar([k + 0.38 for k in ks], [(1 - pg) * pg ** (k - 1) for k in ks],
           width=0.35, color="C2", alpha=0.8,
           label=r"geometric ref. $\nu/n$")
    ax.legend(frameon=False, fontsize=6.5)
    fig.savefig(os.path.join(FIG, "fig5_quasiconsensus.pdf"))
    plt.close(fig)
    print("fig5 done")


# =====================================================================
# Fig 6: partial conversion + clustered-class extinction
# =====================================================================
def fig6():
    pc = load("partial_conversion.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), constrained_layout=True)
    ax = axes[0]
    rows = [r for r in pc["Y_by_alpha"] if r["n_fires"] >= 30]
    xs = [r["alpha_bin"] + 0.01 for r in rows]
    ys = [r["Y_over_n"] for r in rows]
    ax.plot([0, 1], [0, pc["q_H"]], "--", color="k", lw=0.9,
            label=r"$\mathbb{E}[Y]/n=\alpha q$ (prop.)")
    ax.plot(xs, ys, "o", ms=3, color="C0", label="measured (m=3, r=8)")
    ax.set_xlabel(r"class-size fraction $\alpha=N_c/n$ at firing")
    ax.set_ylabel(r"$Y_i/n$ (converted mass)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(a) macroscopic partial conversion")

    ax = axes[1]
    ex = [r for r in pc["extinction_by_k"] if r["fires"] >= 20 and r["k"] <= 60]
    ks = [r["k"] for r in ex]
    fr = [r["freq"] for r in ex]
    ax.semilogy(ks, fr, "o", ms=3.5, color="C0", label="measured (clustered classes)")
    kk = np.arange(2, 61)
    from math import comb
    s = pc["s"]
    nn = pc["n"]
    pred = [comb(s, k - 1) / comb(nn - 1, k - 1) if k - 1 <= s else 0 for k in kk]
    ax.semilogy(kk, pred, "-", color="gray", lw=1.0,
                label=r"uniform placement $\binom{s}{k-1}/\binom{n-1}{k-1}$")
    ax.set_xlabel(r"class size $N_c$")
    ax.set_ylabel(r"$P($one-shot extinction$)$")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("(b) small classes: clustering effect")
    fig.savefig(os.path.join(FIG, "fig6_partial.pdf"))
    plt.close(fig)
    print("fig6 done")


# =====================================================================
# Fig 7: coverage-dynamics collapse with the exact band curve overlaid
# =====================================================================
def fig7_overlay():
    d = load("regen_e2c3_overlay.json")
    fam = d["families"]
    marks = {"2d": ("o", "#1f77b4"), "3d": ("s", "#2ca02c"),
             "config": ("^", "#d62728")}
    N_OF = {("2d", "20"): 400, ("2d", "40"): 1600, ("2d", "80"): 6400,
            ("3d", "8"): 512, ("3d", "10"): 1000, ("3d", "12"): 1728,
            ("3d", "14"): 2744, ("3d", "16"): 4096, ("3d", "18"): 5832,
            ("3d", "20"): 8000,
            ("config", "201"): 201, ("config", "501"): 501,
            ("config", "999"): 999, ("config", "2001"): 2001}
    fig, ax = plt.subplots(figsize=(3.35, 2.7), constrained_layout=True)
    for kind, sizes in fam.items():
        mk, col = marks[kind]
        for size, pts in sizes.items():
            n = N_OF[(kind, size)]
            q = np.array([(p["q_H"] * (n - 1) + 1) / n for p in pts])
            M = np.array([p["M_hat"] for p in pts])
            e = np.array([p["sem"] for p in pts])
            lab = (f"{kind} L={size}" if kind != "config"
                   else f"config n={size}")
            ax.errorbar(q, M, yerr=e, fmt=mk, ms=3, color=col, alpha=0.75,
                        lw=1, label=lab)
    qq = np.linspace(0.45, 1.0, 400)
    EM = (2 - qq) / (3 - 2 * qq)
    dom = qq >= 7 / 9
    ax.plot(qq[dom], EM[dom], "-", color="k", lw=1.4,
            label=r"exact band (Thm.~6), $(2\!-\!\beta/n)/(3\!-\!2\beta/n)$")
    ax.plot(qq[~dom], EM[~dom], ":", color="k", lw=1.0,
            label="extrapolation")
    ax.axvspan(7 / 9, 1.0, color="gray", alpha=0.10, lw=0)
    ax.text(0.895, 0.55, r"$\beta/n\geq 7/9$" + "\n" + "exact domain",
            fontsize=6.0, color="dimgray")
    ax.axhline(0.75, color="k", lw=0.6, ls="--")
    ax.set_xlabel(r"coverage $\beta/n$")
    ax.set_ylabel(r"$\widehat M$")
    ax.legend(ncol=2, loc="lower right", frameon=False)
    ax.set_xlim(0.45, 1.005)
    fig.savefig(os.path.join(FIG, "fig7_overlay.pdf"))
    plt.close(fig)
    print("fig7 done (exact curve overlaid)")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7_overlay()
    print("all figures in", FIG)
