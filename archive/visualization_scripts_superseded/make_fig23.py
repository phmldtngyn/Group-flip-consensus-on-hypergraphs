import pandas as pd, numpy as np, matplotlib.pyplot as plt
df = pd.read_csv("results_v2/measurements_v2.csv")
agg = df.groupby(["family","q"], as_index=False).agg(
    M=("M","mean"), se=("se","mean"), ctilde=("ctilde","mean"))
q = np.linspace(0.5,0.85,120); band = (2-q)/(3-2*q)

fig, ax = plt.subplots(1,2, figsize=(11,4.4))
for f in ["1D","2D","config","WS"]:
    d = agg[agg.family==f]
    ax[0].errorbar(d.q, d.M, yerr=d.se, fmt="o-", ms=4, lw=1, label=f)
    ax[1].plot(d.q, d.ctilde, "o-", ms=4, lw=1, label=f)
ax[0].plot(q, band, "k--", lw=2, label="band $(2-q)/(3-2q)$")
ax[0].axvline(2/3, color="gray", ls=":", lw=1)
ax[0].set_xlabel("q"); ax[0].set_ylabel("M"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("q"); ax[1].set_ylabel(r"$\tilde c$"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig2_measurements.png", dpi=200)

# Fig 3: residual by family, reduced vs full
import statsmodels.formula.api as smf
a = df.groupby(["family","n","q","ctilde"], as_index=False).agg(M=("M","mean"), se=("se","mean"))
a["q2"]=a.q**2; a["q3"]=a.q**3; w=1/a.se**2
red  = smf.wls("M ~ q + q2 + q3 + C(family)", a, weights=w).fit()
full = smf.wls("M ~ q + q2 + q3 + ctilde + C(family)", a, weights=w).fit()
fig, ax = plt.subplots(1,2, figsize=(10,4.2), sharey=True)
for k,mod in [("reduced",red),("full (+ctilde)",full)]:
    for f in a.family.unique():
        m = a.family==f
        ax[0 if k=="reduced" else 1].scatter(mod.resid[m], a.q[m], s=18, label=f)
    ax[0 if k=="reduced" else 1].set_title(k); ax[0 if k=="reduced" else 1].axvline(0, color="k", lw=0.8)
ax[0].set_xlabel("residual"); ax[0].set_ylabel("q"); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig3_residuals.png", dpi=200)
print("done")