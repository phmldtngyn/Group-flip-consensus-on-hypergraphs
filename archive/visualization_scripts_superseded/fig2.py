import json
import numpy as np
import matplotlib.pyplot as plt

# PRE Style Setup
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial'],
    'font.size': 10, 'axes.linewidth': 1.2, 'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'axes.labelsize': 12, 'legend.fontsize': 9, 'figure.dpi': 300, 'savefig.dpi': 300
})

with open('exact_1d_v2.json', 'r') as f:
    data = json.load(f)

q_band = np.linspace(0.5, 0.7, 100)
M_band = (2 - q_band) / (3 - 2 * q_band)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3)) # Double column width ~ 17cm

# Panel A
ax1.plot(q_band, M_band, 'k-', lw=1.5, label=r'Band $f(q) = \frac{2-q}{3-2q}$')
ax1.axhline(0.75, color='gray', ls='--', lw=1, label=r'$M=3/4$')
ax1.axhline(0.80, color='gray', ls=':', lw=1, label=r'$M=4/5$')

qs = [d['q'] for d in data]
Ms = [d['M'] for d in data]
ax1.scatter(qs, Ms, color='red', zorder=5, s=40, edgecolors='black', linewidths=0.5, label='Exact chain (Tab. I)')

ax1.set_xlabel(r'Coverage $q = \beta/n$')
ax1.set_ylabel(r'Stationary majority $E[M]$')
ax1.set_xlim(0.5, 0.7)
ax1.set_ylim(0.72, 0.82)

# thêm vào Panel A, trước lệnh legend:
ax1.axvspan(2/3, 0.70, color='gray', alpha=0.15)
ax1.axvline(2/3, color='gray', ls='-.', lw=1)
ax1.text(0.665, 0.725, r'exact regime', rotation=90, va='bottom',
         ha='right', fontsize=8, color='gray')
ax1.legend(loc='lower right', frameon=False)
ax1.text(0.52, 0.80, '(a)', fontweight='bold')

# Panel B
diffs = [d['M'] - d['M_band'] for d in data]
ax2.scatter(qs, diffs, color='red', zorder=5, s=40, edgecolors='black', linewidths=0.5)
ax2.axhline(0, color='k', ls='-', lw=1.2)
ax2.set_xlabel(r'Coverage $q$')
ax2.set_ylabel(r'$M_{\mathrm{exact}} - M_{\mathrm{band}}$')
ax2.set_xlim(0.5, 0.7)
ax2.text(0.52, -0.002, '(b)', fontweight='bold')

plt.tight_layout()
plt.savefig('fig2_exact_band.png', bbox_inches='tight')
# plt.savefig('fig2_exact_band.pdf', bbox_inches='tight') # Uncomment for vector graphics
plt.show()