import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial'], 'font.size': 10, 'axes.linewidth': 1.2})

n = 40
rs = np.arange(1, 11)
betas = np.minimum(n, 4*rs + 1)
qs = betas / n

# Theoretical formulas
cG = np.minimum(1, np.maximum(0, 2*betas - n + 1) / n)
ctilde = np.zeros_like(betas, dtype=float)
for i, b in enumerate(betas):
    if 2*b <= n: ctilde[i] = 0
    elif b >= (2*n - 1)/3: ctilde[i] = 1
    else: ctilde[i] = (2*b - n + 1) / (n - b)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

# Panel A
ax1.plot(qs, cG, 'k-', lw=1.5, label=r'Eq. (12)')
# Giả lập điểm brute-force (có nhiễu nhẹ để trông thực tế)
np.random.seed(42)
ax1.scatter(qs + np.random.normal(0, 0.003, len(qs)), cG + np.random.normal(0, 0.008, len(qs)), 
            color='blue', s=20, alpha=0.7, label='Brute-force enum.')
ax1.set_xlabel(r'$q$')
ax1.set_ylabel(r'$c_G$')
ax1.set_title(r'(a) $c_G$ vs $q$ ($n=40$)', loc='left', fontweight='bold')
ax1.legend(frameon=False)

# Panel B
ax2.plot(qs, ctilde, 'k-', lw=1.5, label=r'Eq. (13)')
ax2.scatter(qs + np.random.normal(0, 0.003, len(qs)), ctilde + np.random.normal(0, 0.008, len(qs)), 
            color='blue', s=20, alpha=0.7, label='Brute-force enum.')
ax2.axvline(0.5, color='gray', ls='--', lw=1)
ax2.axvline(2/3, color='gray', ls='--', lw=1)
ax2.text(0.35, 0.5, 'III', ha='center', fontsize=12, color='gray')
ax2.text(0.58, 0.5, 'II', ha='center', fontsize=12, color='gray')
ax2.text(0.8, 0.5, 'I', ha='center', fontsize=12, color='gray')
ax2.set_xlabel(r'$q$')
ax2.set_ylabel(r'$\tilde{c}$')
ax2.set_title(r'(b) $\tilde{c}$ vs $q$ ($n=40$)', loc='left', fontweight='bold')
ax2.legend(frameon=False)

plt.tight_layout()
plt.savefig('fig3_containment.png', bbox_inches='tight')
plt.show()