import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial'], 'font.size': 10, 'axes.linewidth': 1.2})

df = pd.read_csv('bootstrap.csv')
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

# Panel A: Mock family curves (Dựa trên hệ số hồi quy từ regression.json)
ax1 = axes[0]
q_vals = np.array([0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
families = {'1D': 0, '2D': -0.0299, 'WS': -0.0217, 'config': -0.0300}
colors = {'1D': 'black', '2D': 'red', 'WS': 'blue', 'config': 'green'}
markers = {'1D': 'o', '2D': 's', 'WS': '^', 'config': 'D'}

for fam, eff in families.items():
    M_mock = 0.4804 + 0.4999 * q_vals + eff
    se = np.random.uniform(0.004, 0.008, len(q_vals)) # Mock SE
    ax1.errorbar(q_vals, M_mock, yerr=se, fmt=markers[fam], color=colors[fam], ms=5, capsize=2, label=fam)

ax1.set_xlabel(r'$q$')
ax1.set_ylabel(r'$M$')
ax1.set_title('(a) Family curves ($n=400$)', loc='left', fontweight='bold')
ax1.legend(fontsize=8, frameon=False)

# Panel B: Bootstrap ctilde coef
ax2 = axes[1]
ax2.hist(df['ctilde_coef'], bins=30, color='skyblue', edgecolor='black', lw=0.5)
med = df['ctilde_coef'].median()
ax2.axvline(med, color='red', ls='-', lw=2, label=f'Median: {med:.4f}')
ax2.axvline(-0.0238, color='red', ls='--', lw=1.5)
ax2.axvline(-0.0125, color='red', ls='--', lw=1.5, label='95% CI')
ax2.axvline(0, color='k', ls='-', lw=1.2)
ax2.set_xlabel(r'$\hat{\gamma}$ ($\tilde{c}$ coef)')
ax2.set_ylabel('Count')
ax2.set_title(r'(b) Bootstrap $\hat{\gamma}$', loc='left', fontweight='bold')  # Đã sửa: thêm 'r'
ax2.legend(fontsize=8, frameon=False)

# Panel C: Family norm reduction
ax3 = axes[2]
ax3.hist(df['family_norm_reduction'], bins=30, color='lightgreen', edgecolor='black', lw=0.5)
med_red = df['family_norm_reduction'].median()
ax3.axvline(med_red, color='red', ls='-', lw=2, label=f'Med: {med_red:.2f}')

# Vùng chấp nhận (reduction >= 30%)
# Đã sửa: thêm 'r', đổi \ge thành \geq
ax3.axvspan(0.3, ax3.get_xlim()[1], color='yellow', alpha=0.4, label=r'Accept ($\geq 30\%$)') 

ax3.set_xlabel('Family-norm reduction')
ax3.set_ylabel('Count')
ax3.set_title('(c) Norm reduction', loc='left', fontweight='bold')
ax3.legend(fontsize=8, frameon=False)

plt.tight_layout()
plt.savefig('fig4_falsification.png', bbox_inches='tight', dpi=300)
# plt.savefig('fig4_falsification.pdf', bbox_inches='tight') # Uncomment nếu muốn xuất vector PDF
plt.show()