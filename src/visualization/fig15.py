# fig15.py  —  schematic FIG.1 and FIG.5 (PRE-style)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial'],
    'font.size': 11,
    'axes.linewidth': 1.2,
})

N = 15
x = np.arange(N)
y = np.zeros(N)
colors = ['black'] * 9 + ['white'] * 6   # 9 majority, 6 minority

# ================= FIG 1 : update schematic =================
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))

# (a) geometry
ax = axes[0]
ax.scatter(x, y, c=colors, s=160, edgecolors='black', linewidths=1.4, zorder=5)
ax.add_patch(patches.FancyBboxPatch((-0.5, -0.45), 9.0, 0.9,
             boxstyle="round,pad=0.12", facecolor='tab:blue', alpha=0.18,
             edgecolor='tab:blue', lw=2, zorder=1))
ax.add_patch(patches.FancyBboxPatch((8.5, -0.45), 6.0, 0.9,
             boxstyle="round,pad=0.12", facecolor='tab:red', alpha=0.15,
             edgecolor='tab:red', lw=2, zorder=1))
ax.text(4.0, -1.05, r'ball $B_i(r)$', ha='center', color='tab:blue', fontweight='bold')
ax.text(11.5, -1.05, r'residual $R_i$', ha='center', color='tab:red', fontweight='bold')
ax.set_xlim(-1.5, 15.5); ax.set_ylim(-1.7, 1.7)
ax.set_title('(a) Geometry', fontweight='bold', loc='left')
ax.axis('off')

# (b) fire event
ax = axes[1]
ax.scatter(x, y, c=colors, s=160, edgecolors='black', linewidths=1.4, zorder=5)
for k in range(5, 9):                      # nodes of sigma_i inside B_i flip
    ax.scatter([k], [0], c='tab:red', s=70, zorder=6)
ax.annotate('', xy=(4, 0.55), xytext=(4, 1.35),
            arrowprops=dict(arrowstyle='->', color='tab:red', lw=2.2))
ax.text(4, 1.55, r'fire at $i$', ha='center', color='tab:red', fontweight='bold')
ax.set_xlim(-1.5, 15.5); ax.set_ylim(-1.7, 2.2)
ax.set_title(r'(b) Collective flip inside $B_i$', fontweight='bold', loc='left')
ax.axis('off')

# (c) epoch outcomes
ax = axes[2]
ax.text(0.5, 0.72, r'Minority fire $\Rightarrow$ erasure', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', fc='mistyrose', ec='tab:red', lw=1.5))
ax.text(0.5, 0.28, r'Majority fire $\Rightarrow$ rotation', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', fc='aliceblue', ec='tab:blue', lw=1.5))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title('(c) Epoch outcomes (Thm. 1, Lemma 1)', fontweight='bold', loc='left')
ax.axis('off')

plt.tight_layout()
plt.savefig('fig1_schematic.png', bbox_inches='tight', dpi=300)
plt.savefig('fig1_schematic.pdf', bbox_inches='tight')
plt.close(fig)

# ================= FIG 5 : erasure vs fragmentation =================
fig, axes = plt.subplots(1, 3, figsize=(12, 2.6))

ax = axes[0]
ax.text(0.5, 0.5, r'Region I:  $\mathrm{diam}(R_i) \leq r$',
        fontsize=13, ha='center', va='center',
        bbox=dict(boxstyle='round', fc='lightgreen', ec='green', lw=2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.set_title('(a) One-fire erasure, $\\tilde{c}=1$', fontweight='bold', loc='left')

ax = axes[1]
ax.text(0.5, 0.5, r'first fire erases only $R_i \cap B_j$',
        fontsize=12, ha='center', va='center',
        bbox=dict(boxstyle='round', fc='lightyellow', ec='orange', lw=2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.set_title('(b) Fragmented survivor', fontweight='bold', loc='left')

ax = axes[2]
ax.text(0.5, 0.62, r'Region II:  $\mathrm{diam}(R_i) > r$',
        fontsize=13, ha='center', va='center')
ax.text(0.5, 0.30, r'$M_{\mathrm{exact}} < M_{\mathrm{band}}$',
        fontsize=13, ha='center', va='center',
        bbox=dict(boxstyle='round', fc='lightcoral', ec='red', lw=2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.set_title('(c) Multi-step survival', fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig('fig5_fragmentation.png', bbox_inches='tight', dpi=300)
plt.savefig('fig5_fragmentation.pdf', bbox_inches='tight')
plt.close(fig)

print('OK: fig1_schematic.* and fig5_fragmentation.* written')