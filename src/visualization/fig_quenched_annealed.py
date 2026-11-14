#!/usr/bin/env python3
"""Main-text Fig. 2: quenched vs annealed (exact chains)."""
# Figure: quenched vs annealed stationary majority fraction (PRE single column).
import numpy as np, matplotlib, json, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'.')
from annealed_size_scan import annealed_EM
f=lambda q:(2-q)/(3-2*q)
# Region-II table of the manuscript (n, beta, E[M] quenched exact, E[M] annealed)
T=[(9,5,0.754386,0.733333),(15,9,0.773810,0.750000),(16,9,0.756831,0.732639),(17,9,0.741294,0.717647),
(21,13,0.781711,0.757576),(22,13,0.769553,0.744318),(23,13,0.757895,0.732441),(24,13,0.746914,0.721726),
(25,13,0.736721,0.712000),(27,17,0.785965,0.761905),(28,17,0.776560,0.751190),(29,17,0.767367,0.741379),
(30,17,0.758488,0.732353),(31,17,0.750000,0.724014),(32,17,0.741958,0.716283),(33,17,0.734396,0.709091),(33,21,0.788618,0.764706)]
# re-check annealed column with this code
mx=0
for n,b,qe,an in T: mx=max(mx,abs(annealed_EM(n,b)[0]-an))
print("max |annealed recomputed - table| =",mx)
n=400; qs=[]; ems=[]
for b in range(201,400):
    qs.append(b/n); ems.append(annealed_EM(n,b)[0])
qs=np.array(qs); ems=np.array(ems)
json.dump({"n":n,"q":qs.tolist(),"EM_ann":ems.tolist()},open('annealed_n400.json','w'))
plt.rcParams.update({'font.family':'serif','font.size':8,'axes.linewidth':0.6,'mathtext.fontset':'cm'})
fig,ax=plt.subplots(figsize=(3.4,2.5))
ax.axvspan(2/3,1,color='0.92',lw=0)
qq=np.linspace(0.5,1,400)
ax.plot(qq[qq>=2/3],f(qq[qq>=2/3]),'k-',lw=1.2,label=r'quenched, exact: $f(q)$ (Thm. 2)')
ax.plot(qq[qq<=2/3],f(qq[qq<=2/3]),'k:',lw=0.9,label=r'$f(q)$ outside its domain')
ax.plot(qs,ems,'-',color='C3',lw=1.0,label=r'annealed, exact chain, $n=400$')
q2=np.array([b/n_ for n_,b,_,_ in T])
ax.plot(q2,[t[2] for t in T],'o',mfc='white',mec='k',ms=3.6,mew=0.8,label=r'quenched ring, exact, $n\leq 33$')
ax.plot(q2,[t[3] for t in T],'s',mfc='white',mec='C3',ms=3.2,mew=0.8,label=r'annealed, same $(n,\beta)$')
ax.set_xlim(0.5,1.0); ax.set_ylim(0.70,1.0)
ax.set_xlabel(r'coverage $q=\beta/n$'); ax.set_ylabel(r'stationary majority fraction $E[M]$')
ax.text(0.83,0.975,'Region I',ha='center',fontsize=7,color='0.35')
ax.text(0.583,0.975,'Region II',ha='center',fontsize=7,color='0.35')
ax.legend(fontsize=6,frameon=False,loc='lower right',handlelength=1.8)
fig.tight_layout(pad=0.3)
fig.savefig('fig2_quenched_annealed.pdf'); fig.savefig('fig2_quenched_annealed.png',dpi=300)
for q in [0.70,0.75,0.80,0.90]:
    i=int(round(q*400))-201; print(q, round(ems[i],4), round(f(qs[i]),4))
