#!/usr/bin/env python3
"""Annealed version of group-flip dynamics: exact stationary chain on the count n_1.
At each fire of i the interaction set is {i} plus beta-1 nodes drawn uniformly
without replacement from V\\{i}. Reproduces SM Table (size scan) and checks the
Region-II annealed column of main-text Table II.
Usage: python3 annealed_size_scan.py
"""
# Annealed analogue: at each fire of i the interaction set is {i} plus beta-1
# nodes drawn uniformly without replacement from V\{i}. Chain on k = #colour-1.
import numpy as np
from scipy.stats import hypergeom
def annealed_EM(n,beta):
    P=np.zeros((n+1,n+1))
    for k in range(n+1):
        # fire at colour-1 node (prob k/n): moved = 1+h, h~HG(n-1, k-1, beta-1)
        if k>0:
            h=np.arange(0,min(k-1,beta-1)+1)
            p=hypergeom.pmf(h,n-1,k-1,beta-1)
            for hh,pp in zip(h,p): P[k,k-1-hh]+=k/n*pp
        if k<n:
            m=n-k; h=np.arange(0,min(m-1,beta-1)+1)
            p=hypergeom.pmf(h,n-1,m-1,beta-1)
            for hh,pp in zip(h,p): P[k,k+1+hh]+=m/n*pp
    A=P.T-np.eye(n+1); A[0,:]=1; b=np.zeros(n+1); b[0]=1
    pi=np.linalg.solve(A,b)
    res=np.abs(P.T@pi-pi).sum()
    k=np.arange(n+1); return (pi*np.maximum(k,n-k)/n).sum(),res

f=lambda q:(2-q)/(3-2*q)
if __name__=="__main__":
    T=[(9,5,0.733333),(25,13,0.712000),(33,21,0.764706)]
    for n,b,ref in T: print("check",n,b,round(annealed_EM(n,b)[0],6),"table",ref)
    for q,ns in [(0.7,[100,200,400,800,1600,3200]),(0.8,[100,200,400,800,1600]),(0.9,[100,200,400,800,1600])]:
        for n in ns:
            b=round(q*n); em,res=annealed_EM(n,b)
            print(f"q={q} n={n} E[M]_ann={em:.5f} f={f(b/n):.5f} gap={f(b/n)-em:.5f} residual={res:.1e}",flush=True)
