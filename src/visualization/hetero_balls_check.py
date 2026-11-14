#!/usr/bin/env python3
"""Verification of the unequal-ball result (main text Prop. 2, SM Prop. S4.x).
For random ball systems satisfying R1 and R2 with unequal ball sizes, solve the
exact stationary law of the group-flip chain on states reachable from consensus,
compute the epoch law as the rotation flux, and compare E[M] with
g(q*) = (2+qbar-2q*)/(3-2q*), q* = E_epoch[S_A]/(n E_epoch[a]).
Usage: python3 hetero_balls_check.py
"""
# (helpers) of the unequal-ball formula E[M] = (2+qbar-2q*)/(3-2q*),
# with q* = E_epoch[S_A]/(n E_epoch[a]) computed DIRECTLY from the epoch law
# (rotation-flux weighted), not inferred from E[M].
import random, sys
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from collections import deque

def balls(adj,r):
    n=len(adj); D=[]
    for s in range(n):
        d=[-1]*n; d[s]=0; q=deque([s])
        while q:
            u=q.popleft()
            for v in adj[u]:
                if d[v]<0: d[v]=d[u]+1; q.append(v)
        D.append(d)
    return D,[[j for j in range(n) if D[i][j]<=r] for i in range(n)]

def R1R2(n,D,B,r):
    if any(2*len(b)<=n for b in B): return False
    for i in range(n):
        Bi=set(B[i]); R=[j for j in range(n) if j not in Bi]
        if any(D[a][b]>r for a in R for b in R): return False
    return True

def solve(n,B):
    masks=[sum(1<<j for j in b) for b in B]; full=(1<<n)-1
    def step(s,i):
        bi=(s>>i)&1
        mv=(s&masks[i]) if bi else ((~s)&masks[i]&full)
        return s^mv
    idx={0:0}; order=[0]; dq=deque([0]); E=[]
    while dq:
        s=dq.popleft()
        for i in range(n):
            t=step(s,i)
            if t not in idx: idx[t]=len(order); order.append(t); dq.append(t)
            E.append((idx[s],idx[t],i))
    N=len(order)
    rows=[e[0] for e in E]; cols=[e[1] for e in E]
    P=sp.csr_matrix((np.full(len(E),1.0/n),(rows,cols)),shape=(N,N))
    A=(P.T-sp.identity(N)).tolil(); A[0,:]=np.ones(N)
    b=np.zeros(N); b[0]=1
    pi=spla.spsolve(A.tocsc(),b)
    res=np.abs(P.T@pi-pi).sum()
    pop=np.array([bin(s).count('1') for s in order])
    EM=(pi*np.maximum(pop,n-pop)/n).sum()
    # epoch law: flux of rotations into each state
    beta=[len(b) for b in B]
    ep=np.zeros(N); rotfrac=0.0
    for (u,v,i) in E:
        s=order[u]; k=pop[u]
        if 2*k==n: continue
        maj1 = k>n-k
        if ((s>>i)&1)==maj1:          # i in majority -> rotation
            ep[v]+=pi[u]/n; rotfrac+=pi[u]/n
    ep/=ep.sum()
    Ea=0.0; ES=0.0
    for v in range(N):
        if ep[v]==0: continue
        s=order[v]; k=pop[v]; minbit=1 if k<n-k else 0
        A_=[j for j in range(n) if ((s>>j)&1)==minbit]
        Ea+=ep[v]*len(A_); ES+=ep[v]*sum(beta[j] for j in A_)
    return EM,res,Ea,ES,rotfrac,N

f=lambda q:(2-q)/(3-2*q)
seen=set(); rows=[]; worst=0
for seed in range(1,40):
    random.seed(seed)
    for trial in range(3000):
        n=random.choice([9,10,11,12,13,14])
        adj=[set() for _ in range(n)]
        for i in range(n): adj[i].add((i+1)%n); adj[(i+1)%n].add(i)
        for _ in range(random.randint(0,5)):
            a,b=random.sample(range(n),2); adj[a].add(b); adj[b].add(a)
        r=random.choice([2,3,4])
        key=(n,r,tuple(sorted(tuple(sorted(a)) for a in adj)))
        D,B=balls(adj,r); sz=[len(b) for b in B]
        if min(sz)==max(sz) or max(sz)==n: continue
        if not R1R2(n,D,B,r): continue
        Bkey=(n,tuple(sorted(tuple(b) for b in B)))
        if Bkey in seen: continue
        seen.add(Bkey)
        EM,res,Ea,ES,rf,N=solve(n,B)
        qbar=sum(sz)/n/n; qs=ES/(n*Ea); qmin=min(sz)/n; qmax=max(sz)/n
        g=lambda q:(2+qbar-2*q)/(3-2*q)
        rows.append((n,EM,g(qs),qs,qbar,g(qmax),g(qmin),f(qbar),abs(rf-EM),res))
        if len(rows)>=60: break
    if len(rows)>=60: break
import numpy as np
R=np.array(rows)
print("distinct ball systems:",len(R),"n range",int(R[:,0].min()),int(R[:,0].max()))
print("max |E[M]-g(q*)|:",np.abs(R[:,1]-R[:,2]).max())
print("max |rotation fraction - E[M]|:",R[:,8].max(),"max residual",R[:,9].max())
print("bounds hold:",bool(np.all((R[:,5]-1e-12<=R[:,1])&(R[:,1]<=R[:,6]+1e-12))))
print("q*<qbar in all:",bool(np.all(R[:,3]<R[:,4])),"; E[M]-f(qbar) range:",(R[:,1]-R[:,7]).min(),(R[:,1]-R[:,7]).max())
