"""Builds the extended Region-II table (main text and SM) from
region2_results.txt (output of region2_exact) and the annealed solver."""
from fractions import Fraction as F
from annealed_baseline import stationary
rows=[]
for line in open('region2_results.txt'):
    if line.startswith(' ') or not line.strip(): continue
    n,r,b,ns,em,res,it,tot=line.split(); n,r,b,ns=int(n),int(r),int(b),int(ns); em=float(em); res=float(res)
    q=F(b,n); f=(2-q)/(3-2*q); ct=F(2*b-n+1,n-b)
    ann,_=stationary(n,b)
    rows.append((n,r,b,ns,float(q),ct,em,float(f),em-float(f),ann,res))
rows.sort(key=lambda x:(x[0],x[1]))
for x in rows:
    n,r,b,ns,q,ct,em,f,d,ann,res=x
    print(f"{n:2d} & {r} & {b:2d} & {q:.4f} & {ct.numerator}/{ct.denominator} & {ns} & {em:.6f} & {f:.6f} & ${d:+.4f}$ & {ann:.6f}\\\\")
print("% count",len(rows),"max residual %.1e"%max(x[10] for x in rows))
print("% all below f:",all(x[8]<0 for x in rows),"; annealed < quenched:",all(x[9]<x[6] for x in rows))
print("%% diff range: %.4f .. %.4f"%(min(x[8] for x in rows),max(x[8] for x in rows)))
s=sorted(rows,key=lambda x:x[4]); import itertools
mono=all(a[8]<=b[8] for a,b in zip(s,s[1:]))
print("% diff monotone in q:",mono)
for a,b in zip(s,s[1:]):
    if a[8]>b[8]: print("%  violation", a[:3],round(a[4],4),round(a[8],5)," -> ",b[:3],round(b[4],4),round(b[8],5))
mc=all(a[5]<=b[5] for a,b in zip(s,s[1:])); print("% ctil monotone in q:",mc)
