/* Independent exact stationary solver for two-colour group-flip dynamics
   on the 1D hypergrid (cycle C_n, hyperedges = 3 consecutive nodes).
   B_i(r) = sites within circular lattice distance 2r of i.
   States: configurations reachable from the monochrome state, lumped
   under colour exchange only (no dihedral reduction).
   Every node fires at rate 1, so the total jump rate is n in every state
   and the CTMC stationary law equals that of the jump chain
   P = (1/n) sum_i T_i. We use the lazy chain (I+P)/2 and power iteration.
   Usage: region2_exact n r                                              */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
typedef uint64_t u64;
static int n; static u64 mask; static u64 B[64];
static u64 canon(u64 s){ int k=__builtin_popcountll(s); u64 c=(~s)&mask;
  if(2*k<n) return s; if(2*k>n) return c; return s<c?s:c; }
/* open-addressing hash set */
static u64 *key; static int32_t *val; static u64 cap;
static u64 h(u64 x){ x^=x>>33; x*=0xff51afd7ed558ccdULL; x^=x>>33; x*=0xc4ceb9fe1a85ec53ULL; x^=x>>33; return x; }
static int32_t find(u64 s){ u64 p=h(s)&(cap-1); while(key[p]!=~0ULL){ if(key[p]==s) return val[p]; p=(p+1)&(cap-1);} return -1; }
static void ins(u64 s,int32_t v){ u64 p=h(s)&(cap-1); while(key[p]!=~0ULL) p=(p+1)&(cap-1); key[p]=s; val[p]=v; }
int main(int argc,char**argv){
  n=atoi(argv[1]); int r=atoi(argv[2]); mask=(n==64)?~0ULL:((1ULL<<n)-1);
  int beta=0;
  for(int i=0;i<n;i++){ B[i]=0; for(int d=-2*r; d<=2*r; d++) B[i]|=1ULL<<(((i+d)%n+n)%n); }
  beta=__builtin_popcountll(B[0]);
  cap=1ULL<<27; key=malloc(cap*8); val=malloc(cap*4); memset(key,0xff,cap*8);
  u64 maxst=60000000; u64 *st=malloc(maxst*8); u64 ns=0, head=0;
  st[ns]=0; ins(0,0); ns++;
  while(head<ns){ u64 s=st[head++];
    for(int i=0;i<n;i++){ u64 ci=(s>>i)&1ULL; u64 cls = ci ? (s&B[i]) : ((~s)&B[i]&mask);
      u64 t=canon(s^cls); if(find(t)<0){ if(ns>=maxst){fprintf(stderr,"too many\n");return 1;} ins(t,(int32_t)ns); st[ns++]=t; } } }
  int32_t *T=malloc((size_t)ns*n*4);
  for(u64 a=0;a<ns;a++){ u64 s=st[a]; for(int i=0;i<n;i++){ u64 ci=(s>>i)&1ULL; u64 cls= ci?(s&B[i]):((~s)&B[i]&mask); T[a*n+i]=find(canon(s^cls)); } }
  free(key); free(val);
  double *p=malloc(ns*8),*q=malloc(ns*8); for(u64 a=0;a<ns;a++) p[a]=1.0/ns;
  double diff=1; long it=0;
  long extra=-1;
  while(it<200000 && extra!=0){ if(extra>0) extra--; else if(diff<1e-12) extra=3000;
    for(u64 a=0;a<ns;a++) q[a]=0.5*p[a];
    for(u64 a=0;a<ns;a++){ double w=0.5*p[a]/n; int32_t *t=T+a*n; for(int i=0;i<n;i++) q[t[i]]+=w; }
    diff=0; for(u64 a=0;a<ns;a++) diff+=fabs(q[a]-p[a]);
    { double z=0; for(u64 a=0;a<ns;a++) z+=q[a]; for(u64 a=0;a<ns;a++) q[a]/=z; }
    double *tmp=p; p=q; q=tmp; it++; if(it%200==0) fprintf(stderr,"it=%ld diff=%.2e\n",it,diff); }
  /* residual of the unlazy chain: || p P - p ||_1 */
  for(u64 a=0;a<ns;a++) q[a]=0;
  for(u64 a=0;a<ns;a++){ double w=p[a]/n; int32_t *t=T+a*n; for(int i=0;i<n;i++) q[t[i]]+=w; }
  double res=0,EM=0,tot=0; for(u64 a=0;a<ns;a++){ res+=fabs(q[a]-p[a]); int k=__builtin_popcountll(st[a]); int m=k>n-k?k:n-k; EM+=p[a]*(double)m/n; tot+=p[a]; }
  printf("%d %d %d %llu %.12f %.2e %ld %.15f\n",n,r,beta,(unsigned long long)ns,EM/tot,res,it,tot);
  return 0; }
