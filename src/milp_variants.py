import numpy as np, time, json
from scipy.optimize import milp, LinearConstraint, Bounds
import selcm2 as S2
def bound(R, npass=2, pre=False, wrap=True):
    (f, fk), (b, bk) = S2.schedules(R, wrap)
    passes = [f, b, f, b][:npass]
    nv = R + sum(len(o) for o in passes)
    A=[];lo=[];hi=[]
    def add(coef,l,h):
        v=np.zeros(nv)
        for i,c in coef: v[i]+=c
        A.append(v);lo.append(l);hi.append(h)
    q={r:r for r in range(R)}; off=R; cost=np.zeros(nv)
    for order in passes:
        d=None
        for k,r in enumerate(order):
            a=off; off+=1; qi=q[r]; cost[a]+=1
            if pre: cost[qi]+=1          # pre-S-box layer on the row value: active iff row diff nonzero
            if d is None: add([(a,1),(qi,-1)],0,0)
            else:
                add([(a,1),(qi,-1),(d,1)],0,np.inf); add([(a,1),(qi,1),(d,-1)],0,np.inf); add([(a,1),(qi,-1),(d,-1)],-np.inf,0)
            q[r]=a; d=a
    add([(r,1) for r in range(R)],1,np.inf)
    res=milp(cost,constraints=LinearConstraint(np.array(A),lo,hi),integrality=np.ones(nv),bounds=Bounds(0,1))
    return int(round(res.fun))
for name,kw in [("2 passes",{}),("2 passes + pre-S",{"pre":True}),("3 passes",{"npass":3}),("3 passes + pre-S",{"npass":3,"pre":True}),("4 passes",{"npass":4})]:
    print(name,[ (R,bound(R,**kw)) for R in [16,32,64,128]],flush=True)
