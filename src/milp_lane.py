"""Lane-level MILP: minimum number of active S-boxes in the SELCM-AE diffusion.

Truncated model: a binary per 16-bit word says whether the two encryptions differ there. Every operation
used (modular addition with a second input, rotation, S-box, XOR with key) is a bijection in each input,
so for z = f(x, y):  z >= x - y, z >= y - x, z <= x + y   (equal inputs -> equal; exactly one differs ->
differs; both differ -> the characteristic may choose equality). An S-box is active iff its input differs.
"""
from paths import DATA, RES, FIG
import json, sys, time, numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

def schedules(R, W):
    f = list(range(R)) + list(range(W)); b = list(range(R - 1, -1, -1)) + list(range(R - 1, R - 1 - W, -1))
    return f, b

def solve(R, L, pre=True, npass=2, time_limit=600):
    logL = L.bit_length() - 1; W = logL if R > logL else 0
    f, b = schedules(R, W); passes = [f, b, f][:npass]
    nv = [0]; cons = []; cost = {}
    def var():
        nv[0] += 1; return nv[0] - 1
    def xorlike(x, y):
        z = var()
        if x is None and y is None: cons.append(([(z, 1)], 0, 0)); return z
        if y is None: x, y = y, x
        if x is None:
            cons.append(([(z, 1), (y, -1)], 0, 0)); return z
        cons.append(([(z, 1), (x, -1), (y, 1)], 0, np.inf)); cons.append(([(z, 1), (x, 1), (y, -1)], 0, np.inf))
        cons.append(([(z, 1), (x, -1), (y, -1)], -np.inf, 0)); return z
    Q = [[var() for _ in range(L)] for _ in range(R)]; P0 = [q for row in Q for q in row]
    for order in passes:
        prev = None
        for k, r in enumerate(order):
            s = 1 << (k % logL)
            if prev is None: m = [None] * L
            else:
                m = [None] * L
                for lo in range(L):
                    if lo & s: continue
                    hi = lo | s
                    uh = xorlike(prev[hi], prev[lo]); ul = xorlike(prev[lo], uh); m[hi] = uh; m[lo] = ul
            row = []
            for j in range(L):
                z = Q[r][j]
                if pre: cost[z] = cost.get(z, 0) + 1
                t = xorlike(z, m[j]); cost[t] = cost.get(t, 0) + 1; row.append(t)
            Q[r] = row; prev = row
    cons.append(([(v, 1) for v in P0], 1, np.inf))
    n = nv[0]; A = lil_matrix((len(cons), n)); lo = np.empty(len(cons)); hi = np.empty(len(cons))
    for i, (coef, l, h) in enumerate(cons):
        for v, c in coef: A[i, v] += c
        lo[i] = l; hi[i] = h
    c = np.zeros(n)
    for v, w in cost.items(): c[v] = w
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "disp": False})
    x = np.round(res.x).astype(int) if res.x is not None else None
    lb = getattr(res, "mip_dual_bound", None)
    return {"R": R, "L": L, "pre": pre, "npass": npass, "status": int(res.status), "obj": float(res.fun) if res.fun is not None else None,
            "dual_bound": float(lb) if lb is not None else None, "vars": n, "secs": round(time.time() - t0, 1),
            "dP_words": int(sum(x[v] for v in P0)) if x is not None else None}

if __name__ == "__main__":
    pre = sys.argv[1] == "pre"; out = []
    for L in [int(v) for v in sys.argv[2].split(",")]:
        for R in [int(v) for v in sys.argv[3].split(",")]:
            r = solve(R, L, pre=pre, time_limit=float(sys.argv[4]) if len(sys.argv) > 4 else 600); out.append(r); print(r, flush=True)
    json.dump(out, open(f"{RES}/milp_lane_{sys.argv[1]}.json", "w"), indent=1)
