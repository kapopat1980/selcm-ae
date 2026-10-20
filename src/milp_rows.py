"""Rigorous row-level MILP bounds for SELCM-AE v2 (3 passes F-B-F, each step: pre-S-box, mix, add, main S-box).

Step k on row r:  z = S(Q_r ^ K0),  t = z + Mix(prev) + K1,  v = S(t);  Q_r <- v;  prev <- v.
Mix is a bijection of prev (two-round Feistel butterfly); S is a bijection; addition is a bijection in each input.

Differential (truncated, 1 bit per row): pre active iff Q_r differs; main active iff t differs;
  t: >= |q - d|, <= q + d  (cancellation allowed when both differ).
Linear (truncated masks): bijections preserve zero-ness of masks in both directions; for t = z + m (+K) a
  nonzero-correlation approximation has masks on t, z, m all zero or all nonzero, so one bit x_k per step
  covers z, m, t, v and the input row value; a value with two consumers (next step's mix, next use of the
  row) is a fork: its mask is the XOR of the consumers' masks (cancellation allowed).
Every active row holds >= 1 active S-box, so the objective is a valid lower bound on active S-boxes.
"""
from paths import DATA, RES, FIG
import json, time, numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

LOG2L = 8

def schedules(R):
    W = LOG2L if R > LOG2L else 0
    f = list(range(R)) + list(range(W)); b = list(range(R - 1, -1, -1)) + list(range(R - 1, R - 1 - W, -1))
    return [f, b, f]

class Model:
    def __init__(self): self.n = 0; self.cons = []; self.cost = {}
    def var(self): self.n += 1; return self.n - 1
    def add(self, coef, lo, hi): self.cons.append((coef, lo, hi))
    def xorlike(self, z, x, y):
        self.add([(z, 1), (x, -1), (y, 1)], 0, np.inf); self.add([(z, 1), (x, 1), (y, -1)], 0, np.inf)
        self.add([(z, 1), (x, -1), (y, -1)], -np.inf, 0)
    def solve(self, tl=600):
        A = lil_matrix((len(self.cons), self.n)); lo = np.empty(len(self.cons)); hi = np.empty(len(self.cons))
        for i, (coef, l, h) in enumerate(self.cons):
            for v, c in coef: A[i, v] += c
            lo[i] = l; hi[i] = h
        c = np.zeros(self.n)
        for v, w in self.cost.items(): c[v] += w
        r = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.ones(self.n), bounds=Bounds(0, 1),
                 options={"time_limit": tl})
        return r

def differential(R, npass=3, pre=True):
    M = Model(); passes = schedules(R)[:npass]
    q = [M.var() for _ in range(R)]; p = list(q)
    for order in passes:
        d = None
        for r in order:
            a = M.var(); M.cost[a] = M.cost.get(a, 0) + 1
            if pre: M.cost[q[r]] = M.cost.get(q[r], 0) + 1
            if d is None: M.add([(a, 1), (q[r], -1)], 0, 0)
            else: M.xorlike(a, q[r], d)
            q[r] = a; d = a
    M.add([(v, 1) for v in p], 1, np.inf)
    res = M.solve(); return int(round(res.fun)), int(res.status)

def linear(R, npass=3, pre=True):
    M = Model(); passes = schedules(R)[:npass]
    steps = [(pi, k, r) for pi, o in enumerate(passes) for k, r in enumerate(o)]
    x = {s: M.var() for s in steps}
    for s in steps: M.cost[x[s]] = 2 if pre else 1
    pvar = [M.var() for _ in range(R)]; cvar = [M.var() for _ in range(R)]
    # for each produced value: list of consumer mask vars
    last_writer = {r: ("p", r) for r in range(R)}; consumers = {}
    for pi, o in enumerate(passes):
        for k, r in enumerate(o):
            s = (pi, k, r)
            consumers.setdefault(last_writer[r], []).append(x[s])      # row value consumed as Q
            if k > 0: consumers.setdefault(("v", (pi, k - 1, o[k - 1])), []).append(x[s])   # prev consumed by mix
            last_writer[r] = ("v", s)
    for r in range(R): consumers.setdefault(last_writer[r], []).append(cvar[r])
    for key, cons in consumers.items():
        val = pvar[key[1]] if key[0] == "p" else x[key[1]]
        if len(cons) == 1: M.add([(val, 1), (cons[0], -1)], 0, 0)
        elif len(cons) == 2: M.xorlike(val, cons[0], cons[1])
        else: raise ValueError(cons)
    M.add([(v, 1) for v in pvar + cvar], 1, np.inf)
    res = M.solve(); return int(round(res.fun)), int(res.status)

if __name__ == "__main__":
    out = {"differential": {}, "linear": {}, "differential_2pass_pre": {}, "linear_2pass_pre": {}}
    for R in [1, 2, 4, 8, 9, 12, 16, 24, 32, 64, 128, 256, 512, 1024, 2048]:
        t = time.time()
        out["differential"][R] = differential(R); out["linear"][R] = linear(R)
        out["differential_2pass_pre"][R] = differential(R, 2); out["linear_2pass_pre"][R] = linear(R, 2)
        print(R, "3-pass diff", out["differential"][R], "lin", out["linear"][R], "| 2-pass diff", out["differential_2pass_pre"][R],
              "lin", out["linear_2pass_pre"][R], round(time.time() - t, 1), flush=True)
        json.dump(out, open(RES + "/milp_rows.json", "w"), indent=1)
