"""Row-level MILP lower bound on the number of active S-box rows in SELCM-AE v2 diffusion.

Model (truncated differences, one bit per row):
  p_r   plaintext row r has a nonzero difference (attacker's choice)
  per step k of a pass on row r with previous-output difference d_{k-1}:
     a_k >= q_r - d_{k-1},  a_k >= d_{k-1} - q_r,  a_k <= q_r + d_{k-1}
  (both zero -> inactive; exactly one nonzero -> active, because S is a bijection and the Feistel mixing
   is a bijection of prev; both nonzero -> attacker may choose cancellation, a_k = 0)
  output difference of the step = a_k; the row value becomes a_k; d_k = a_k.
  An active row contains >= 1 active S-box, so #active S-boxes >= sum a_k.
"""
from paths import DATA, RES, FIG
import json, numpy as np, time
from scipy.optimize import milp, LinearConstraint, Bounds
import selcm2 as S2

def bound(R, wrap=True, both=True):
    (f, fk), (b, bk) = S2.schedules(R, wrap)
    passes = [f, b] if both else [f]
    nv = R + sum(len(o) for o in passes)
    rows, lo, hi = [], [], []
    def add(coef, l, h):
        v = np.zeros(nv); [v.__setitem__(i, c) for i, c in coef]; rows.append(v); lo.append(l); hi.append(h)
    q = {r: ("p", r) for r in range(R)}          # current row-diff var for each row
    idx = lambda v: v[1] if v[0] == "p" else R + v[1]
    off = 0; acts = []
    for order in passes:
        d = None
        for k, r in enumerate(order):
            a = ("a", off + k); acts.append(idx(a)); qi = idx(q[r])
            if d is None:      # d = 0 (identical IV)
                add([(idx(a), 1), (qi, -1)], 0, 0)
            else:
                di = idx(d)
                add([(idx(a), 1), (qi, -1), (di, 1)], 0, np.inf)
                add([(idx(a), 1), (qi, 1), (di, -1)], 0, np.inf)
                add([(idx(a), 1), (qi, -1), (di, -1)], -np.inf, 0)
            q[r] = a; d = a
        off += len(order)
    add([(r, 1) for r in range(R)], 1, np.inf)
    c = np.zeros(nv); c[acts] = 1
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=np.ones(nv), bounds=Bounds(0, 1))
    x = np.round(res.x).astype(int)
    return int(round(res.fun)), x[:R].tolist()

if __name__ == "__main__":
    out = {}
    for R in [2, 4, 8, 9, 10, 12, 16, 20, 32, 48, 64, 96, 128]:
        t = time.time()
        full, pat = bound(R); nw, _ = bound(R, wrap=False); fo, _ = bound(R, both=False)
        out[R] = {"full": full, "no_wrap": nw, "forward_only": fo, "pattern": pat}
        print(R, "full", full, "no-wrap", nw, "fwd-only", fo, "rows with dP:", [i for i, v in enumerate(pat) if v], round(time.time() - t, 1), flush=True)
    json.dump(out, open(RES + "/milp.json", "w"), indent=1)
