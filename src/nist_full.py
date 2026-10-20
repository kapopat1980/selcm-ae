"""Full NIST SP 800-22 rev1a battery (15 tests). Tests 1-8 from nist.py; tests 9-15 implemented here.
Parameters follow SP 800-22 rev1a defaults for n = 10^6 bits."""
import numpy as np
from numba import njit
from scipy.special import erfc, gammaincc
import nist

# ---------------------------------------------------------------- binary matrix rank (32x32)
@njit(cache=True)
def _rank32(rows):  # rows: uint32[32]
    r = rows.copy(); rank = 0
    for col in range(31, -1, -1):
        bit = np.uint32(1) << np.uint32(col); piv = -1
        for i in range(rank, 32):
            if r[i] & bit:
                piv = i; break
        if piv < 0: continue
        t = r[rank]; r[rank] = r[piv]; r[piv] = t
        for i in range(32):
            if i != rank and (r[i] & bit): r[i] ^= r[rank]
        rank += 1
    return rank

@njit(cache=True)
def _rank_counts(e):
    N = e.size // 1024; f32 = 0; f31 = 0
    for k in range(N):
        rows = np.zeros(32, dtype=np.uint32)
        for i in range(32):
            v = np.uint32(0)
            for j in range(32):
                v = (v << np.uint32(1)) | np.uint32(e[k * 1024 + i * 32 + j])
            rows[i] = v
        rk = _rank32(rows)
        if rk == 32: f32 += 1
        elif rk == 31: f31 += 1
    return N, f32, f31

def rank(e):
    N, f32, f31 = _rank_counts(e); f30 = N - f32 - f31
    chi = (f32 - 0.2888 * N) ** 2 / (0.2888 * N) + (f31 - 0.5776 * N) ** 2 / (0.5776 * N) + (f30 - 0.1336 * N) ** 2 / (0.1336 * N)
    return float(np.exp(-chi / 2))

# ---------------------------------------------------------------- non-overlapping template (m = 9, all 148 aperiodic templates)
def aperiodic_templates(m=9):
    out = []
    for v in range(1 << m):
        b = [(v >> (m - 1 - i)) & 1 for i in range(m)]
        if all(b[k:] != b[:m - k] for k in range(1, m)): out.append(v)
    return np.array(out, dtype=np.int64)
TEMPL = aperiodic_templates()
assert TEMPL.size == 148

@njit(cache=True)
def _nonoverlap_counts(e, templ, m, N):
    M = e.size // N; W = np.zeros((templ.size, N), dtype=np.int64)
    for blk in range(N):
        base = blk * M
        # rolling window values
        for t in range(templ.size):
            tv = templ[t]; i = 0
            while i <= M - m:
                v = 0
                for k in range(m): v = (v << 1) | e[base + i + k]
                if v == tv:
                    W[t, blk] += 1; i += m
                else:
                    i += 1
    return W

def non_overlapping(e, m=9, N=8):
    M = e.size // N; W = _nonoverlap_counts(e.astype(np.int64), TEMPL, m, N)
    mu = (M - m + 1) / 2 ** m; var = M * (1 / 2 ** m - (2 * m - 1) / 2 ** (2 * m))
    chi = ((W - mu) ** 2).sum(1) / var
    return gammaincc(N / 2, chi / 2)            # one p-value per template (148)

# ---------------------------------------------------------------- overlapping template (m = 9, all ones)
@njit(cache=True)
def _overlap_counts(e, m, M):
    N = e.size // M; nu = np.zeros(6, dtype=np.int64)
    for blk in range(N):
        c = 0; run = 0
        for i in range(M):
            if e[blk * M + i] == 1:
                run += 1
                if run >= m: c += 1
            else:
                run = 0
        nu[min(c, 5)] += 1
    return N, nu

def overlapping(e, m=9, M=1032):
    N, nu = _overlap_counts(e.astype(np.int64), m, M)
    pi = np.array([0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865])
    chi = np.sum((nu - N * pi) ** 2 / (N * pi))
    return float(gammaincc(5 / 2, chi / 2))

# ---------------------------------------------------------------- Maurer universal (L = 7, Q = 1280)
@njit(cache=True)
def _maurer_sum(e, L, Q):
    nb = e.size // L; K = nb - Q; T = np.zeros(1 << L, dtype=np.int64); s = 0.0
    for i in range(nb):
        v = 0
        for k in range(L): v = (v << 1) | e[i * L + k]
        if i < Q:
            T[v] = i + 1
        else:
            s += np.log2(i + 1 - T[v]); T[v] = i + 1
    return s, K

def maurer(e, L=7, Q=1280):
    s, K = _maurer_sum(e.astype(np.int64), L, Q); fn = s / K
    c = 0.7 - 0.8 / L + (4 + 32 / L) * K ** (-3 / L) / 15; sigma = c * np.sqrt(3.125 / K)
    return float(erfc(abs(fn - 6.1962507) / (np.sqrt(2) * sigma)))

# ---------------------------------------------------------------- linear complexity (M = 500)
@njit(cache=True)
def _bm(s):
    n = s.size; b = np.zeros(n, dtype=np.int64); c = np.zeros(n, dtype=np.int64)
    b[0] = 1; c[0] = 1; L = 0; m = -1
    for N in range(n):
        d = s[N]
        for i in range(1, L + 1): d ^= c[i] & s[N - i]
        if d == 1:
            t = c.copy()
            for i in range(0, n - N + m):
                c[N - m + i] ^= b[i]
            if L <= N // 2:
                L = N + 1 - L; m = N; b = t
    return L

@njit(cache=True)
def _lc_all(e, M):
    N = e.size // M; out = np.empty(N, dtype=np.int64)
    for k in range(N): out[k] = _bm(e[k * M:(k + 1) * M])
    return out

def linear_complexity(e, M=500):
    Ls = _lc_all(e.astype(np.int64), M); N = Ls.size
    mu = M / 2 + (9 + (-1) ** (M + 1)) / 36 - (M / 3 + 2 / 9) / 2 ** M
    T = (-1) ** M * (Ls - mu) + 2 / 9
    edges = [-np.inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, np.inf]
    nu = np.histogram(T, bins=edges)[0]
    pi = np.array([0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833])
    chi = np.sum((nu - N * pi) ** 2 / (N * pi))
    return float(gammaincc(3, chi / 2))

# ---------------------------------------------------------------- random excursions (+ variant)
def _cycles(e):
    S = np.concatenate(([0], np.cumsum(2 * e.astype(np.int64) - 1), [0]))
    zeros = np.where(S == 0)[0]
    return S, zeros, zeros.size - 1

def random_excursions(e):
    S, zeros, J = _cycles(e)
    if J < 500: return None
    ps = []
    for x in [-4, -3, -2, -1, 1, 2, 3, 4]:
        ax = abs(x); pi = [1 - 1 / (2 * ax)] + [1 / (4 * x * x) * (1 - 1 / (2 * ax)) ** (k - 1) for k in range(1, 5)] + [1 / (2 * ax) * (1 - 1 / (2 * ax)) ** 4]
        hits = (S == x).astype(np.int64); cs = np.cumsum(hits)
        visits = cs[zeros[1:]] - cs[zeros[:-1]]
        nu = np.bincount(np.minimum(visits, 5), minlength=6)
        chi = np.sum((nu - J * np.array(pi)) ** 2 / (J * np.array(pi)))
        ps.append(gammaincc(5 / 2, chi / 2))
    return np.array(ps)

def random_excursions_variant(e):
    S, zeros, J = _cycles(e)
    if J < 500: return None
    ps = []
    for x in list(range(-9, 0)) + list(range(1, 10)):
        xi = np.count_nonzero(S == x)
        ps.append(erfc(abs(xi - J) / np.sqrt(2 * J * (4 * abs(x) - 2))))
    return np.array(ps)

# ---------------------------------------------------------------- battery
SINGLE = dict(nist.TESTS); SINGLE.update({"Binary matrix rank": rank, "Overlapping template": overlapping,
                                          "Maurer universal": maurer, "Linear complexity": linear_complexity})
MULTI = {"Non-overlapping template (148)": non_overlapping, "Random excursions (8)": random_excursions,
         "Random excursions variant (18)": random_excursions_variant}

def battery(seqs):
    """Returns {test: dict(prop, unif_p, n)}. Multi-p tests report the worst sub-test (lowest proportion)."""
    from scipy.stats import chisquare
    def summarize(ps):
        ps = np.asarray(ps); h = np.histogram(ps, bins=10, range=(0, 1))[0]
        return float(np.mean(ps >= 0.01)), float(chisquare(h).pvalue) if ps.size >= 10 else float('nan')
    out = {}
    for name, f in SINGLE.items():
        p, u = summarize([f(s) for s in seqs]); out[name] = {"prop": p, "unif_p": u, "n": len(seqs), "subtests": 1}
    for name, f in MULTI.items():
        rows = [f(s) for s in seqs]; rows = [r for r in rows if r is not None]
        P = np.array(rows)  # n_seq x n_sub
        props, unifs = zip(*[summarize(P[:, j]) for j in range(P.shape[1])])
        j = int(np.argmin(props))
        out[name] = {"prop": props[j], "unif_p": unifs[j], "n": P.shape[0], "subtests": P.shape[1],
                     "subtests_passing": int(np.sum(np.array(props) >= 0.99 - 3 * np.sqrt(0.99 * 0.01 / P.shape[0])))}
    return out
