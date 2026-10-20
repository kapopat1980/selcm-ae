"""Dynamics of 2D-SELCM (bifurcation, LE sweeps, phase portraits, entropies, 0-1 test, sensitivity) and S-box metrics."""
from paths import DATA, RES, FIG
import json, hashlib, math, numpy as np
from numba import njit
import selcm

# ---------------------------------------------------------------- map + references
@njit(cache=True)
def orbit_selcm(a, b, x, y, n, burn):
    xs = np.empty(n); ys = np.empty(n)
    for i in range(burn + n):
        x, y = selcm.selcm_step(x, y, a, b)
        if i >= burn: xs[i - burn] = x; ys[i - burn] = y
    return xs, ys

@njit(cache=True)
def hp(u): return math.exp(u) * (2.0 - u - u * u)

@njit(cache=True)
def lyap_selcm(a, b, x, y, n, burn):
    for _ in range(burn): x, y = selcm.selcm_step(x, y, a, b)
    q11, q12, q21, q22 = 1.0, 0.0, 0.0, 1.0; s1 = 0.0; s2 = 0.0
    for _ in range(n):
        xn, yn = selcm.selcm_step(x, y, a, b)
        A = b * hp(x); Bc = a * math.pi * math.cos(math.pi * y); c = a * math.pi * math.cos(math.pi * xn)
        j11, j12, j21, j22 = A, Bc, c * A, c * Bc + b * hp(y)
        m11 = j11 * q11 + j12 * q21; m12 = j11 * q12 + j12 * q22; m21 = j21 * q11 + j22 * q21; m22 = j21 * q12 + j22 * q22
        r11 = math.hypot(m11, m21); q11 = m11 / r11; q21 = m21 / r11
        r12 = q11 * m12 + q21 * m22; v1 = m12 - r12 * q11; v2 = m22 - r12 * q21
        r22 = math.hypot(v1, v2); q12 = v1 / r22; q22 = v2 / r22
        s1 += math.log(r11); s2 += math.log(r22); x, y = xn, yn
    return s1 / n, s2 / n

@njit(cache=True)
def orbit_logistic(x, n, burn):
    out = np.empty(n)
    for i in range(burn + n):
        x = 4.0 * x * (1.0 - x)
        if i >= burn: out[i - burn] = x
    return out

@njit(cache=True)
def orbit_sine(x, n, burn):
    out = np.empty(n)
    for i in range(burn + n):
        x = math.sin(math.pi * x)
        if i >= burn: out[i - burn] = x
    return out

@njit(cache=True)
def orbit_henon(x, y, n, burn):
    xs = np.empty(n); ys = np.empty(n)
    for i in range(burn + n):
        x, y = 1.0 - 1.4 * x * x + y, 0.3 * x
        if i >= burn: xs[i - burn] = x; ys[i - burn] = y
    return xs, ys

# ---------------------------------------------------------------- complexity measures
@njit(cache=True)
def sample_entropy(x, m, r):
    n = x.size; A = 0; Bn = 0
    for i in range(n - m):
        for j in range(i + 1, n - m):
            ok = True
            for k in range(m):
                if abs(x[i + k] - x[j + k]) > r: ok = False; break
            if ok:
                Bn += 1
                if abs(x[i + m] - x[j + m]) <= r: A += 1
    return -math.log(A / Bn) if A > 0 and Bn > 0 else np.inf

def perm_entropy(x, order=6):
    n = x.size - order + 1
    pats = np.argsort(np.lib.stride_tricks.sliding_window_view(x, order), axis=1)
    codes = (pats * (order ** np.arange(order))).sum(1)
    _, c = np.unique(codes, return_counts=True); p = c / n
    return float(-(p * np.log2(p)).sum() / np.log2(math.factorial(order)))

def zero_one_K(phi, ncs=100, seed=0):
    rng = np.random.default_rng(seed); N = phi.size; nmax = N // 10; Ks = []
    j = np.arange(1, N + 1); n = np.arange(1, nmax + 1)
    for c in rng.uniform(np.pi / 5, 4 * np.pi / 5, ncs):
        p = np.cumsum(phi * np.cos(j * c)); q = np.cumsum(phi * np.sin(j * c))
        M = np.array([np.mean((p[k:] - p[:-k]) ** 2 + (q[k:] - q[:-k]) ** 2) for k in n])
        D = M - phi.mean() ** 2 * (1 - np.cos(n * c)) / (1 - np.cos(c))
        Ks.append(np.corrcoef(n, D)[0, 1])
    return float(np.median(Ks))

# ---------------------------------------------------------------- S-box metrics (16-bit)
@njit(cache=True)
def fwht(f):
    a = f.astype(np.int64).copy(); h = 1; n = a.size
    while h < n:
        for i in range(0, n, 2 * h):
            for j in range(i, i + h):
                x = a[j]; y = a[j + h]; a[j] = x + y; a[j + h] = x - y
        h *= 2
    return a

@njit(cache=True)
def nonlinearity_component(S, mask):
    n = S.size; f = np.empty(n, dtype=np.int64)
    for x in range(n):
        v = S[x] & mask; par = 0
        while v:
            par ^= 1; v &= v - 1
        f[x] = 1 - 2 * par
    w = fwht(f); mx = 0
    for i in range(n):
        if abs(w[i]) > mx: mx = abs(w[i])
    return n // 2 - mx // 2

@njit(cache=True)
def diff_uniformity(S):
    n = S.size; best = 0; cnt = np.zeros(n, dtype=np.int32)
    for da in range(1, n):
        cnt[:] = 0
        for x in range(n):
            cnt[S[x ^ da] ^ S[x]] += 1
        m = cnt.max()
        if m > best: best = m
    return best

@njit(cache=True)
def sac_bic(S):
    n = S.size; sac = np.zeros((16, 16)); bic = 0.0
    for i in range(16):
        d = np.empty((n, 16), dtype=np.float64)
        for x in range(n):
            v = S[x] ^ S[x ^ (1 << i)]
            for j in range(16): d[x, j] = (v >> j) & 1
        for j in range(16): sac[i, j] = d[:, j].mean()
        for j in range(16):
            for k in range(j + 1, 16):
                a = d[:, j] - d[:, j].mean(); b = d[:, k] - d[:, k].mean()
                c = abs((a * b).mean() / math.sqrt((a * a).mean() * (b * b).mean()))
                if c > bic: bic = c
    return sac, bic

def gf16_inverse_sbox():
    poly = 0x1002B  # x^16 + x^5 + x^3 + x + 1 (primitive)
    exp = np.zeros(2 * 65535, dtype=np.int64); log = np.zeros(65536, dtype=np.int64); v = 1
    for i in range(65535):
        exp[i] = v; log[v] = i; v <<= 1
        if v & 0x10000: v ^= poly
    exp[65535:] = exp[:65535]
    S = np.zeros(65536, dtype=np.int64)
    for x in range(1, 65536): S[x] = exp[(65535 - log[x]) % 65535]
    return S

def sbox_report(S, nsamp=256, seed=1):
    rng = np.random.default_rng(seed)
    coord = [nonlinearity_component(S, 1 << i) for i in range(16)]
    comb = [nonlinearity_component(S, int(m)) for m in rng.integers(1, 65536, nsamp)]
    sac, bic = sac_bic(S)
    return {"NL_coord_min": int(min(coord)), "NL_coord_mean": float(np.mean(coord)),
            "NL_comb_min": int(min(comb)), "NL_comb_mean": float(np.mean(comb)),
            "DU": int(diff_uniformity(S)), "SAC_mean": float(sac.mean()), "SAC_min": float(sac.min()),
            "SAC_max": float(sac.max()), "BIC_max_corr": float(bic)}

if __name__ == "__main__":
    R = {}
    # bifurcation + LE sweeps
    bs = np.linspace(1, 100, 400); as_ = np.linspace(0.05, 12, 400)
    bif_b = [orbit_selcm(2.0, b, 0.31, 0.72, 120, 500)[0].tolist() for b in bs]
    bif_a = [orbit_selcm(a, 20.0, 0.31, 0.72, 120, 500)[0].tolist() for a in as_]
    le_b = [lyap_selcm(2.0, b, 0.31, 0.72, 4000, 300) for b in bs]
    le_a = [lyap_selcm(a, 20.0, 0.31, 0.72, 4000, 300) for a in as_]
    le_grid = {f"{a},{b}": lyap_selcm(a, b, 0.21, 0.63, 4000, 300) for a in np.linspace(0.5, 8, 16) for b in np.linspace(10, 100, 19)}
    R["sweep"] = {"b": bs.tolist(), "le_b": le_b, "a": as_.tolist(), "le_a": le_a,
                  "le2_min_region": min(v[1] for v in le_grid.values()), "le1_max_region": max(v[0] for v in le_grid.values()),
                  "grid_points": len(le_grid)}
    np.savez_compressed(RES + "/bif.npz", bs=bs, as_=as_, bif_b=np.array(bif_b), bif_a=np.array(bif_a),
                        le_b=np.array(le_b), le_a=np.array(le_a))
    # complexity vs reference maps
    N = 3000
    series = {"2D-SELCM (a=2, b=20)": orbit_selcm(2.0, 20.0, 0.31, 0.72, N, 1000)[0],
              "2D-SELCM (a=8, b=10)": orbit_selcm(8.0, 10.0, 0.31, 0.72, N, 1000)[0],
              "Logistic (mu=4)": orbit_logistic(0.31, N, 1000), "Sine (r=1)": orbit_sine(0.31, N, 1000),
              "Henon (1.4, 0.3)": orbit_henon(0.1, 0.1, N, 1000)[0]}
    R["complexity"] = {k: {"SampEn": float(sample_entropy(v, 2, 0.2 * v.std())), "PermEn6": perm_entropy(v, 6),
                           "K01": zero_one_K(v)} for k, v in series.items()}
    # sensitivity to 1e-15 perturbation of x0
    xa, _ = orbit_selcm(2.0, 20.0, 0.31, 0.72, 60, 0); xb, _ = orbit_selcm(2.0, 20.0, 0.31 + 1e-15, 0.72, 60, 0)
    la = orbit_logistic(0.31, 60, 0); lb = orbit_logistic(0.31 + 1e-15, 60, 0)
    R["sensitivity"] = {"selcm_diff": np.abs(xa - xb).tolist(), "logistic_diff": np.abs(la - lb).tolist(),
                        "selcm_first_gt_0.1": int(np.argmax(np.abs(xa - xb) > 0.1)) + 1,
                        "logistic_first_gt_0.1": int(np.argmax(np.abs(la - lb) > 0.1)) + 1}
    # keystream key sensitivity (1-bit key flip -> keystream bit difference)
    def ks(key):
        a, b, eps, X, Y, _, _ = selcm.key_schedule(key, bytes(16), 100000); return selcm.lattice_keystream(X, Y, a, b, eps, 1 << 18)
    k0 = hashlib.sha256(b"k").digest(); diffs = []
    for bit in range(0, 256, 17):
        k1 = bytearray(k0); k1[bit // 8] ^= 1 << (bit % 8)
        diffs.append(np.unpackbits((ks(k0) ^ ks(bytes(k1))).view(np.uint8)).mean() * 100)
    R["keystream_bit_diff_pct"] = {"mean": float(np.mean(diffs)), "min": float(np.min(diffs)), "max": float(np.max(diffs)), "n_flips": len(diffs)}
    json.dump(R, open(RES + "/dyn.json", "w"), indent=1); print("dyn done", flush=True)
    # S-boxes
    SB = {}
    for i in range(5):
        k = hashlib.sha256(b"sbox%d" % i).digest(); a, b, eps, X, Y, _, _ = selcm.key_schedule(k, bytes(16), 1000)
        S = np.argsort(selcm.lattice_keystream(X, Y, a, b, eps, 65536), kind="stable").astype(np.int64)
        SB[f"SELCM-AE key {i+1}"] = sbox_report(S); print(SB[f"SELCM-AE key {i+1}"], flush=True)
    for i in range(3):
        S = np.random.default_rng(100 + i).permutation(65536).astype(np.int64)
        SB[f"Uniform random permutation {i+1}"] = sbox_report(S); print(SB[f"Uniform random permutation {i+1}"], flush=True)
    SB["GF(2^16) inversion (optimal reference)"] = sbox_report(gf16_inverse_sbox()); print(SB["GF(2^16) inversion (optimal reference)"])
    R["sbox"] = SB
    json.dump(R, open(RES + "/dyn.json", "w"), indent=1); print("all done")
