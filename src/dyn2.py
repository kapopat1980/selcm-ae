"""Dynamics of the portable 2D-SELCM (v2 polynomial map) and S-box metrics of the v2 keyed inversion S-box."""
from paths import DATA, RES, FIG
import json, hashlib, math, numpy as np
from numba import njit
import selcm2, dyn

@njit(cache=True)
def orbit(a, b, x, y, n, burn):
    xs = np.empty(n); ys = np.empty(n)
    for i in range(burn + n):
        x, y = selcm2.selcm_step(x, y, a, b)
        if i >= burn: xs[i - burn] = x; ys[i - burn] = y
    return xs, ys

@njit(cache=True)
def hp(u): return math.exp(u) * (2.0 - u - u * u)

@njit(cache=True)
def lyap(a, b, x, y, n, burn):
    for _ in range(burn): x, y = selcm2.selcm_step(x, y, a, b)
    q11, q12, q21, q22 = 1.0, 0.0, 0.0, 1.0; s1 = 0.0; s2 = 0.0
    for _ in range(n):
        xn, yn = selcm2.selcm_step(x, y, a, b)
        A = b * hp(x); Bc = a * math.pi * math.cos(math.pi * y); c = a * math.pi * math.cos(math.pi * xn)
        j11, j12, j21, j22 = A, Bc, c * A, c * Bc + b * hp(y)
        m11 = j11 * q11 + j12 * q21; m12 = j11 * q12 + j12 * q22; m21 = j21 * q11 + j22 * q21; m22 = j21 * q12 + j22 * q22
        r11 = math.hypot(m11, m21); q11 = m11 / r11; q21 = m21 / r11
        r12 = q11 * m12 + q21 * m22; v1 = m12 - r12 * q11; v2 = m22 - r12 * q21
        r22 = math.hypot(v1, v2); q12 = v1 / r22; q22 = v2 / r22
        s1 += math.log(r11); s2 += math.log(r22); x, y = xn, yn
    return s1 / n, s2 / n

if __name__ == "__main__":
    R = {}
    ICS = [(0.21, 0.63), (0.31, 0.72), (0.55, 0.11)]
    grid = {}
    for a in np.linspace(0.5, 8, 16):
        for b in np.linspace(10, 100, 19):
            v = [lyap(a, b, x0, y0, 20000, 500) for x0, y0 in ICS]
            grid[f"{a:.3f},{b:.2f}"] = [float(np.mean([t[0] for t in v])), float(np.mean([t[1] for t in v]))]
    R["le_grid"] = grid; R["le2_min"] = min(v[1] for v in grid.values()); R["le1_max"] = max(v[0] for v in grid.values())
    R["le_table"] = {}
    for a in [0.5, 2, 4, 6, 8]:
        R["le_table"][str(a)] = []
        for b in [10, 25, 50, 100]:
            v = [lyap(a, b, x0, y0, 20000, 500) for x0, y0 in ICS]
            R["le_table"][str(a)].append([round(float(np.mean([t[0] for t in v])), 2), round(float(np.mean([t[1] for t in v])), 2)])
    bs = np.linspace(1, 100, 400); as_ = np.linspace(0.05, 12, 400)
    bif_b = np.array([orbit(2.0, b, 0.31, 0.72, 120, 500)[0] for b in bs]); bif_a = np.array([orbit(a, 20.0, 0.31, 0.72, 120, 500)[0] for a in as_])
    le_b = np.array([lyap(2.0, b, 0.31, 0.72, 4000, 300) for b in bs]); le_a = np.array([lyap(a, 20.0, 0.31, 0.72, 4000, 300) for a in as_])
    np.savez_compressed(RES + "/bif2.npz", bs=bs, as_=as_, bif_b=bif_b, bif_a=bif_a, le_b=le_b, le_a=le_a)
    R["le_a_min_b20"] = float(le_a[:, 1].min()); R["b_lambda2_pos"] = float(bs[np.argmax(le_b[:, 1] > 0)])
    # periodicity check: no orbit returns within 1e-12 in 10^6 steps (sampled parameters)
    N = 3000
    series = {"2D-SELCM v2 (a=2, b=20)": orbit(2.0, 20.0, 0.31, 0.72, N, 1000)[0], "2D-SELCM v2 (a=8, b=10)": orbit(8.0, 10.0, 0.31, 0.72, N, 1000)[0],
              "Logistic (mu=4)": dyn.orbit_logistic(0.31, N, 1000), "Sine (r=1)": dyn.orbit_sine(0.31, N, 1000), "Henon (1.4, 0.3)": dyn.orbit_henon(0.1, 0.1, N, 1000)[0]}
    R["complexity"] = {k: {"SampEn": float(dyn.sample_entropy(v, 2, 0.2 * v.std())), "PermEn6": dyn.perm_entropy(v, 6), "K01": dyn.zero_one_K(v)} for k, v in series.items()}
    xa, _ = orbit(2.0, 20.0, 0.31, 0.72, 60, 0); xb, _ = orbit(2.0, 20.0, 0.31 + 1e-15, 0.72, 60, 0)
    la = dyn.orbit_logistic(0.31, 60, 0); lb = dyn.orbit_logistic(0.31 + 1e-15, 60, 0)
    R["sensitivity"] = {"selcm_diff": np.abs(xa - xb).tolist(), "logistic_diff": np.abs(la - lb).tolist(),
                        "selcm_first_gt_0.1": int(np.argmax(np.abs(xa - xb) > 0.1)) + 1, "logistic_first_gt_0.1": int(np.argmax(np.abs(la - lb) > 0.1)) + 1}
    def ks(key):
        a, b, eps, X, Y, _, _ = selcm2.key_schedule(key, bytes(16), 100000); return selcm2.lattice_keystream(X, Y, a, b, eps, 1 << 18)
    k0 = hashlib.sha256(b"k").digest(); diffs = []
    for bit in range(0, 256, 17):
        k1 = bytearray(k0); k1[bit // 8] ^= 1 << (bit % 8); diffs.append(np.unpackbits((ks(k0) ^ ks(bytes(k1))).view(np.uint8)).mean() * 100)
    R["keystream_bit_diff_pct"] = {"mean": float(np.mean(diffs)), "min": float(np.min(diffs)), "max": float(np.max(diffs)), "n_flips": len(diffs)}
    json.dump(R, open(RES + "/dyn2.json", "w"), indent=1); print("dyn2 done", R["le2_min"], R["le1_max"], flush=True)
    SB = {}
    for i in range(5):
        k = hashlib.sha256(b"sbox%d" % i).digest(); S = selcm2._streams(k, bytes(16), 1000)[2].astype(np.int64)
        SB[f"SELCM-AE v2 key {i+1}"] = dyn.sbox_report(S); print(SB[f"SELCM-AE v2 key {i+1}"], flush=True)
    R["sbox"] = SB; json.dump(R, open(RES + "/dyn2.json", "w"), indent=1); print("all done")
