"""Full evaluation of SELCM-AE v2 against v1 and baselines B1-B4 on corpus v2."""
from paths import DATA, RES, FIG
import json, hashlib, time, numpy as np
from scipy.stats import chisquare
from scipy.signal import resample_poly
from pystoi import stoi
from pesq import pesq
import selcm, selcm2, baselines as B

Z = np.load(DATA + "/dataset2.npz"); META = json.load(open(DATA + "/meta2.json"))
FILES = list(META.keys()); rng = np.random.default_rng(11)
KEY = hashlib.sha256(b"v2 key").digest(); NONCE = hashlib.md5(b"v2 nonce").digest()
M = {"SELCM-AE v2": (lambda x, k, n: selcm2.encrypt(x, k, n), lambda c, k, n, m: selcm2.decrypt(c, k, n, m)),
     "SELCM-AE v1": (lambda x, k, n: selcm.encrypt(x, k, n), lambda c, k, n, m: selcm.decrypt(c, k, n, m)),
     "B1 Logistic-XOR": (lambda x, k, n: B.b1_encrypt(x, k), lambda c, k, n, m: B.b1_decrypt(c, k, m)),
     "B2 Henon-perm + Lorenz-XOR": (lambda x, k, n: B.b2_encrypt(x, k), lambda c, k, n, m: B.b2_decrypt(c, k, m)),
     "B3 Forward chained diffusion": (lambda x, k, n: B.b3_encrypt(x, k), lambda c, k, n, m: B.b3_decrypt(c, k, m)),
     "B4 AES-256-CTR": (lambda x, k, n: B.b4_encrypt(x, k, n), lambda c, k, n, m: B.b4_decrypt(c, k, n, m))}
u16 = selcm2.to_u16
def corr(a, b):
    a = a.astype(float); b = b.astype(float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
def h16(c): p = np.bincount(c, minlength=65536) / c.size; p = p[p > 0]; return float(-(p * np.log2(p)).sum())
def h8(c): p = np.bincount(c.view(np.uint8), minlength=256) / (2 * c.size); p = p[p > 0]; return float(-(p * np.log2(p)).sum())
def nu(c1, c2):
    m = min(c1.size, c2.size); a = c1[:m].astype(np.int64); b = c2[:m].astype(np.int64)
    return float((a != b).mean() * 100), float(np.abs(a - b).mean() / 65535 * 100)
def tweak(x, p): y = x.copy(); y[p] = y[p] + 1 if y[p] < 32767 else y[p] - 1; return y
def segsnr(ref, t, sr, frame=0.02):
    Lf = int(sr * frame); vals = []
    for i in range(ref.size // Lf):
        r = ref[i * Lf:(i + 1) * Lf].astype(float); e = r - t[i * Lf:(i + 1) * Lf].astype(float)
        if np.sum(r ** 2) < 1e-3: continue
        vals.append(np.clip(10 * np.log10(np.sum(r ** 2) / max(np.sum(e ** 2), 1e-12)), -10, 35))
    return float(np.mean(vals))
def edge_positions(x, key, nonce, k):
    Np, R_, S, Si, perm, K, a1, a2 = selcm2._streams(key, nonce, x.size)
    inv = np.empty_like(perm); inv[perm] = np.arange(perm.size); rows = inv[: x.size] // selcm2.L
    cand = np.where((rows >= R_ - 8) | (rows < 8))[0]; return list(rng.choice(cand, min(k, cand.size), replace=False))
def save(R): json.dump(R, open(RES + "/v2_results.json", "w"), indent=1)

for n, (E, D) in M.items(): x = Z[FILES[0]][:5000]; D(E(x, KEY, NONCE), KEY, NONCE, x.size)
R = {"per_file": {}, "multikey": {}, "cpa": {}, "kpa": {}, "attacks": {}, "quality": {}, "bitdepth": {}, "ablation": {}, "framed": {}, "speed_py": {}}
t0 = time.time()
# ---------------- A. per-file
for f in FILES:
    x = Z[f]; pos = [0, x.size - 1] + list(rng.integers(0, x.size, 2)) + edge_positions(x, KEY, NONCE, 2)
    for mn, (E, D) in M.items():
        c = E(x, KEY, NONCE); p = D(c, KEY, NONCE, x.size); m = min(c.size, x.size)
        ns, ua = zip(*[nu(c, E(tweak(x, q), KEY, NONCE)) for q in pos])
        R["per_file"].setdefault(mn, {})[f] = {
            "cat": META[f]["cat"], "plain_adj": corr(x[:-1], x[1:]), "adj": corr(c[:-1], c[1:]),
            "pc": corr(x[:m], c[:m].view(np.int16)), "H16": h16(c), "H8": h8(c), "H16_plain": h16(u16(x)),
            "chi2_p": float(chisquare(np.bincount(c.view(np.uint8), minlength=256)).pvalue),
            "NSCR_mean": float(np.mean(ns)), "NSCR_min": float(np.min(ns)), "UACI_mean": float(np.mean(ua)),
            "lossless": bool(np.array_equal(p, x))}
print("A", round(time.time() - t0), flush=True); save(R)
# ---------------- B. 30 keys x 8 representative files (v2)
REP = ["A1 LibriSpeech 198-209-0000", "AudioMNIST spk01", "FSDD jackson take0", "ESC-50 1-100032-A-0.wav", "Humpback whale song",
       "Vibe Ace seg01", "Hungarian Dance 5 seg01", "Solo trumpet seg01"]
for f in REP:
    x = Z[f]; rows = []
    for k in range(30):
        key = hashlib.sha256(b"mk%d" % k).digest(); nonce = hashlib.md5(b"mk%d" % k).digest(); c = selcm2.encrypt(x, key, nonce)
        pos = [int(rng.integers(0, x.size))] + edge_positions(x, key, nonce, 2)
        ns, ua = zip(*[nu(c, selcm2.encrypt(tweak(x, q), key, nonce)) for q in pos])
        k2 = bytearray(key); bit = int(rng.integers(0, 256)); k2[bit // 8] ^= 1 << (bit % 8); k2 = bytes(k2)
        n2 = bytearray(nonce); n2[int(rng.integers(0, 16))] ^= 1 << int(rng.integers(0, 8)); n2 = bytes(n2)
        kc = nu(c, selcm2.encrypt(x, k2, nonce)); ncg = nu(c, selcm2.encrypt(x, key, n2)); wrong = selcm2.decrypt(c, k2, nonce, x.size)
        rows.append([np.mean(ns), np.min(ns), np.mean(ua), corr(c[:-1], c[1:]), h8(c), kc[0], kc[1], ncg[0], float((wrong != x).mean() * 100), corr(wrong, x)])
    A = np.array(rows); names = ["NSCR_mean", "NSCR_min", "UACI_mean", "adj", "H8", "keyflip_cipher_diff", "keyflip_cipher_UACI", "nonceflip_cipher_diff", "wrongkey_diff", "wrongkey_corr"]
    R["multikey"][f] = {nm: {"mean": float(A[:, i].mean()), "sd": float(A[:, i].std(ddof=1)), "min": float(A[:, i].min()), "max": float(A[:, i].max())} for i, nm in enumerate(names)}
print("B", round(time.time() - t0), flush=True); save(R)
# ---------------- C. CPA (all-zero chosen plaintext) on A1 and KPA (10 AudioMNIST pairs)
xt = Z["A1 LibriSpeech 198-209-0000"]
def strip(ct, c0, p0, target):
    n = target.size; g1 = (ct[:n] ^ c0[:n] ^ u16(p0)[:n]); g2 = ((ct[:n].astype(np.int64) - c0[:n].astype(np.int64) + u16(p0)[:n].astype(np.int64)) & 0xFFFF).astype(np.uint16)
    return max(float((g1.view(np.int16) == target).mean() * 100), float((g2.view(np.int16) == target).mean() * 100))
sp = [f for f in FILES if f.startswith("AudioMNIST")][:20]
for mn, (E, D) in M.items():
    out = {}
    for mode in ["nonce reused", "fresh nonce"]:
        n2 = NONCE if mode == "nonce reused" else hashlib.md5(b"other").digest()
        z = np.zeros_like(xt); out["cpa " + mode] = strip(E(xt, KEY, n2), E(z, KEY, NONCE), z, xt)
        vals = []
        for i in range(10):
            p1 = Z[sp[2 * i]]; p2 = Z[sp[2 * i + 1]]; n = min(p1.size, p2.size); p1 = p1[:n]; p2 = p2[:n]
            n3 = NONCE if mode == "nonce reused" else hashlib.md5(b"fresh%d" % i).digest()
            vals.append(strip(E(p2, KEY, n3), E(p1, KEY, NONCE), p1, p2))
        out["kpa " + mode] = float(np.mean(vals)); out["kpa max " + mode] = float(np.max(vals))
    R["cpa"][mn] = out
print("C", round(time.time() - t0), flush=True); save(R)
# ---------------- D. impairments (whole-file), on 8 representative files
def framed_fns(F):
    return (lambda x, k, n: selcm2.encrypt_framed(x, k, n, F), lambda cs, k, n, m: selcm2.decrypt_framed(cs, k, n, m, F))
IMP = {"SELCM-AE v2": M["SELCM-AE v2"], "SELCM-AE v2 framed 16384": framed_fns(16384), "B4 AES-256-CTR": M["B4 AES-256-CTR"], "B1 Logistic-XOR": M["B1 Logistic-XOR"]}
for mn, (E, D) in IMP.items():
    out = {}; framed = "framed" in mn
    for f in REP:
        x = Z[f]; c0 = E(x, KEY, NONCE); flat = np.concatenate(c0) if framed else c0
        sizes = [c.size for c in c0] if framed else None
        def dec(cc): return D(np.split(cc, np.cumsum(sizes)[:-1]) if framed else cc, KEY, NONCE, x.size)
        sd = np.std(flat.view(np.int16).astype(float))
        errs = []
        for q in rng.integers(0, flat.size, 10):
            cc = flat.copy(); cc[q] ^= 0x0101; errs.append(int((dec(cc) != x).sum()))
        out.setdefault("one corrupted sample", []).append(float(np.mean(errs)))
        for frac in [0.01, 0.05, 0.10]:
            cc = flat.copy(); s0 = cc.size // 3; cc[s0:s0 + int(cc.size * frac)] = 0; d = dec(cc)
            out.setdefault(f"crop {int(frac*100)}%", []).append(float((d != x).mean() * 100))
        for frac in [0.001, 0.01]:
            cc = flat.copy(); idx = rng.choice(cc.size, int(cc.size * frac), replace=False); cc[idx] = rng.integers(0, 65536, idx.size).astype(np.uint16)
            out.setdefault(f"random sample errors {frac*100:g}%", []).append(float((dec(cc) != x).mean() * 100))
        for sn in [20, 50]:
            cc = np.clip(np.round(flat.view(np.int16).astype(float) + rng.normal(0, sd * 10 ** (-sn / 20), flat.size)), -32768, 32767).astype(np.int16).view(np.uint16)
            out.setdefault(f"AWGN {sn} dB", []).append(float((dec(cc) != x).mean() * 100))
    R["attacks"][mn] = {k: float(np.mean(v)) for k, v in out.items()}
print("D", round(time.time() - t0), flush=True); save(R)
# ---------------- E. perceptual: STOI, PESQ, segSNR on 55 speech files; STOI after 5% crop on 20 files
def pesq_any(ref, deg, sr):
    if sr == 8000: return pesq(8000, ref, deg, "nb")
    if sr != 16000:
        g = np.gcd(sr, 16000); ref = resample_poly(ref, 16000 // g, sr // g); deg = resample_poly(deg, 16000 // g, sr // g)
    return pesq(16000, ref, deg, "wb")
for mn in ["SELCM-AE v2", "B4 AES-256-CTR"]:
    E, D = M[mn]; rows = []
    for f in [f for f in FILES if META[f]["cat"] == "speech"]:
        x = Z[f]; sr = META[f]["sr"]; c = E(x, KEY, NONCE)[: x.size].view(np.int16); d = D(E(x, KEY, NONCE), KEY, NONCE, x.size)
        xf = x.astype(float); cf = c.astype(float)
        try: pq = pesq_any(xf, cf, sr)
        except Exception: pq = np.nan
        rows.append([stoi(xf, cf, sr), segsnr(x, c, sr), pq, stoi(xf, d.astype(float), sr), pesq_any(xf, d.astype(float), sr)])
    A = np.array(rows, dtype=float)
    R["quality"][mn] = {"n_files": len(rows), "STOI_cipher_mean": float(np.nanmean(A[:, 0])), "STOI_cipher_max": float(np.nanmax(A[:, 0])),
                        "segSNR_cipher_mean": float(np.nanmean(A[:, 1])), "PESQ_cipher_mean": float(np.nanmean(A[:, 2])), "PESQ_cipher_max": float(np.nanmax(A[:, 2])),
                        "PESQ_valid": int(np.sum(~np.isnan(A[:, 2]))), "STOI_decrypted_mean": float(np.nanmean(A[:, 3])), "PESQ_decrypted_mean": float(np.nanmean(A[:, 4]))}
cropq = {}
for mn, (E, D) in IMP.items():
    if mn == "B1 Logistic-XOR": continue
    framed = "framed" in mn; vals = []; pv = []
    for f in [f for f in FILES if f.startswith("AudioMNIST")][:20]:
        x = Z[f]; sr = META[f]["sr"]; c0 = E(x, KEY, NONCE); flat = np.concatenate(c0) if framed else c0; sizes = [c.size for c in c0] if framed else None
        cc = flat.copy(); s0 = cc.size // 3; cc[s0:s0 + int(0.05 * cc.size)] = 0
        d = D(np.split(cc, np.cumsum(sizes)[:-1]) if framed else cc, KEY, NONCE, x.size)
        vals.append(stoi(x.astype(float), d.astype(float), sr)); pv.append(pesq_any(x.astype(float), d.astype(float), sr))
    cropq[mn] = {"STOI_after_5pct_crop": float(np.mean(vals)), "PESQ_after_5pct_crop": float(np.mean(pv))}
R["quality"]["after_crop"] = cropq
print("E", round(time.time() - t0), flush=True); save(R)
# ---------------- F. 24-bit and compressed-file payloads
for setname in ["set24", "setogg"]:
    S = np.load(f"{DATA}/{setname}.npz"); out = {}
    for f in S.files:
        x = S[f]; c = selcm2.encrypt(x, KEY, NONCE); p = selcm2.decrypt(c, KEY, NONCE, x.size)
        pos = [0, x.size - 1] + list(rng.integers(0, x.size, 2)); ns = [nu(c, selcm2.encrypt(tweak(x, q), KEY, NONCE))[0] for q in pos]
        pb = x.view(np.uint8); cb = c.view(np.uint8)
        out[f] = {"words": int(x.size), "lossless": bool(np.array_equal(p, x)), "H8_plain": h8(u16(x)), "H8_cipher": h8(c),
                  "byte_adj_plain": corr(pb[:-1], pb[1:]), "byte_adj_cipher": corr(cb[:-1], cb[1:]), "NSCR_mean": float(np.mean(ns)), "NSCR_min": float(np.min(ns))}
    R["bitdepth"][setname] = out
print("F", round(time.time() - t0), flush=True); save(R)
# ---------------- G. ablation on A1 (62 positions: first, last, 30 random, 30 edge)
x = xt; pos = [0, x.size - 1] + list(rng.integers(0, x.size, 30)) + edge_positions(x, KEY, NONCE, 30)
Np, R_, S2, Si2, perm, K, a1, a2 = selcm2._streams(KEY, NONCE, x.size)
v1S = selcm._streams(KEY, NONCE, 1000)[2].astype(np.uint16)   # v1 S-box: stable rank of 65536 keystream words
variants = {"Full SELCM-AE v2": {}, "Two passes": {"npass": 2}, "No pre-S-box layer": {"pre": False}, "No cyclic wrap": {"wrap": False},
            "Random S-box (as v1)": {"sbox": v1S}, "No permutation": {"use_perm": False}, "One pass, no pre layer": {"npass": 1, "pre": False}}
for vn, kw in variants.items():
    E = lambda a, kk=KEY, nn=NONCE: selcm2.encrypt(a, kk, nn, **kw); c = E(x)
    ns, ua = zip(*[nu(c, E(tweak(x, q))) for q in pos]); z = np.zeros_like(x)
    cpa = strip(E(x), E(z), z, x)
    R["ablation"][vn] = {"NSCR_mean": float(np.mean(ns)), "NSCR_min": float(np.min(ns)), "UACI_mean": float(np.mean(ua)), "adj": corr(c[:-1], c[1:]), "cpa_same_nonce": cpa}
print("G", round(time.time() - t0), flush=True); save(R)
# ---------------- H. framed-mode trade-off and Python speed
files = [f for f in FILES if f.startswith("AudioMNIST")][:10]
for F in [None, 65536, 16384, 4096]:
    rows = []
    for f in files:
        x = Z[f]; sr = META[f]["sr"]
        E = (lambda a: [selcm2.encrypt(a, KEY, NONCE)]) if F is None else (lambda a: selcm2.encrypt_framed(a, KEY, NONCE, F))
        cs = E(x); flat = np.concatenate(cs); q = int(rng.integers(0, x.size)); cs2 = E(tweak(x, q)); flat2 = np.concatenate(cs2)
        fi = 0 if F is None else q // F
        t = time.perf_counter(); E(x); te = time.perf_counter() - t
        rows.append([(flat != flat2).mean() * 100, (cs[fi] != cs2[fi]).mean() * 100, x.nbytes / 1e6 / te])
    A = np.array(rows); R["framed"]["whole file" if F is None else f"{F} samples"] = {"NSCR_file": float(A[:, 0].mean()), "NSCR_frame": float(A[:, 1].mean()), "NSCR_frame_min": float(A[:, 1].min()), "MBps_python": float(A[:, 2].mean())}
x = Z["Vibe Ace seg01"][: 441000]
for mn, (E, D) in M.items():
    te, td = [], []
    for _ in range(15):
        t = time.perf_counter(); c = E(x, KEY, NONCE); te.append(time.perf_counter() - t)
        t = time.perf_counter(); D(c, KEY, NONCE, x.size); td.append(time.perf_counter() - t)
    R["speed_py"][mn] = {"enc_MBps": x.nbytes / 1e6 / float(np.median(te)), "dec_MBps": x.nbytes / 1e6 / float(np.median(td))}
print("H", round(time.time() - t0), flush=True); save(R); print("ok")
