"""Adaptive two-sample cancellation distinguisher (nonce reuse) suggested by the MILP model."""
from paths import DATA, RES, FIG
import json, hashlib, numpy as np
import baselines as B, selcm2, selcm
Z = np.load(DATA + "/dataset2.npz"); x = Z["A1 LibriSpeech 198-209-0000"]
KEY = hashlib.sha256(b"adapt").digest(); NONCE = hashlib.md5(b"adapt").digest(); rng = np.random.default_rng(4)
u16 = selcm2.to_u16; out = {}
def attack(E, name, trials=30):
    conf = []
    for _ in range(trials):
        i = int(rng.integers(1, x.size - 2)); c = E(x)
        y = x.copy(); y[i] = y[i] + 1 if y[i] < 32767 else y[i] - 1; c1 = E(y)
        # B3: c_i = rotl5((p_i + c_{i-1} + k_i) mod 2^16) ^ k'_i ; choose p'_{i+1} so that p'_{i+1} + c'_i = p_{i+1} + c_i
        v = (int(u16(x)[i + 1]) + int(c[i]) - int(c1[i])) & 0xFFFF
        z = y.copy(); z[i + 1] = v - 65536 if v >= 32768 else v
        c2 = E(z); conf.append(float((c2 != c).mean() * 100))
    out[name] = {"NSCR_after_adaptive_pair_mean": float(np.mean(conf)), "max": float(np.max(conf)), "min": float(np.min(conf)), "trials": trials}
    print(name, out[name], flush=True)
attack(lambda a: B.b3_encrypt(a, KEY), "B3 Forward chained diffusion")
attack(lambda a: selcm.encrypt(a, KEY, NONCE), "SELCM-AE v1", 10)
attack(lambda a: selcm2.encrypt(a, KEY, NONCE), "SELCM-AE v2", 10)
json.dump(out, open(RES + "/adaptive.json", "w"), indent=1)
