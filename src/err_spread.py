from paths import DATA, RES, FIG
import json, hashlib, numpy as np, selcm, selcm2, baselines as B
Z = np.load(DATA + "/dataset2.npz"); rng = np.random.default_rng(11)
KEY = hashlib.sha256(b"v2 key").digest(); NONCE = hashlib.md5(b"v2 nonce").digest()
REP = ["A1 LibriSpeech 198-209-0000", "AudioMNIST spk01", "FSDD jackson take0", "ESC-50 1-100032-A-0.wav", "Humpback whale song", "Vibe Ace seg01", "Hungarian Dance 5 seg01", "Solo trumpet seg01"]
M = {"SELCM-AE v1": (lambda x: selcm.encrypt(x, KEY, NONCE), lambda c, m: selcm.decrypt(c, KEY, NONCE, m)),
     "B2": (lambda x: B.b2_encrypt(x, KEY), lambda c, m: B.b2_decrypt(c, KEY, m)),
     "B3": (lambda x: B.b3_encrypt(x, KEY), lambda c, m: B.b3_decrypt(c, KEY, m))}
out = {}
for mn, (E, D) in M.items():
    r = []
    for f in REP:
        x = Z[f]; c = E(x); e = []
        for q in rng.integers(0, c.size, 10):
            cc = c.copy(); cc[q] ^= 0x0101; e.append(int((D(cc, x.size) != x).sum()))
        r.append(np.mean(e))
    out[mn] = float(np.mean(r)); print(mn, out[mn], flush=True)
json.dump(out, open(RES + "/err_spread.json", "w"))
