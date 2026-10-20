from paths import DATA, RES, FIG
import json, hashlib, os, numpy as np, time
import selcm2 as selcm, baselines as B, nist_full as F
NSEQ, NB = 100, 1_000_000; nw = NB // 16 + 1
bits = lambda w: np.unpackbits(w.view(np.uint8))[:NB]
def g_prop(i):
    k = hashlib.sha256(b"nist%d" % i).digest(); a, b, eps, X, Y, _, _ = selcm.key_schedule(k, bytes(16), 1000)
    return selcm.lattice_keystream(X, Y, a, b, eps, nw)
def g_log(i): return B.logistic_ks(0.1 + 0.8 * B._seed(b"%d" % i, b"B1", 0), 3.99 + 0.0099 * B._seed(b"%d" % i, b"B1", 1), nw)
def g_lor(i): return B.lorenz_ks(1 + B._seed(b"%d" % i, b"B2", 2), 1 + B._seed(b"%d" % i, b"B2", 3), 1.5, nw)
def g_aes(i): return B.b4_encrypt(np.zeros(nw, np.int16), hashlib.sha256(b"%d" % i).digest(), bytes(16))
def g_cipher(i):  # SELCM-AE ciphertext of an all-zero (worst-case) audio file under 100 nonces
    return selcm.encrypt(np.zeros(nw, np.int16), hashlib.sha256(b"ct").digest(), hashlib.md5(b"%d" % i).digest())[:nw]
out = {}
for name, g in [("SELCM-AE v2 lattice keystream", g_prop), ("SELCM-AE v2 ciphertext of silence", g_cipher)]:
    t = time.time(); out[name] = F.battery([bits(g(i)) for i in range(NSEQ)])
    print(name, round(time.time() - t), {k: round(v["prop"], 2) for k, v in out[name].items()}, flush=True)
    json.dump(out, open(RES + "/nist_full_v2.json", "w"), indent=1)
