from paths import DATA, RES, FIG
import numpy as np, os, hashlib, json
from scipy.special import gammaincc
from scipy.stats import chisquare
import nist, selcm2, baselines as B
def serial2(e, m=16):
    p0, p1, p2 = nist._psi(e, m), nist._psi(e, m - 1), nist._psi(e, m - 2)
    return gammaincc(2 ** (m - 2), (p0 - p1) / 2), gammaincc(2 ** (m - 3), (p0 - 2 * p1 + p2) / 2)
def summ(ps):
    ps = np.array(ps); return float(np.mean(ps >= 0.01)), float(chisquare(np.histogram(ps, 10, (0, 1))[0]).pvalue)
NB = 1_000_000; nw = NB // 16 + 1; bits = lambda w: np.unpackbits(w.view(np.uint8))[:NB]
def g_prop(i):
    k = hashlib.sha256(b"nist%d" % i).digest(); a, b, eps, X, Y, _, _ = selcm2.key_schedule(k, bytes(16), 1000); return selcm2.lattice_keystream(X, Y, a, b, eps, nw)
def g_cipher(i): return selcm2.encrypt(np.zeros(nw, np.int16), hashlib.sha256(b"ct").digest(), hashlib.md5(b"%d" % i).digest())[:nw]
def g_log(i): return B.logistic_ks(0.1 + 0.8 * B._seed(b"%d" % i, b"B1", 0), 3.99 + 0.0099 * B._seed(b"%d" % i, b"B1", 1), nw)
def g_lor(i): return B.lorenz_ks(1 + B._seed(b"%d" % i, b"B2", 2), 1 + B._seed(b"%d" % i, b"B2", 3), 1.5, nw)
def g_aes(i): return B.b4_encrypt(np.zeros(nw, np.int16), hashlib.sha256(b"%d" % i).digest(), bytes(16))
def g_urand(i): return np.frombuffer(os.urandom(2 * nw), dtype=np.uint16)
out = {}
for name, g in [("os.urandom (control)", g_urand), ("SELCM-AE v2 lattice keystream", g_prop), ("SELCM-AE v2 ciphertext of silence", g_cipher),
                ("Logistic map (B1/B3)", g_log), ("Lorenz system (B2)", g_lor), ("AES-256-CTR (B4)", g_aes)]:
    P = np.array([serial2(bits(g(i))) for i in range(100)])
    out[name] = {"Serial-1": summ(P[:, 0]), "Serial-2": summ(P[:, 1]), "min-of-two (old)": summ(P.min(1))}
    print(name, out[name], flush=True)
json.dump(out, open(RES + "/serial_fix.json", "w"), indent=1)
