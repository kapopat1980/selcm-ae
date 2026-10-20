import numpy as np, hashlib
from scipy.stats import chisquare
import selcm2 as S, nist
tot = {t: np.zeros(65536) for t in range(4)}
for kk in range(4):
    a, b, eps, X, Y, _, _ = S.key_schedule(hashlib.sha256(b"wb%d" % kk).digest(), bytes(16), 1000)
    w = S.lattice_keystream(X, Y, a, b, eps, 1 << 25)
    for t in range(4): tot[t] += np.bincount(w[t::4], minlength=65536)
for t in range(4):
    c = tot[t]; hb = c.reshape(256, 256).sum(1)
    print('word type', t, 'chi2 16-bit p=%.3g' % chisquare(c).pvalue, ' high-byte p=%.3g' % chisquare(hb).pvalue, ' low-byte p=%.3g' % chisquare(c.reshape(256,256).sum(0)).pvalue)
# serial test per stream variant
def seqs(sel, n=40):
    out = []
    for i in range(n):
        a, b, eps, X, Y, _, _ = S.key_schedule(hashlib.sha256(b"sv%d" % i).digest(), bytes(16), 1000)
        w = S.lattice_keystream(X, Y, a, b, eps, 4 * 62600)
        w = sel(w); out.append(np.unpackbits(w.view(np.uint8))[:1000000])
    return out
for name, sel in [("all words", lambda w: w), ("type 0 only", lambda w: np.ascontiguousarray(w[0::4])), ("type 1 only", lambda w: np.ascontiguousarray(w[1::4])),
                  ("type 2 only", lambda w: np.ascontiguousarray(w[2::4])), ("type 3 only", lambda w: np.ascontiguousarray(w[3::4]))]:
    ps = np.array([nist.serial(s) for s in seqs(sel)]); print(name, 'serial pass %.2f' % np.mean(ps >= 0.01), 'median p %.3f' % np.median(ps), flush=True)
