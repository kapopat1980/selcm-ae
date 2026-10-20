import numpy as np, hashlib
from scipy.stats import chisquare
import selcm2 as S
for kk in range(3):
    a, b, eps, X, Y, _, _ = S.key_schedule(hashlib.sha256(b"w16%d" % kk).digest(), bytes(16), 1000)
    w = S.lattice_keystream(X, Y, a, b, eps, 1 << 24).astype(np.uint32)
    ps = []
    for o in range(16):
        win = ((w[:-1] << o) | (w[1:] >> (16 - o))) & 0xFFFF if o else w
        ps.append(chisquare(np.bincount(win, minlength=65536)).pvalue)
    # consecutive-word pair correlation by position within lane group
    corr = [np.corrcoef(w[t:-4:4].astype(float), w[t+1:-3:4].astype(float))[0,1] for t in range(4)]
    print(kk, 'window chi2 p by offset:', ' '.join(f'{p:.2g}' for p in ps), '| corr word t,t+1 by type:', ' '.join(f'{c:+.4f}' for c in corr), flush=True)
