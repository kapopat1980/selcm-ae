import numpy as np, hashlib, math
from numba import njit
import selcm2 as S
@njit(cache=True)
def raw(X, Y, a, b, eps, steps):
    n = X.shape[0]; out = np.empty((steps, n), dtype=np.uint64); xn = np.empty(n); yn = np.empty(n)
    for t in range(steps + 32):
        for j in range(n): xn[j], yn[j] = S.selcm_step(X[j], Y[j], a, b)
        for j in range(n):
            X[j] = S._frac((1.0 - eps) * xn[j] + eps * xn[(j + 1) % n]); Y[j] = S._frac((1.0 - eps) * yn[j] + eps * yn[(j - 1) % n])
        if t >= 32:
            for j in range(n): out[t - 32, j] = np.uint64(X[j] * 4503599627370496.0)
    return out
res = []
for kk in range(6):
    a, b, eps, X, Y, _, _ = S.key_schedule(hashlib.sha256(b"bb%d" % kk).digest(), bytes(16), 1000)
    m = raw(X, Y, a, b, eps, 8000).ravel()
    bits = np.array([((m >> np.uint64(t)) & np.uint64(1)).mean() for t in range(52)])
    res.append(bits); print(kk, round(a, 2), round(b, 1), 'bias (|p-0.5|*1e3) bits 0..51:', ' '.join(f'{abs(v-0.5)*1e3:.1f}' for v in bits), flush=True)
