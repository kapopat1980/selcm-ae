"""SELCM-AE: audio cipher built on the 2D Sine-Exponential-Logistic Cross-Coupled Map.

Pipeline (16-bit PCM samples):
  1. KDF: SHAKE-256(key || nonce || length) -> map parameters, lattice seeds, IVs
  2. Coupled lattice of L SELCM maps -> 16-bit keystream words (4 per lane-step)
  3. Key-dependent 16-bit S-box (bijection) + keystream-driven Fisher-Yates sample permutation
  4. Forward row-chained diffusion with butterfly cross-lane mixing + 8-row cyclic wrap
  5. Backward row-chained diffusion (same structure, reverse order) + 8-row cyclic wrap
Decryption inverts 5 -> 4 -> 3.
"""
import hashlib, math
import numpy as np
from numba import njit

L = 256            # lanes (columns); must be a power of two
LOG2L = 8
BURN = 32
TAG = b"SELCM-AE|v1|"

# ---------------------------------------------------------------- chaotic map
@njit(cache=True, fastmath=False)
def _h(u):
    return math.exp(u) * (1.0 + u * (1.0 - u))

@njit(cache=True)
def _frac(v):
    return v - math.floor(v)

@njit(cache=True)
def selcm_step(x, y, a, b):
    xn = _frac(a * math.sin(math.pi * y) + b * _h(x))
    yn = _frac(a * math.sin(math.pi * xn) + b * _h(y))
    return xn, yn

@njit(cache=True)
def lattice_keystream(X, Y, a, b, eps, nwords):
    """Coupled map lattice of len(X) SELCM maps -> uint16 keystream."""
    n = X.shape[0]
    out = np.empty(nwords + 4 * n, dtype=np.uint16)
    xn = np.empty(n); yn = np.empty(n)
    k = 0
    step = 0
    while k < nwords:
        for j in range(n):
            xn[j], yn[j] = selcm_step(X[j], Y[j], a, b)
        for j in range(n):
            X[j] = _frac((1.0 - eps) * xn[j] + eps * xn[(j + 1) % n])
            Y[j] = _frac((1.0 - eps) * yn[j] + eps * yn[(j - 1) % n])
        step += 1
        if step <= BURN:
            continue
        for j in range(n):
            mx = np.uint64(X[j] * 4503599627370496.0)  # 2^52
            my = np.uint64(Y[j] * 4503599627370496.0)
            out[k] = (mx >> np.uint64(12)) & np.uint64(0xFFFF)
            out[k + 1] = (my >> np.uint64(12)) & np.uint64(0xFFFF)
            out[k + 2] = (mx >> np.uint64(28)) & np.uint64(0xFFFF)
            out[k + 3] = (my >> np.uint64(28)) & np.uint64(0xFFFF)
            k += 4
    return out[:nwords]

# ---------------------------------------------------------------- diffusion
@njit(cache=True)
def _rotl16(v, r):
    return ((v << r) | (v >> (16 - r))) & 0xFFFF

@njit(cache=True)
def diffuse(M, order, krow, K1, K2, S, IV):
    """Row-chained diffusion along `order` (rows may repeat: cyclic wrap). In place."""
    R, n = M.shape
    prev = IV.astype(np.int64).copy(); cur = np.empty(n, dtype=np.int64)
    for k in range(order.size):
        r = order[k]; kr = krow[k]; s = 1 << (k % LOG2L)
        for j in range(n):
            m = (prev[j] + _rotl16(prev[(j - s) % n], 7)) & 0xFFFF
            t = (np.int64(M[r, j]) + m + np.int64(K1[kr, j])) & 0xFFFF
            cur[j] = np.int64(S[t]) ^ np.int64(K2[kr, j])
        for j in range(n):
            M[r, j] = cur[j]; prev[j] = cur[j]
    return M

@njit(cache=True)
def undiffuse(M, order, krow, K1, K2, Sinv, IV):
    R, n = M.shape
    for k in range(order.size - 1, -1, -1):
        r = order[k]; kr = krow[k]; s = 1 << (k % LOG2L)
        for j in range(n):
            if k == 0:
                a = np.int64(IV[j]); bb = np.int64(IV[(j - s) % n])
            else:
                a = np.int64(M[order[k - 1], j]); bb = np.int64(M[order[k - 1], (j - s) % n])
            m = (a + _rotl16(bb, 7)) & 0xFFFF
            t = np.int64(Sinv[np.int64(M[r, j]) ^ np.int64(K2[kr, j])])
            M[r, j] = (t - m - np.int64(K1[kr, j])) & 0xFFFF
    return M

def schedules(R, wrap=True):
    W = LOG2L if (wrap and R > LOG2L) else 0
    f = np.concatenate((np.arange(R), np.arange(W))).astype(np.int64)
    fk = np.concatenate((np.arange(R), R - 1 - np.arange(W))).astype(np.int64)
    b = np.concatenate((np.arange(R - 1, -1, -1), np.arange(R - 1, R - 1 - W, -1))).astype(np.int64)
    bk = np.concatenate((np.arange(R - 1, -1, -1), np.arange(W))).astype(np.int64)
    return (f, fk), (b, bk)

@njit(cache=True)
def fisher_yates(words, n):
    """Keystream-driven Fisher-Yates shuffle; 32-bit draw per swap, multiply-shift range reduction."""
    p = np.arange(n)
    for i in range(n - 1, 0, -1):
        w = (np.uint64(words[2 * i]) << np.uint64(16)) | np.uint64(words[2 * i + 1])
        j = np.int64((w * np.uint64(i + 1)) >> np.uint64(32))
        t = p[i]; p[i] = p[j]; p[j] = t
    return p

# ---------------------------------------------------------------- key schedule
def _u(b):  # 8 bytes -> float in [0,1)
    return (int.from_bytes(b, "little") >> 11) / float(1 << 53)

def key_schedule(key: bytes, nonce: bytes, n_samples: int):
    assert len(key) == 32 and len(nonce) == 16
    need = 24 + 16 * L + 4 * L
    d = hashlib.shake_256(TAG + key + nonce + n_samples.to_bytes(8, "little")).digest(need)
    a = 0.5 + 7.5 * _u(d[0:8])
    b = 10.0 + 90.0 * _u(d[8:16])
    eps = 0.05 + 0.10 * _u(d[16:24])
    o = 24
    X = np.array([_u(d[o + 8 * i:o + 8 * i + 8]) for i in range(L)]); o += 8 * L
    Y = np.array([_u(d[o + 8 * i:o + 8 * i + 8]) for i in range(L)]); o += 8 * L
    iv = np.frombuffer(d[o:o + 4 * L], dtype=np.uint16).copy()
    return a, b, eps, X, Y, iv[:L], iv[L:]

def _streams(key, nonce, n):
    Np = -(-n // L) * L
    R = Np // L
    a, b, eps, X, Y, iv1, iv2 = key_schedule(key, nonce, n)
    total = 65536 + 2 * Np + 4 * Np
    ks = lattice_keystream(X, Y, a, b, eps, total)
    o = 0
    S = np.argsort(ks[o:o + 65536], kind="stable").astype(np.uint16); o += 65536
    perm = fisher_yates(ks[o:o + 2 * Np], Np); o += 2 * Np
    K = ks[o:o + 4 * Np].reshape(4, R, L)
    return Np, R, S, perm, K, iv1, iv2

# ---------------------------------------------------------------- API
def encrypt(samples, key, nonce, passes=("fwd", "bwd"), use_sbox=True, use_perm=True, wrap=True):
    x = (samples.astype(np.int32) & 0xFFFF).astype(np.uint16)   # int16 -> uint16 two's complement
    n = x.size
    Np, R, S, perm, K, iv1, iv2 = _streams(key, nonce, n)
    if not use_sbox:
        S = np.arange(65536, dtype=np.uint16)
    buf = np.zeros(Np, dtype=np.uint16); buf[:n] = x
    if use_perm:
        buf = buf[perm]
    M = buf.reshape(R, L).copy()
    (f, fk), (b, bk) = schedules(R, wrap)
    if "fwd" in passes:
        diffuse(M, f, fk, K[0], K[1], S, iv1)
    if "bwd" in passes:
        diffuse(M, b, bk, K[2], K[3], S, iv2)
    return M.reshape(-1)

def decrypt(cipher, key, nonce, n, wrap=True):
    Np, R, S, perm, K, iv1, iv2 = _streams(key, nonce, n)
    Sinv = np.argsort(S, kind="stable").astype(np.uint16)
    M = cipher.astype(np.uint16).reshape(R, L).copy()
    (f, fk), (b, bk) = schedules(R, wrap)
    undiffuse(M, b, bk, K[2], K[3], Sinv, iv2)
    undiffuse(M, f, fk, K[0], K[1], Sinv, iv1)
    buf = np.empty(Np, dtype=np.uint16); buf[perm] = M.reshape(-1)
    return buf[:n].view(np.int16).copy()
