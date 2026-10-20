"""SELCM-AE v2: audio cipher on the 2D Sine-Exponential-Logistic Cross-Coupled Map.

Changes from v1 (all measured in the paper):
  * Portable map: sin(pi*y) and e^u are fixed polynomials (degree 20 / 17) evaluated by Horner's rule
    with IEEE-754 double arithmetic and no fused multiply-add, so every platform produces the same bits.
  * S-box: keyed affine equivalent of inversion in GF(2^16): S(x) = A * inv(B*x ^ beta) ^ alpha,
    A and B key-dependent invertible 16x16 binary matrices (P*L*U form). Differential uniformity 4,
    nonlinearity 32512 for every key (affine equivalence preserves both).
  * Three diffusion passes (forward, backward, forward), each step with a keyed pre-S-box layer;
    files are padded to at least 16 rows.
  * Mixing: a bijective two-round Feistel butterfly on lane pairs (j, j ^ s), s = 2^(k mod 8), so that
    a nonzero difference in one row always yields a nonzero mixing difference in the next row.
  * Framed mode: one S-box per file, per-frame nonce for permutation and diffusion keys.
"""
import hashlib, math
import numpy as np
from numba import njit

L = 256
LOG2L = 8
BURN = 32
TAG = b"SELCM-AE|v2|"
SBOX_WORDS = 128
NPASS = 3
MIN_ROWS = 16          # files shorter than 16 x 256 samples are zero-padded (bound in Section 4.7)

# ---------------------------------------------------------------- portable elementary functions
COS_C = np.array([(-1.0) ** k / math.factorial(2 * k) for k in range(11)])      # cos(t) = sum c_k t^(2k)
EXP_C = np.array([1.0 / math.factorial(k) for k in range(18)])                   # e^u   = sum c_k u^k
PI = 3.141592653589793

@njit(cache=True)
def psin_pi(y):
    """sin(pi*y) for y in [0,1): cos(pi*(y-0.5)) as a degree-20 even polynomial (|error| < 1e-16)."""
    t = PI * (y - 0.5); w = t * t
    c = COS_C[10]
    for k in range(9, -1, -1):
        c = c * w + COS_C[k]
    return c

@njit(cache=True)
def pexp(u):
    """e^u for u in [0,1): degree-17 Taylor polynomial (relative error < 2e-16)."""
    c = EXP_C[17]
    for k in range(16, -1, -1):
        c = c * u + EXP_C[k]
    return c

@njit(cache=True)
def _h(u):
    return pexp(u) * (1.0 + u * (1.0 - u))

@njit(cache=True)
def _frac(v):
    return v - math.floor(v)

@njit(cache=True)
def selcm_step(x, y, a, b):
    xn = _frac(a * psin_pi(y) + b * _h(x))
    yn = _frac(a * psin_pi(xn) + b * _h(y))
    return xn, yn

@njit(cache=True)
def lattice_keystream(X, Y, a, b, eps, nwords):
    n = X.shape[0]
    out = np.empty(nwords + 4 * n, dtype=np.uint16)
    xn = np.empty(n); yn = np.empty(n)
    k = 0; step = 0
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
            mx = np.uint64(X[j] * 4503599627370496.0)
            my = np.uint64(Y[j] * 4503599627370496.0)
            out[k] = (mx >> np.uint64(12)) & np.uint64(0xFFFF)
            out[k + 1] = (my >> np.uint64(12)) & np.uint64(0xFFFF)
            out[k + 2] = (mx >> np.uint64(28)) & np.uint64(0xFFFF)
            out[k + 3] = (my >> np.uint64(28)) & np.uint64(0xFFFF)
            k += 4
    return out[:nwords]

# ---------------------------------------------------------------- GF(2^16) inversion S-box
POLY = 0x1100B   # x^16 + x^12 + x^3 + x + 1, primitive

def _gf_inv_table():
    exp = np.zeros(65535, dtype=np.int64); log = np.zeros(65536, dtype=np.int64); v = 1
    for i in range(65535):
        exp[i] = v; log[v] = i; v <<= 1
        if v & 0x10000: v ^= POLY
    inv = np.zeros(65536, dtype=np.uint16)
    idx = np.arange(1, 65536)
    inv[1:] = exp[(65535 - log[idx]) % 65535]
    return inv
GF_INV = _gf_inv_table()

@njit(cache=True)
def _matrix(words, o):
    """Invertible 16x16 GF(2) matrix as 16 column words, M = P*L*U, from 32 keystream words at offset o.
    Returns columns col[i] = M * e_i."""
    Lm = np.zeros(16, dtype=np.int64); Um = np.zeros(16, dtype=np.int64)   # row bitmasks
    bits = np.zeros(256, dtype=np.int64)
    for w in range(16):
        v = np.int64(words[o + w])
        for t in range(16): bits[16 * w + t] = (v >> t) & 1
    bi = 0
    for i in range(16):
        Lm[i] = 1 << i; Um[i] = 1 << i
        for j in range(i):
            if bits[bi]: Lm[i] |= 1 << j
            bi += 1
        for j in range(i + 1, 16):
            if bits[bi]: Um[i] |= 1 << j
            bi += 1
    perm = np.arange(16)
    for i in range(15, 0, -1):
        jj = (np.int64(words[o + 16 + i]) * (i + 1)) >> 16
        t = perm[i]; perm[i] = perm[jj]; perm[jj] = t
    # rows of P*L*U: (L*U) rows then permuted
    LU = np.zeros(16, dtype=np.int64)
    for i in range(16):
        r = 0
        for j in range(16):
            if (Lm[i] >> j) & 1: r ^= Um[j]
        LU[i] = r
    rows = np.zeros(16, dtype=np.int64)
    for i in range(16): rows[i] = LU[perm[i]]
    cols = np.zeros(16, dtype=np.int64)
    for c in range(16):
        v = 0
        for i in range(16):
            if (rows[i] >> c) & 1: v |= 1 << i
        cols[c] = v
    return cols

@njit(cache=True)
def _apply(cols, x):
    v = 0
    for i in range(16):
        if (x >> i) & 1: v ^= cols[i]
    return v

@njit(cache=True)
def build_sbox(words, inv):
    A = _matrix(words, 0); B = _matrix(words, 32)
    alpha = np.int64(words[64]); beta = np.int64(words[65])
    S = np.empty(65536, dtype=np.uint16); Si = np.empty(65536, dtype=np.uint16)
    for x in range(65536):
        y = _apply(A, np.int64(inv[_apply(B, x) ^ beta])) ^ alpha
        S[x] = y; Si[y] = x
    return S, Si

# ---------------------------------------------------------------- diffusion with bijective Feistel butterfly
@njit(cache=True)
def _rotl16(v, r):
    return ((v << r) | (v >> (16 - r))) & 0xFFFF

@njit(cache=True)
def _mix(prev, s, m):
    n = prev.size
    for lo in range(n):
        if lo & s: continue
        hi = lo | s
        uh = (prev[hi] + _rotl16(prev[lo], 7)) & 0xFFFF
        ul = (prev[lo] + _rotl16(uh, 11)) & 0xFFFF
        m[hi] = uh; m[lo] = ul

@njit(cache=True)
def diffuse(M, order, krow, K0, K1, S, IV, pre):
    """One pass. Step on row r: z = S[Q ^ K0] (pre layer), t = z + Mix(prev) + K1, v = S[t]; Q_r <- v; prev <- v."""
    R, n = M.shape
    prev = IV.astype(np.int64).copy(); m = np.empty(n, dtype=np.int64)
    for k in range(order.size):
        r = order[k]; kr = krow[k]; s = 1 << (k % LOG2L)
        _mix(prev, s, m)
        for j in range(n):
            z = np.int64(M[r, j])
            if pre: z = np.int64(S[z ^ np.int64(K0[kr, j])])
            v = np.int64(S[(z + m[j] + np.int64(K1[kr, j])) & 0xFFFF])
            M[r, j] = v; prev[j] = v
    return M

@njit(cache=True)
def undiffuse(M, order, krow, K0, K1, Si, IV, pre):
    R, n = M.shape
    prev = np.empty(n, dtype=np.int64); m = np.empty(n, dtype=np.int64)
    for k in range(order.size - 1, -1, -1):
        r = order[k]; kr = krow[k]; s = 1 << (k % LOG2L)
        if k == 0:
            for j in range(n): prev[j] = IV[j]
        else:
            rp = order[k - 1]
            for j in range(n): prev[j] = M[rp, j]
        _mix(prev, s, m)
        for j in range(n):
            z = (np.int64(Si[np.int64(M[r, j])]) - m[j] - np.int64(K1[kr, j])) & 0xFFFF
            if pre: z = np.int64(Si[z]) ^ np.int64(K0[kr, j])
            M[r, j] = z
    return M

def schedules(R, wrap=True):
    W = LOG2L if (wrap and R > LOG2L) else 0
    f = np.concatenate((np.arange(R), np.arange(W))).astype(np.int64)
    fk = np.concatenate((np.arange(R), R - 1 - np.arange(W))).astype(np.int64)
    b = np.concatenate((np.arange(R - 1, -1, -1), np.arange(R - 1, R - 1 - W, -1))).astype(np.int64)
    bk = np.concatenate((np.arange(R - 1, -1, -1), np.arange(W))).astype(np.int64)
    return [(f, fk), (b, bk), (f, fk)]

@njit(cache=True)
def fisher_yates(words, n):
    p = np.arange(n)
    for i in range(n - 1, 0, -1):
        w = (np.uint64(words[2 * i]) << np.uint64(16)) | np.uint64(words[2 * i + 1])
        j = np.int64((w * np.uint64(i + 1)) >> np.uint64(32))
        t = p[i]; p[i] = p[j]; p[j] = t
    return p

# ---------------------------------------------------------------- key schedule
def _u(b):
    return (int.from_bytes(b, "little") >> 11) / float(1 << 53)

def key_schedule(key, nonce, n_samples, domain=b""):
    assert len(key) == 32 and len(nonce) == 16
    need = 24 + 16 * L + 4 * L
    d = hashlib.shake_256(TAG + domain + key + nonce + n_samples.to_bytes(8, "little")).digest(need)
    a = 0.5 + 7.5 * _u(d[0:8]); b = 10.0 + 90.0 * _u(d[8:16]); eps = 0.05 + 0.10 * _u(d[16:24])
    o = 24
    X = np.array([_u(d[o + 8 * i:o + 8 * i + 8]) for i in range(L)]); o += 8 * L
    Y = np.array([_u(d[o + 8 * i:o + 8 * i + 8]) for i in range(L)]); o += 8 * L
    iv = np.frombuffer(d[o:o + 4 * L], dtype=np.uint16).copy()
    return a, b, eps, X, Y, iv[:L], iv[L:]

def padded_len(n):
    return max(-(-n // L), MIN_ROWS) * L

def _streams(key, nonce, n, with_sbox=True, domain=b""):
    Np = padded_len(n); R = Np // L
    a, b, eps, X, Y, iv1, iv2 = key_schedule(key, nonce, n, domain)
    pre = SBOX_WORDS if with_sbox else 0
    ks = lattice_keystream(X, Y, a, b, eps, pre + (2 + 2 * NPASS) * Np)
    S = Si = None
    if with_sbox:
        S, Si = build_sbox(ks[:SBOX_WORDS], GF_INV)
    perm = fisher_yates(ks[pre:pre + 2 * Np], Np)
    K = ks[pre + 2 * Np: pre + (2 + 2 * NPASS) * Np].reshape(2 * NPASS, R, L)
    return Np, R, S, Si, perm, K, iv1, iv2

def _enc_core(x, Np, R, S, perm, K, iv1, iv2, npass=NPASS, pre=True, use_perm=True, wrap=True):
    buf = np.zeros(Np, dtype=np.uint16); buf[:x.size] = x
    if use_perm: buf = buf[perm]
    M = buf.reshape(R, L).copy(); ivs = (iv1, iv2, iv1 ^ iv2)
    for p, (o, ok) in enumerate(schedules(R, wrap)[:npass]):
        diffuse(M, o, ok, K[2 * p], K[2 * p + 1], S, ivs[p], pre)
    return M.reshape(-1)

def _dec_core(c, n, Np, R, Si, perm, K, iv1, iv2, npass=NPASS, pre=True, use_perm=True, wrap=True):
    M = c.astype(np.uint16).reshape(R, L).copy(); ivs = (iv1, iv2, iv1 ^ iv2)
    sch = schedules(R, wrap)[:npass]
    for p in range(npass - 1, -1, -1):
        o, ok = sch[p]; undiffuse(M, o, ok, K[2 * p], K[2 * p + 1], Si, ivs[p], pre)
    if use_perm:
        buf = np.empty(Np, dtype=np.uint16); buf[perm] = M.reshape(-1)
    else:
        buf = M.reshape(-1)
    return buf[:n].view(np.int16).copy()

def to_u16(samples):
    return (np.asarray(samples).astype(np.int32) & 0xFFFF).astype(np.uint16)

# ---------------------------------------------------------------- API (whole-file mode)
RANDOM_SBOX = False   # ablation switch: v1-style S-box (rank of 65536 keystream words)

def encrypt(samples, key, nonce, npass=NPASS, pre=True, use_sbox=True, use_perm=True, wrap=True, sbox=None):
    x = to_u16(samples); n = x.size
    Np, R, S, Si, perm, K, iv1, iv2 = _streams(key, nonce, n)
    if sbox is not None: S = sbox
    if not use_sbox: S = np.arange(65536, dtype=np.uint16)
    return _enc_core(x, Np, R, S, perm, K, iv1, iv2, npass, pre, use_perm, wrap)

def decrypt(cipher, key, nonce, n, npass=NPASS, pre=True, use_perm=True, wrap=True, sbox=None):
    Np, R, S, Si, perm, K, iv1, iv2 = _streams(key, nonce, n)
    if sbox is not None:
        Si = np.empty(65536, dtype=np.uint16); Si[sbox] = np.arange(65536, dtype=np.uint16)
    return _dec_core(cipher, n, Np, R, Si, perm, K, iv1, iv2, npass, pre, use_perm, wrap)

# ---------------------------------------------------------------- framed mode
def _frame_nonce(nonce, i):
    return hashlib.shake_256(b"frame|" + nonce + i.to_bytes(8, "little")).digest(16)

def _file_sbox(key, nonce, n):
    a, b, eps, X, Y, _, _ = key_schedule(key, nonce, n, b"sbox|")
    return build_sbox(lattice_keystream(X, Y, a, b, eps, SBOX_WORDS), GF_INV)

def encrypt_framed(samples, key, nonce, F):
    x = to_u16(samples); n = x.size; S, Si = _file_sbox(key, nonce, n); out = []
    for i, s0 in enumerate(range(0, n, F)):
        xf = x[s0:s0 + F]
        Np, R, _, _, perm, K, iv1, iv2 = _streams(key, _frame_nonce(nonce, i), xf.size, with_sbox=False)
        out.append(_enc_core(xf, Np, R, S, perm, K, iv1, iv2))
    return out

def decrypt_framed(frames, key, nonce, n, F):
    S, Si = _file_sbox(key, nonce, n); out = []
    for i, c in enumerate(frames):
        nf = min(F, n - i * F)
        Np, R, _, _, perm, K, iv1, iv2 = _streams(key, _frame_nonce(nonce, i), nf, with_sbox=False)
        out.append(_dec_core(c, nf, Np, R, Si, perm, K, iv1, iv2))
    return np.concatenate(out)
