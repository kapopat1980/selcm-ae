"""Re-implemented representative baseline designs from the audio-encryption literature.

B1  LOG-XOR : 1D logistic-map keystream XOR (classic 1D chaotic stream cipher)
B2  PS-HL   : Henon-map sample permutation + Lorenz-system keystream XOR (permutation-substitution)
B3  CHAIN-F : chaotic keystream + forward-only chained diffusion c_i = S(p_i + c_{i-1} + k_i) ^ k'_i
B4  AES-CTR : AES-256 in counter mode (OpenSSL via `cryptography`)
All chaotic loops are numba-compiled so that timing compares algorithms, not interpreters.
"""
import hashlib, math
import numpy as np
from numba import njit
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

def _seed(key, tag, i):
    d = hashlib.sha256(tag + key + bytes([i])).digest()
    return (int.from_bytes(d[:8], "little") >> 11) / float(1 << 53)

# ---------------------------------------------------------------- B1
@njit(cache=True)
def logistic_ks(x, mu, n):
    out = np.empty(n, dtype=np.uint16)
    for _ in range(200):
        x = mu * x * (1.0 - x)
    for i in range(n):
        x = mu * x * (1.0 - x)
        out[i] = np.uint64(x * 1e14) % np.uint64(65536)
    return out

def b1_encrypt(x, key):
    u = (x.astype(np.int32) & 0xFFFF).astype(np.uint16)
    ks = logistic_ks(0.1 + 0.8 * _seed(key, b"B1", 0), 3.99 + 0.0099 * _seed(key, b"B1", 1), u.size)
    return u ^ ks

def b1_decrypt(c, key, n):
    return b1_encrypt(c.view(np.int16), key).view(np.int16)

# ---------------------------------------------------------------- B2
@njit(cache=True)
def henon_seq(x, y, n):
    out = np.empty(n)
    for _ in range(200):
        x, y = 1.0 - 1.4 * x * x + y, 0.3 * x
    for i in range(n):
        x, y = 1.0 - 1.4 * x * x + y, 0.3 * x
        out[i] = x
    return out

@njit(cache=True)
def lorenz_ks(x, y, z, n):
    h = 0.005; out = np.empty(n, dtype=np.uint16)
    for i in range(n + 500):
        dx = 10.0 * (y - x); dy = x * (28.0 - z) - y; dz = x * y - 8.0 / 3.0 * z
        x += h * dx; y += h * dy; z += h * dz
        if i >= 500:
            out[i - 500] = np.uint64(abs(x) * 1e12) % np.uint64(65536)
    return out

def _b2_streams(key, n):
    perm = np.argsort(henon_seq(0.1 * _seed(key, b"B2", 0), 0.1 * _seed(key, b"B2", 1), n), kind="stable")
    ks = lorenz_ks(1 + _seed(key, b"B2", 2), 1 + _seed(key, b"B2", 3), 1 + _seed(key, b"B2", 4), n)
    return perm, ks

def b2_encrypt(x, key):
    u = (x.astype(np.int32) & 0xFFFF).astype(np.uint16)
    perm, ks = _b2_streams(key, u.size)
    return u[perm] ^ ks

def b2_decrypt(c, key, n):
    perm, ks = _b2_streams(key, n)
    out = np.empty(n, dtype=np.uint16); out[perm] = c ^ ks
    return out.view(np.int16)

# ---------------------------------------------------------------- B3
@njit(cache=True)
def chain_fwd(u, k1, k2, iv):
    c = np.empty_like(u); prev = np.int64(iv)
    for i in range(u.size):
        t = (np.int64(u[i]) + prev + np.int64(k1[i])) & 0xFFFF
        t = ((t << 5) | (t >> 11)) & 0xFFFF
        prev = t ^ np.int64(k2[i]); c[i] = prev
    return c

@njit(cache=True)
def chain_inv(c, k1, k2, iv):
    u = np.empty_like(c); prev = np.int64(iv)
    for i in range(c.size):
        t = np.int64(c[i]) ^ np.int64(k2[i]); t = ((t >> 5) | (t << 11)) & 0xFFFF
        u[i] = (t - prev - np.int64(k1[i])) & 0xFFFF; prev = np.int64(c[i])
    return u

def b3_encrypt(x, key):
    u = (x.astype(np.int32) & 0xFFFF).astype(np.uint16)
    ks = logistic_ks(0.1 + 0.8 * _seed(key, b"B3", 0), 3.99 + 0.0099 * _seed(key, b"B3", 1), 2 * u.size)
    return chain_fwd(u, ks[0::2], ks[1::2], 12345)

def b3_decrypt(c, key, n):
    ks = logistic_ks(0.1 + 0.8 * _seed(key, b"B3", 0), 3.99 + 0.0099 * _seed(key, b"B3", 1), 2 * n)
    return chain_inv(c, ks[0::2], ks[1::2], 12345).view(np.int16)

# ---------------------------------------------------------------- B4
def b4_encrypt(x, key, nonce):
    e = Cipher(algorithms.AES(key), modes.CTR(nonce)).encryptor()
    return np.frombuffer(e.update(x.astype(np.int16).tobytes()) + e.finalize(), dtype=np.uint16).copy()

def b4_decrypt(c, key, nonce, n):
    d = Cipher(algorithms.AES(key), modes.CTR(nonce)).decryptor()
    return np.frombuffer(d.update(c.tobytes()) + d.finalize(), dtype=np.int16).copy()
