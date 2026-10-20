"""Subset of NIST SP 800-22 rev1a statistical tests (alpha = 0.01)."""
import numpy as np
from scipy.special import erfc, gammaincc
from scipy.stats import norm

def frequency(e):
    n = e.size; s = np.abs(np.sum(2 * e.astype(np.int64) - 1)) / np.sqrt(n)
    return erfc(s / np.sqrt(2))

def block_frequency(e, M=128):
    N = e.size // M; pi = e[:N * M].reshape(N, M).mean(1)
    return gammaincc(N / 2, 4 * M * np.sum((pi - 0.5) ** 2) / 2)

def runs(e):
    n = e.size; pi = e.mean()
    if abs(pi - 0.5) >= 2 / np.sqrt(n): return 0.0
    v = 1 + np.count_nonzero(e[1:] != e[:-1])
    return erfc(abs(v - 2 * n * pi * (1 - pi)) / (2 * np.sqrt(2 * n) * pi * (1 - pi)))

def longest_run(e):
    M = 10000; K = 6; N = e.size // M
    v = [11, 12, 13, 14, 15, 16]; p = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
    blocks = e[:N * M].reshape(N, M); cnt = np.zeros(7)
    for b in blocks:
        d = np.diff(np.concatenate(([0], b, [0])).astype(np.int8))
        st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
        lr = (en - st).max() if st.size else 0
        cnt[0 if lr <= 10 else (6 if lr >= 16 else lr - 10)] += 1
    chi = np.sum((cnt - N * np.array(p)) ** 2 / (N * np.array(p)))
    return gammaincc(K / 2, chi / 2)

def dft(e):
    n = e.size; X = 2 * e.astype(float) - 1; S = np.abs(np.fft.rfft(X))[: n // 2]
    T = np.sqrt(np.log(1 / 0.05) * n); N0 = 0.95 * n / 2; N1 = np.count_nonzero(S < T)
    return erfc(abs((N1 - N0) / np.sqrt(n * 0.95 * 0.05 / 4)) / np.sqrt(2))

def _psi(e, m):
    if m == 0: return 0.0
    n = e.size; ext = np.concatenate((e, e[: m - 1])).astype(np.int64)
    idx = np.zeros(n, dtype=np.int64)
    for k in range(m): idx = (idx << 1) | ext[k:k + n]
    c = np.bincount(idx, minlength=2 ** m)
    return (2 ** m / n) * np.sum(c.astype(float) ** 2) - n

def serial(e, m=16):
    p0, p1, p2 = _psi(e, m), _psi(e, m - 1), _psi(e, m - 2)
    return min(gammaincc(2 ** (m - 2), (p0 - p1) / 2), gammaincc(2 ** (m - 3), (p0 - 2 * p1 + p2) / 2))

def approx_entropy(e, m=10):
    n = e.size
    def phi(mm):
        ext = np.concatenate((e, e[: mm - 1])).astype(np.int64); idx = np.zeros(n, dtype=np.int64)
        for k in range(mm): idx = (idx << 1) | ext[k:k + n]
        c = np.bincount(idx, minlength=2 ** mm) / n; c = c[c > 0]
        return np.sum(c * np.log(c))
    ap = phi(m) - phi(m + 1)
    return gammaincc(2 ** (m - 1), 2 * n * (np.log(2) - ap) / 2)

def cusum(e):
    n = e.size; S = np.cumsum(2 * e.astype(np.int64) - 1); z = np.max(np.abs(S))
    k1 = np.arange(int((-n / z + 1) // 4), int((n / z - 1) // 4) + 1)
    k2 = np.arange(int((-n / z - 3) // 4), int((n / z - 1) // 4) + 1)
    p = 1 - np.sum(norm.cdf((4 * k1 + 1) * z / np.sqrt(n)) - norm.cdf((4 * k1 - 1) * z / np.sqrt(n)))
    p += np.sum(norm.cdf((4 * k2 + 3) * z / np.sqrt(n)) - norm.cdf((4 * k2 + 1) * z / np.sqrt(n)))
    return p

TESTS = {"Frequency": frequency, "Block frequency": block_frequency, "Runs": runs,
         "Longest run": longest_run, "DFT spectral": dft, "Approximate entropy": approx_entropy,
         "Serial": serial, "Cumulative sums": cusum}

def battery(seqs):
    """seqs: list of uint8 0/1 arrays. Returns {test: (pass proportion, uniformity p-value)}."""
    from scipy.stats import chisquare
    out = {}
    for name, f in TESTS.items():
        ps = np.array([f(s) for s in seqs])
        hist = np.histogram(ps, bins=10, range=(0, 1))[0]
        out[name] = (float(np.mean(ps >= 0.01)), float(chisquare(hist).pvalue))
    return out
