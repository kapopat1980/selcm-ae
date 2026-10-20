"""Quick self-test: round trip, determinism, key sensitivity and framed mode for SELCM-AE v2."""
import hashlib, numpy as np, selcm2
rng = np.random.default_rng(1); key = hashlib.sha256(b"k").digest(); nonce = bytes(16)
for n in (1, 255, 4096, 100_000):
    x = rng.integers(-32768, 32768, n).astype(np.int16)
    c = selcm2.encrypt(x, key, nonce)
    assert np.array_equal(selcm2.decrypt(c, key, nonce, n), x)
    assert np.array_equal(selcm2.encrypt(x, key, nonce), c)
    k2 = bytearray(key); k2[0] ^= 1
    assert (selcm2.encrypt(x, bytes(k2), nonce) != c).mean() > 0.999
fr = selcm2.encrypt_framed(x, key, nonce, 16384)
assert np.array_equal(selcm2.decrypt_framed(fr, key, nonce, x.size, 16384), x)
print("SELCM-AE v2 self-test passed")
