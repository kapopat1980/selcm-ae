"""Encrypt the same random audio with the Python reference and the C build, and check the outputs are bit-identical.
Usage (from the repository root, after `make -C c`):  python scripts/check_c_vs_python.py [c/selcm2]"""
import sys, os, subprocess, tempfile, hashlib, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import selcm2
exe = sys.argv[1] if len(sys.argv) > 1 else os.path.join('c', 'selcm2')
rng = np.random.default_rng(7); key = hashlib.sha256(b'check').digest(); nonce = hashlib.md5(b'check').digest()
with tempfile.TemporaryDirectory() as d:
    ok = True
    for n in (1000, 44100, 300000):
        x = rng.integers(-32768, 32768, n).astype(np.int16)
        p = lambda f: os.path.join(d, f)
        x.tofile(p('in.raw')); open(p('k.bin'), 'wb').write(key); open(p('n.bin'), 'wb').write(nonce)
        subprocess.run([exe, p('in.raw'), p('k.bin'), p('n.bin'), p('out.raw')], check=True)
        c_out = np.fromfile(p('out.raw'), dtype=np.uint16)
        py_out = selcm2.encrypt(x, key, nonce).astype(np.uint16)
        same = c_out.size == py_out.size and np.array_equal(c_out, py_out)
        subprocess.run([exe, '-d', p('out.raw'), p('k.bin'), p('n.bin'), str(n), p('dec.raw')], check=True)
        back = np.array_equal(np.fromfile(p('dec.raw'), dtype=np.int16), x)
        print(f'N = {n:6d}: C == Python {same}; C decrypts {back}'); ok &= same and back
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
