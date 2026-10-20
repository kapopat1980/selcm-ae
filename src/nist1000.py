"""NIST SP 800-22 on 1000 sequences x 10^6 bits; stores every per-sequence p-value. Serial split into its two sub-tests."""
from paths import DATA, RES, FIG
import sys, numpy as np, hashlib
import nist_full as F, serial_fix_lib as SF, selcm2
which = sys.argv[1]; NSEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
NB = 1_000_000; nw = NB // 16 + 1; bits = lambda w: np.unpackbits(w.view(np.uint8))[:NB]
def g_prop(i):
    k = hashlib.sha256(b"nist1k%d" % i).digest(); a, b, eps, X, Y, _, _ = selcm2.key_schedule(k, bytes(16), 1000); return selcm2.lattice_keystream(X, Y, a, b, eps, nw)
def g_cipher(i): return selcm2.encrypt(np.zeros(nw, np.int16), hashlib.sha256(b"ct1k").digest(), hashlib.md5(b"1k%d" % i).digest())[:nw]
g = {'ks': g_prop, 'ct': g_cipher}[which]
single = {k: [] for k in F.SINGLE if k != 'Serial'}; single['Serial-1'] = []; single['Serial-2'] = []; multi = {k: [] for k in F.MULTI}
for i in range(NSEQ):
    s = bits(g(i))
    for k, f in F.SINGLE.items():
        if k == 'Serial': a, b = SF.serial2(s); single['Serial-1'].append(a); single['Serial-2'].append(b)
        else: single[k].append(f(s))
    for k, f in F.MULTI.items():
        r = f(s)
        if r is not None: multi[k].append(r)
    if i % 50 == 49:
        np.savez(f'{RES}/nist1k_{which}.npz', **{'S|' + k: np.array(v) for k, v in single.items()}, **{'M|' + k: np.array(v) for k, v in multi.items()}, done=i + 1)
        print(which, i + 1, flush=True)
