"""Figures for SELCM-AE v2: dynamics (Fig. 2), signal statistics (Fig. 4), spectrograms and loss behaviour (Fig. 5)."""
from paths import DATA, RES, FIG
import json, hashlib, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import dyn2, dyn, selcm2, baselines as B
plt.rcParams.update({'font.size': 8, 'font.family': 'Liberation Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.6, 'lines.linewidth': 0.9, 'axes.spines.top': False, 'axes.spines.right': False})
grey = '#555'; blue = '#2f6fb3'; light = '#9aa0a6'
kf = FuncFormatter(lambda v, _: f'{v/1000:.0f}k' if v else '0')
# ---------- Fig. 2 dynamics
Bz = np.load(RES + '/bif2.npz'); D2 = json.load(open(RES + '/dyn2.json'))
fig, ax = plt.subplots(2, 3, figsize=(6.85, 4.4)); b = Bz['bs']; a = Bz['as_']
ax[0, 0].plot(np.repeat(b, Bz['bif_b'].shape[1]), Bz['bif_b'].ravel(), '.', ms=1.0, mew=0, color='#333', alpha=0.5, rasterized=True); ax[0, 0].set_xlabel('b  (a = 2)'); ax[0, 0].set_ylabel('x'); ax[0, 0].set_title('(a) Bifurcation diagram vs b')
ax[0, 1].plot(np.repeat(a, Bz['bif_a'].shape[1]), Bz['bif_a'].ravel(), '.', ms=1.0, mew=0, color='#333', alpha=0.5, rasterized=True); ax[0, 1].set_xlabel('a  (b = 20)'); ax[0, 1].set_ylabel('x'); ax[0, 1].set_title('(b) Bifurcation diagram vs a')
for axx, xs, le, lab, span, txt in [(ax[1, 0], b, Bz['le_b'], 'b  (a = 2)', (10, 100), 'key range $b \\in [10, 100]$'), (ax[1, 1], a, Bz['le_a'], 'a  (b = 20)', (0.5, 8), 'key range $a \\in [0.5, 8]$')]:
    axx.plot(xs, le[:, 0], color=light, label='λ1'); axx.plot(xs, le[:, 1], color=blue, label='λ2'); axx.axhline(0, color='k', lw=0.6); axx.axvspan(*span, color=blue, alpha=0.06)
    axx.set_xlabel(lab); axx.set_ylabel('Lyapunov exponent'); axx.legend(frameon=False); axx.text(np.mean(span), 0.35, txt, fontsize=8, ha='center', color=blue)
ax[1, 0].set_title('(d) Lyapunov exponents vs b'); ax[1, 1].set_title('(e) Lyapunov exponents vs a')
xs, ys = dyn2.orbit(2.0, 20.0, 0.31, 0.72, 30000, 1000); hx, hy = dyn.orbit_henon(0.1, 0.1, 30000, 1000)
ax[0, 2].plot(xs, ys, '.', ms=1.0, mew=0, color=blue, alpha=0.5, rasterized=True); ax[0, 2].set_xlim(0, 1); ax[0, 2].set_ylim(0, 1); ax[0, 2].set_aspect('equal'); ax[0, 2].set_xlabel('x'); ax[0, 2].set_ylabel('y'); ax[0, 2].set_title('(c) Phase portrait, a = 2, b = 20')
ins = ax[0, 2].inset_axes([0.62, 0.62, 0.36, 0.36]); ins.plot(hx, hy, '.', ms=0.4, mew=0, color=grey, rasterized=True); ins.set_xticks([]); ins.set_yticks([]); ins.text(0.5, 0.04, 'Hénon (inset)', transform=ins.transAxes, fontsize=8, ha='center')
S = D2['sensitivity']; n = np.arange(1, 61)
ax[1, 2].semilogy(n, np.maximum(S['selcm_diff'], 1e-17), color=blue, label='2D-SELCM'); ax[1, 2].semilogy(n, np.maximum(S['logistic_diff'], 1e-17), color=light, label='Logistic, μ = 4')
ax[1, 2].set_xlabel('iteration'); ax[1, 2].set_ylabel('|Δx|'); ax[1, 2].set_title('(f) Divergence from $\\Delta x_0 = 10^{-15}$'); ax[1, 2].legend(frameon=False, loc='lower right')
fig.tight_layout(); [fig.savefig(f'{FIG}/Fig2.{e}', dpi=600) for e in ('pdf',)]; [fig.savefig(f'{FIG}/Fig{n}.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'}) for n in [2]]; plt.close(fig)
# ---------- Fig. 4 signal statistics (A1 speech and a stereo music segment)
Z = np.load(DATA + '/dataset2.npz'); key = hashlib.sha256(b"v2 key").digest(); nonce = hashlib.md5(b"v2 nonce").digest()
fig, ax = plt.subplots(3, 3, figsize=(6.85, 5.6)); x = Z['A1 LibriSpeech 198-209-0000']; sr = 16000
c = selcm2.encrypt(x, key, nonce); p = selcm2.decrypt(c, key, nonce, x.size); cs = c[:x.size].view(np.int16); t = np.arange(x.size) / sr
for i, (name, s, col) in enumerate([('Plaintext speech', x, grey), ('Ciphertext (SELCM-AE v2)', cs, blue), ('Decrypted (bit-exact)', p, grey)]):
    ax[i, 0].plot(t, s, lw=0.3, color=col, rasterized=True); ax[i, 0].set_xlim(0, t[-1]); ax[i, 0].set_ylim(-33000, 33000); ax[i, 0].set_ylabel(name, fontsize=9); ax[i, 0].set_xlabel('time (s)' if i == 2 else '')
    ax[i, 1].hist(s, bins=256, color=col); ax[i, 1].set_xlim(-33000, 33000); ax[i, 1].set_yticks([])
    idx = np.random.default_rng(0).integers(0, s.size - 1, 6000)
    ax[i, 2].scatter(s[idx], s[idx + 1], s=0.3, color=col, rasterized=True); ax[i, 2].set_xlim(-33000, 33000); ax[i, 2].set_ylim(-33000, 33000); ax[i, 2].set_aspect('equal')
    r = np.corrcoef(s[:-1].astype(float), s[1:].astype(float))[0, 1]
    ax[i, 2].text(0.03, 0.93, f'r = {r:.4f}', transform=ax[i, 2].transAxes, fontsize=8, bbox=dict(fc='white', ec='none', alpha=0.8))
    for k in range(3):
        ax[i, k].yaxis.set_major_formatter(kf) if k != 1 else None; ax[i, k].xaxis.set_major_formatter(kf) if k else None
ax[0, 0].set_title('Waveform'); ax[0, 1].set_title('Histogram (256 bins)'); ax[0, 2].set_title('Adjacent samples s(n) vs s(n+1)')
fig.tight_layout(); [fig.savefig(f'{FIG}/Fig4.{e}', dpi=600) for e in ('pdf',)]; [fig.savefig(f'{FIG}/Fig{n}.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'}) for n in [4]]; plt.close(fig)
# ---------- Fig. 5 spectrograms, wrong key, crop in whole-file / framed / AES
x = Z['AudioMNIST spk01']; sr = 48000
c = selcm2.encrypt(x, key, nonce); d = selcm2.decrypt(c, key, nonce, x.size)
k2 = bytearray(key); k2[0] ^= 1; w = selcm2.decrypt(c, bytes(k2), nonce, x.size)
def crop(flat): cc = flat.copy(); s0 = cc.size // 3; cc[s0:s0 + int(0.05 * cc.size)] = 0; return cc
dc = selcm2.decrypt(crop(c), key, nonce, x.size)
fr = selcm2.encrypt_framed(x, key, nonce, 16384); sizes = [f.size for f in fr]; cf = crop(np.concatenate(fr))
dfr = selcm2.decrypt_framed(np.split(cf, np.cumsum(sizes)[:-1]), key, nonce, x.size, 16384)
ca = B.b4_encrypt(x, key, nonce); da = B.b4_decrypt(crop(ca), key, nonce, x.size)
fig, ax = plt.subplots(2, 3, figsize=(6.85, 4.0)); t = np.arange(x.size) / sr
kw = dict(NFFT=1024, Fs=sr, noverlap=768, cmap='Greys', vmin=-40, vmax=60)
for i, (name, s) in enumerate([('Plaintext', x), ('SELCM-AE v2 ciphertext', c[:x.size].view(np.int16)), ('Decrypted, 1-bit-wrong key', w)]):
    ax[0, i].specgram(s.astype(float), **kw); ax[0, i].set_title(f'({"abc"[i]}) {name}'); ax[0, i].set_xlabel('time (s)'); ax[0, i].set_ylabel('frequency (Hz)' if i == 0 else ''); ax[0, i].yaxis.set_major_formatter(kf)
for i, (name, s) in enumerate([('Whole-file mode', dc), ('Framed mode', dfr), ('AES-256-CTR', da)]):
    ax[1, i].plot(t, s, lw=0.3, color=blue if i == 1 else grey, rasterized=True); ax[1, i].set_ylim(-33000, 33000); ax[1, i].set_title(f'({"def"[i]}) 5% crop: {name}'); ax[1, i].set_xlabel('time (s)'); ax[1, i].yaxis.set_major_formatter(kf)
    ax[1, i].text(0.02, 0.92, f'{(s != x).mean()*100:.1f}% of samples lost', transform=ax[1, i].transAxes, fontsize=8, bbox=dict(fc='white', ec='none', alpha=0.85))
fig.tight_layout(); [fig.savefig(f'{FIG}/Fig5.{e}', dpi=600) for e in ('pdf',)]; [fig.savefig(f'{FIG}/Fig{n}.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'}) for n in [5]]; plt.close(fig); print('figures ok')
