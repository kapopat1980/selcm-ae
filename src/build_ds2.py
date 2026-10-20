"""Corpus v2: speech, environmental/bioacoustic, stereo music segments; plus 24-bit and compressed-file sets."""
from paths import DATA, RES, FIG
import numpy as np, soundfile as sf, glob, os, json
D = DATA; A = DATA + '/librosa/'
out = {}; meta = {}
def rd(f, dtype='int16'):
    x, sr = sf.read(f, dtype=dtype); return x, sr
def put(name, cat, x, sr, ch=1, src=''):
    out[name] = np.ascontiguousarray(x.reshape(-1)); meta[name] = {'cat': cat, 'sr': int(sr), 'ch': ch, 'n': int(x.size), 'src': src}
# speech
for s in range(1, 41):
    parts = []
    for d in range(10): x, sr = rd(f'{D}/amnist/{d}_{s:02d}_0.wav'); parts.append(x)
    put(f'AudioMNIST spk{s:02d}', 'speech', np.concatenate(parts), sr, src='AudioMNIST')
for sp in ['jackson', 'nicolas', 'theo', 'yweweler', 'george', 'lucas']:
    for i in (0, 1):
        parts = []
        for d in range(10): x, sr = rd(f'{D}/fsdd/{d}_{sp}_{i}.wav'); parts.append(x)
        put(f'FSDD {sp} take{i}', 'speech', np.concatenate(parts), sr, src='FSDD')
for f, n in [('198-209-0000', 'A1 LibriSpeech 198-209-0000'), ('5703-47212-0000', 'LibriSpeech 5703-47212-0000'), ('3436-172162-0000', 'LibriSpeech 3436-172162-0000')]:
    x, sr = rd(A + f + '.hq.ogg'); put(n, 'speech', x, sr, src='LibriSpeech (librosa)')
# environmental / bioacoustic
for f in sorted(glob.glob(f'{D}/esc/*.wav')):
    x, sr = rd(f); x = x[:, 0] if x.ndim == 2 else x; put('ESC-50 ' + os.path.basename(f), 'environmental', x, sr, src='ESC-50')
for f, n in [('glacier-bay-humpback', 'Humpback whale song'), ('456440__inspectorj__bird-whistling-robin-single-13', 'Robin whistle'), ('pibble', 'Dog vocalisation (pibble)')]:
    x, sr = rd(A + f + '.hq.ogg'); x = x[:, 0] if x.ndim == 2 else x; put(n, 'environmental', x, sr, src='librosa')
# music: 10-s stereo segments, channels interleaved L,R,L,R,...
tracks = [('Kevin_MacLeod_-_Vibe_Ace', 'Vibe Ace'), ('147793__setuniman__sweet-waltz-0i-22mi', 'Sweet waltz'),
          ('sorohanro_-_solo-trumpet-06', 'Solo trumpet'), ('admiralbob77_-_Choice_-_Drum-bass', 'Choice (drum and bass)'),
          ('Kevin_MacLeod_-_P_I_Tchaikovsky_Dance_of_the_Sugar_Plum_Fairy', 'Sugar Plum Fairy'),
          ('Hungarian_Dance_number_5_-_Allegro_in_F_sharp_minor_(string_orchestra)', 'Hungarian Dance 5'),
          ('Karissa_Hobbs_-_Lets_Go_Fishin', "Let's Go Fishin'"), ('442789__lena-orsa__happy-music-pistachio-ice-cream-ragtime', 'Pistachio ragtime')]
for f, n in tracks:
    x, sr = rd(A + f + '.hq.ogg'); seg = 10 * sr
    starts = range(0, max(1, x.shape[0] - seg + 1), seg) if x.shape[0] >= seg else [0]
    for i, s0 in enumerate(starts):
        put(f'{n} seg{i+1:02d}', 'music', x[s0:s0 + seg], sr, ch=2, src=n)
np.savez_compressed(f'{D}/dataset2.npz', **out); json.dump(meta, open(f'{D}/meta2.json', 'w'), indent=1)
from collections import Counter
c = Counter(v['cat'] for v in meta.values()); print(c, len(meta), 'samples', sum(v['n'] for v in meta.values()),
      'minutes', round(sum(v['n'] / v['sr'] / v['ch'] for v in meta.values()) / 60, 1))
# 24-bit set: decode music/speech to float, quantise to 24-bit, pack as little-endian 3-byte samples -> 16-bit words
x24 = {}
for f, n in tracks[:4] + [('198-209-0000', 'A1 LibriSpeech')]:
    x, sr = sf.read(A + f + '.hq.ogg', dtype='float64'); x = x[:10 * sr]; x = x.reshape(-1)
    q = np.clip(np.round(x * 8388607), -8388608, 8388607).astype(np.int32)
    b = np.stack([q & 255, (q >> 8) & 255, (q >> 16) & 255], 1).astype(np.uint8).reshape(-1)
    if b.size % 2: b = np.concatenate([b, [0]]).astype(np.uint8)
    x24[n] = b.view(np.int16)
np.savez_compressed(f'{D}/set24.npz', **x24)
# compressed files: the .ogg bytes themselves
xc = {}
for f, n in tracks[:4] + [('198-209-0000', 'A1 LibriSpeech')]:
    b = np.frombuffer(open(A + f + '.hq.ogg', 'rb').read(), dtype=np.uint8)
    if b.size % 2: b = np.concatenate([b, [0]]).astype(np.uint8)
    xc[n] = b.view(np.int16)
np.savez_compressed(f'{D}/setogg.npz', **xc); print('24-bit', {k: v.size for k, v in x24.items()}, 'ogg', {k: v.size for k, v in xc.items()})
