# SELCM-AE, a portable hyperchaotic lattice cipher for audio

Code, data scripts and raw results for the paper:

> L. Patil and K. Popat, "A Portable Hyperchaotic Lattice Cipher for Audio with Provable Bounds on Differential and Linear Trails," submitted to *Nonlinear Dynamics*, 2026.

SELCM-AE is a lossless, nonce-based audio cipher. It has four parts:

- **The 2D-SELCM map:** a new two-dimensional Sine–Exponential–Logistic Cross-Coupled Map whose two Lyapunov exponents are both positive over the whole key range. It is built from fixed polynomials, so every IEEE-754 platform computes the same orbit bit for bit.
- **The keystream:** a 256-lane coupled lattice of these maps.
- **The S-box:** a keyed inversion S-box over GF(2^16).
- **The diffusion:** three passes of cyclic butterfly diffusion.

Mixed-integer linear programming proves that every differential characteristic activates at least 13 S-boxes and every linear trail at least 10.

> **Research code.** SELCM-AE has not had independent cryptanalysis and has no integrity protection. Do not use it to protect real data; use a standard authenticated cipher (AES-GCM, AES-GCM-SIV, ChaCha20-Poly1305) instead.

## Repository layout

| Folder | Contents |
| --- | --- |
| `src/` | Python reference implementation, baselines and every experiment and figure script |
| `c/` | Portable C implementation (SSE2 / AVX2 / AVX-512), TestU01 drivers, `Makefile` |
| `scripts/` | Data download script and the C-versus-Python check |
| `data/` | File lists for the public audio corpus (the audio itself is downloaded, not stored here) |
| `results/` | Raw result files behind every table in the paper |
| `figures/` | Figures 1–5 and the graphical abstract |

## Quick start

```bash
pip install -r requirements.txt
cd src && python selftest.py                     # round trip, determinism, key sensitivity, framed mode
```

Using the cipher from Python:

```python
import hashlib, numpy as np, selcm2
key = hashlib.sha256(b"my key").digest()        # 32 bytes
nonce = bytes(16)                                 # 16 bytes, unique per message
x = np.zeros(48000, np.int16)                     # 16-bit PCM samples (or any bytes viewed as int16)
c = selcm2.encrypt(x, key, nonce)
assert np.array_equal(selcm2.decrypt(c, key, nonce, x.size), x)
```

### C implementation

```bash
make -C c                                         # builds selcm2 (native), selcm2_avx2, selcm2_sse2
python scripts/check_c_vs_python.py               # checks C output == Python output, bit for bit
./c/selcm2 in.raw key32.bin nonce16.bin out.raw 31        # encrypt raw int16; time 31 runs
./c/selcm2 -d out.raw key32.bin nonce16.bin N dec.raw     # decrypt N samples
```

Compile with `-ffp-contract=off`, as the Makefile does. Fused multiply–add changes the floating-point results and breaks bit-identical output across platforms.

## Reproducing the paper

1. **Download the public audio** (about 120 MB, sparse git checkouts):
   ```bash
   bash scripts/fetch_data.sh
   cd src && python build_ds2.py          # builds data/dataset2.npz (156 files), set24.npz, setogg.npz
   ```
   The rebuilt corpus is bit-identical to the one used in the paper.

2. **Run the experiments** (from `src/`; each script writes into `results/`):

| Paper item | Script | Output in `results/` |
| --- | --- | --- |
| Dynamics: Table 2, Table 3, Fig. 2 | `dyn2.py` | `dyn2.json`, `bif2.npz` |
| Trail bounds: Tables 4 and 11 | `milp_rows.py`, `milp_ablation.py`, `milp_lane.py pre 8 12` | `milp_rows.json`, `milp_ablation.json`, `milp_lane_*.json` |
| Statistics, differential, multi-key, attacks, S-box, ablation, robustness, perceptual, payloads, Python speed: Tables 7–15 | `ds_run2.py` | `v2_results.json` |
| Adaptive cancellation attack | `adaptive_attack.py` | `adaptive.json` |
| Error spread of v1, B2, B3 | `err_spread.py` | `err_spread.json` |
| NIST SP 800-22, 1,000 sequences: Table 16 | `nist1000.py ks`, `nist1000.py ct` | `nist1k_*.npz`, `nist1k_summary.json` |
| NIST SP 800-22 baselines, Serial sub-tests | `run_nist_v2.py`, `serial_fix.py` | `nist_full*.json`, `serial_fix.json` |
| 16-bit window test | `window16.py` | printed |
| TestU01 SmallCrush and Crush: Table 17 | `make -C c testu01 TU01=/path/to/testu01`, then `c/tu_selcm <selcm\|logistic\|lorenz\|aes> <small\|crush> 1` | `testu01/*.txt` |
| Savir2 repeat under 20 keys | `c/tu_savir <selcm\|aes> savir2 <seed>` | `testu01/savir2_reps.txt` |
| C speed and portability: Table 15 | `make -C c`; `c/selcm_v1_baselines.c` for v1 and baselines | — |
| Figures | `fig_journal.py` (Figs. 2, 4, 5), `fig_vector.py` (Figs. 1, 3), `graphical_abstract.py` | `figures/` |

Run times on a 2-vCPU machine:
- `ds_run2.py`: about 15 minutes.
- `nist1000.py`: about 1 hour per stream.
- TestU01 Crush: about 30 minutes per generator.
- `dyn2.py`: about 20 minutes.

## Main source files

- `src/selcm2.py`: SELCM-AE v2, the cipher in the paper.
- `src/selcm.py`: SELCM-AE v1, the two-pass predecessor used for comparison.
- `src/baselines.py`: the comparison ciphers.
  - B1: logistic-map XOR.
  - B2: Hénon permutation plus Lorenz XOR.
  - B3: forward chained diffusion.
  - B4: AES-256-CTR.
- `src/nist.py`, `src/nist_full.py`: an implementation of all 15 NIST SP 800-22 tests, validated against `os.urandom`.
- `src/milp_*.py`: MILP models for active S-boxes (SciPy / HiGHS).

## Requirements

Python 3.11 or later, plus the packages in `requirements.txt`. The C build needs GCC or Clang and OpenSSL's libcrypto. The TestU01 targets need TestU01 1.2.3. If `pip install pesq` fails, build it with `pip install --no-build-isolation pesq`; it is only needed for the PESQ scores.

## Data licences

The audio is not redistributed here. `scripts/fetch_data.sh` downloads it from the original sources:
- AudioMNIST (MIT).
- Free Spoken Digit Dataset (CC BY-SA 4.0).
- ESC-50 (CC BY-NC 3.0, non-commercial).
- The librosa example data (LibriSpeech excerpts and Creative Commons music; see each file's licence in that repository).

## Licence

- **Code:** MIT licence (see `LICENSE`).
- **Result files in `results/` and figures in `figures/`:** CC BY 4.0.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).
