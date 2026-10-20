# Data

The audio is downloaded from its original public sources. It is not stored in this repository.

```bash
bash scripts/fetch_data.sh        # from the repository root
cd src && python build_ds2.py     # writes dataset2.npz, meta2.json, set24.npz, setogg.npz here
```

- `esc50_files.txt` lists the 50 ESC-50 clips (one per class).
- `librosa_files.txt` lists the 14 librosa example recordings.

AudioMNIST uses speakers 01–40, digits 0–9, take 0. The Free Spoken Digit Dataset uses six speakers, with takes 0 and 1.
