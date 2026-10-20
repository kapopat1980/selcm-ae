#!/usr/bin/env bash
# Download exactly the public audio files used in the paper into data/ (sparse git checkouts; about 120 MB).
# Usage: bash scripts/fetch_data.sh        (run from the repository root)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; DATA="$ROOT/data"; TMP="$DATA/_downloads"
mkdir -p "$TMP" "$DATA/amnist" "$DATA/fsdd" "$DATA/esc" "$DATA/librosa"
sparse() {  # sparse <repo-url> <dir> <pattern>...
  local url=$1 dir=$2; shift 2
  [ -d "$TMP/$dir" ] || git clone -q --depth 1 --filter=blob:none --sparse "$url" "$TMP/$dir"
  git -C "$TMP/$dir" sparse-checkout set --no-cone "$@"
}
# AudioMNIST (MIT licence): speakers 01-40, digits 0-9, take 0
P=(); for s in $(seq -w 1 40); do P+=("/data/$s/*_${s}_0.wav"); done
sparse https://github.com/soerenab/AudioMNIST.git audiomnist "${P[@]}"
for s in $(seq -w 1 40); do cp "$TMP"/audiomnist/data/$s/*_${s}_0.wav "$DATA/amnist/"; done
# Free Spoken Digit Dataset (CC BY-SA 4.0): six speakers, takes 0 and 1
P=(); for sp in jackson nicolas theo yweweler george lucas; do for i in 0 1; do P+=("/recordings/*_${sp}_${i}.wav"); done; done
sparse https://github.com/Jakobovski/free-spoken-digit-dataset.git fsdd "${P[@]}"
cp "$TMP"/fsdd/recordings/*.wav "$DATA/fsdd/"
# ESC-50 (CC BY-NC 3.0): one clip per class, listed in data/esc50_files.txt
P=(); while read -r f; do P+=("/audio/$f"); done < "$DATA/esc50_files.txt"
sparse https://github.com/karolpiczak/ESC-50.git esc50 "${P[@]}"
while read -r f; do cp "$TMP/esc50/audio/$f" "$DATA/esc/"; done < "$DATA/esc50_files.txt"
# librosa example data (LibriSpeech excerpts, Creative Commons music and bioacoustics)
P=(); while read -r f; do P+=("/audio/$f"); done < "$DATA/librosa_files.txt"
sparse https://github.com/librosa/data.git librosa "${P[@]}"
while read -r f; do cp "$TMP/librosa/audio/$f" "$DATA/librosa/"; done < "$DATA/librosa_files.txt"
echo "Downloaded: $(ls "$DATA/amnist" | wc -l) AudioMNIST, $(ls "$DATA/fsdd" | wc -l) FSDD, $(ls "$DATA/esc" | wc -l) ESC-50, $(ls "$DATA/librosa" | wc -l) librosa files"
echo "Next: cd src && python build_ds2.py"
