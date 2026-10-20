"""Repository paths used by every script (run scripts from the src/ folder or anywhere)."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = str(ROOT / 'data')        # raw downloads and the built corpus (dataset2.npz, meta2.json, set24.npz, setogg.npz)
RES = str(ROOT / 'results')      # raw result files behind every table
FIG = str(ROOT / 'figures')      # figure files
