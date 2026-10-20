"""Row-level MILP bounds for the ablation variants of Table 11 (R = 64 rows). Writes results/milp_ablation.json."""
from paths import RES
import json
from milp_rows import differential, linear
R = 64
V = {"Full SELCM-AE v2": (3, True), "Two passes": (2, True), "No pre-S-box layer": (3, False),
     "One pass, no pre layer": (1, False), "Two passes, no pre (v1 structure)": (2, False)}
out = {}
for name, (npass, pre) in V.items():
    out[name] = {"diff": differential(R, npass, pre)[0], "lin": linear(R, npass, pre)[0]}
    print(name, out[name], flush=True)
json.dump(out, open(RES + "/milp_ablation.json", "w"), indent=1)
