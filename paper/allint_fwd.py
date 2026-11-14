#!/usr/bin/env python3
"""Table 2 forward error (prior only / two anchors) over every core count 2..16, anchors excluded (416 configurations).
   python3 allint_fwd.py   路径同 paper_numbers_v2.py 的 fwd.v2.*(grid 模式应复现 28.8/21.9 与 6.5/87.5)。"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))
from paths import DATA, RESULTS

import json, os, sys, statistics as st, numpy as np
import paper_numbers_v2 as pn, eval_parts, score_rc
from sql_model2 import SQLPredictor
from phoebe_predictor import effective_cores
D = DATA; TOL = 0.15; R = (2, 4, 6, 8, 11, 13, 16)
TRAIN = eval_parts.build()
def og(wl, s, th, ag):
    d = json.load(open(f"{D}/testv2og/{wl}/s{s}_l4_t{th}_a{ag}_b0/point.json")); assert d["steps"] == d["passes"] * d["questions"]
    return d["completion_seconds"]
out = {}
for anc in (False, True):
    for mode in ("grid", "all"):
        ew = []
        for wl in ("test", "test2"):
            M, cost, steps, pts = pn.fitted(wl, TRAIN, SQLPredictor, anc)
            combos = {(th, ag) for (s, th, ag) in pts}
            for (s, th, ag), p in pts.items():
                if (s, th, ag) in pn.ANC: continue
                ew.append(abs(score_rc.to_wall(M.predict(cost, steps, s, th, ag), p["ag"], p["C"]) / p["wall"] - 1))
            if mode == "all":
                for th, ag in combos:
                    for s in range(2, 17):
                        if s in R: continue
                        ew.append(abs(score_rc.to_wall(M.predict(cost, steps, s, th, ag), ag, effective_cores(s, th, ag)) / og(wl, s, th, ag) - 1))
        out[f"{'2anchor' if anc else 'prior'}.{mode}"] = dict(n=len(ew), median=round(100 * st.median(ew), 1), within15=round(100 * float(np.mean(np.array(ew) <= TOL)), 1))
print(json.dumps(out, indent=1)); json.dump(out, open(f"{RESULTS}/allint_fwd.json", "w"), indent=1)
