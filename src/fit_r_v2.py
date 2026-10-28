#!/usr/bin/env python3
"""用 rc3+rc4 重拟 db_time -> 墙钟 的换算: r = db_time / (并发 x 墙钟) = exp(a·log ag + b·log C + c)。
与 r_model.json(老网格拟的)同式, 只换数据。每格点取重复均值。"""
import json, glob, os, re, collections, statistics as st, math
import numpy as np, badpoints
from paths import DATA
from phoebe_predictor import effective_cores
NOM = badpoints.nominal_steps(); G = collections.defaultdict(list)
for run in ("rc3", "rc4"):
    for p in glob.glob(f"{DATA}/{run}/pdb_*/*/point.json"):
        lib, tag = p.split(os.sep)[-3], p.split(os.sep)[-2]
        m = re.match(r"s(\d+)_l4_t2_a(\d+)_b0_x\d+_k\d+$", tag)
        if not m: continue
        d = json.load(open(p))
        if d.get("steps", 0) != NOM.get(lib, -1) or not d.get("completion_seconds"): continue
        s, a = int(m[1]), int(m[2])
        G[(lib, s, a)].append(d["db_time_s"] / (a * d["completion_seconds"]))
X, y = [], []
for (lib, s, a), v in G.items():
    X.append([math.log(a), math.log(effective_cores(s, 2, a)), 1.0]); y.append(math.log(st.mean(v)))
b = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)[0]
json.dump({"a": float(b[0]), "b": float(b[1]), "c": float(b[2]), "source": "rc3+rc4", "n": len(y),
           "cmd": "python3 fit_r_v2.py"}, open(f"{DATA}/r_model_v2.json", "w"), indent=1)
old = json.load(open(f"{DATA}/r_model.json"))
print(f"老 r_model : a={old['a']:+.4f} b={old['b']:+.4f} c={old['c']:+.4f}")
print(f"新 r_model : a={b[0]:+.4f} b={b[1]:+.4f} c={b[2]:+.4f}   ({len(y)} 个格点)")
