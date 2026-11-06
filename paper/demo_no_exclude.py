#!/usr/bin/env python3
"""Result when bal_test question 900086 (1.1 s alone, correct result) is NOT excluded, from the pre-exclusion grid.
不重采: 剔除前的完整网格在 data/testv2_pre900086/test(98 点, 10-15 采, 与现网格同协议, 唯一差别是含 900086),
锚点/离线文件用当时的 data/*_pre900086.json。test2 不含被剔轮次, 原样。
判定逐行照 demo_failmap.py(closest ±15%), 前向误差同 demo_usl_baseline.py(192 点, 去锚点)。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))
from paths import DATA, RESULTS

import io, sys, os, glob, json, statistics as st, importlib.util
from types import SimpleNamespace
import score_rc, eval_testv2
from phoebe_predictor import effective_cores
D = DATA
_g = glob.glob
eval_testv2.glob.glob = lambda p: _g(p.replace("testv2/test/", "testv2_pre900086/test/"))   # 只换 bal_test 的网格目录
sp = importlib.util.spec_from_file_location("cli", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src", "phoebe_cli.py"))
cli = importlib.util.module_from_spec(sp); sp.loader.exec_module(cli)
TOL = 0.15; RUNGS = (2, 4, 6, 8, 11, 13, 16)
R = {}
for wl, nm, suf in (("test", "bal_test", "_pre900086"), ("test2", "test2", "")):
    e = sys.stderr; sys.stderr = io.StringIO()
    try: adv = cli._load(SimpleNamespace(offline=f"{D}/offline_{nm}_v2{suf}.json", anchors=f"{D}/anchors_{nm}_v2{suf}.json",
                                         zero_shot_quantile=0.0, anchor_quantile=None))
    finally: sys.stderr = e
    _, _, P = eval_testv2.build(wl)
    assert len(P) == 98, (nm, len(P))
    pw = lambda s, th, ag: score_rc.to_wall(adv.db_time(s, th, ag), ag, effective_cores(s, th, ag)) / score_rc.BIAS_K
    fwd = [abs(pw(*k) / p["wall"] - 1) for k, p in P.items() if k not in ((16, 2, 224), (8, 2, 8))]
    base = P[(16, 2, 224)]["wall"]; ok = []
    for th in (2, 1):
        for ag in eval_testv2.AGS:
            L = {s: P[(s, th, ag)]["wall"] for s in RUNGS if (s, th, ag) in P}
            if len(L) < 4: continue
            for k in score_rc.KS:
                T = base * k
                truly = any(abs(v / T - 1) <= TOL for v in L.values())
                d, srv = min((abs(pw(s, th, ag) / T - 1), s) for s in L)
                mf = d <= TOL; hit = abs(L[srv] / T - 1) <= TOL
                ok.append((truly and mf and hit) or (not truly and not mf))
    R[nm] = dict(base=base, n=len(ok), ok=sum(ok), fwd=fwd)
allf = R["bal_test"]["fwd"] + R["test2"]["fwd"]; n = sum(r["n"] for r in R.values()); k = sum(r["ok"] for r in R.values())
out = dict(cmd="python3 demo_no_exclude.py", n=n, ok=k, closest=100 * k / n,
           per={w: dict(base=r["base"], ok=r["ok"], n=r["n"], pct=100 * r["ok"] / r["n"]) for w, r in R.items()},
           fwd_median=100 * st.median(allf), fwd_within15=100 * sum(e <= TOL for e in allf) / len(allf), n_fwd=len(allf))
assert n == 224 and len(allf) == 192
json.dump(out, open(f"{RESULTS}/demo_no_exclude.json", "w"), indent=1)
print(json.dumps(out, indent=1))
