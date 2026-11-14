#!/usr/bin/env python3
"""Scoring of the 224 targets over every core count 2..16, and prediction error at the counts outside the earlier grid.
224 个目标的评分: (a) 现行规则(只在网格 7 档里挑, = check_prod_v2.py 口径);(b) 遍历全部整数挑。
可行 = 2..16 某个整数核数实测在 ±15% 内(b 的口径);同时给出 (a) 口径的可行(只看网格)。
另给网格外点的预测误差(pred/meas-1)。预测走生产路径 phoebe_cli._load。"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))
from paths import DATA, RESULTS

import io, sys, os, json, importlib.util, statistics as st
from types import SimpleNamespace
import score_rc, eval_testv2
from phoebe_predictor import effective_cores
D = DATA
sp = importlib.util.spec_from_file_location("cli", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src", "phoebe_cli.py"))
cli = importlib.util.module_from_spec(sp); sp.loader.exec_module(cli)
TOL = 0.15; RUNGS = (2, 4, 6, 8, 11, 13, 16); ALL = range(2, 17)
def og(wl, s, th, ag):
    p = f"{D}/testv2og/{wl}/s{s}_l4_t{th}_a{ag}_b0/point.json"
    if not os.path.exists(p): return None
    d = json.load(open(p)); ex = d.get("passes", 0) * d.get("questions", 0)
    return d["completion_seconds"] if d.get("steps") == ex and ex > 0 else None
res = {"grid": [0, 0], "all": [0, 0], "all_complete": [0, 0]}; err = []; miss = 0
for wl, nm in (("test", "bal_test"), ("test2", "test2")):
    e = sys.stderr; sys.stderr = io.StringIO()
    try: adv = cli._load(SimpleNamespace(offline=f"{D}/offline_{nm}_v2.json", anchors=f"{D}/anchors_{nm}_v2.json",
                                         zero_shot_quantile=0.0, anchor_quantile=None))
    finally: sys.stderr = e
    _, _, P = eval_testv2.build(wl); base = P[(16, 2, 224)]["wall"]
    for th in (2, 1):
        for ag in eval_testv2.AGS:
            L = {s: P[(s, th, ag)]["wall"] for s in RUNGS if (s, th, ag) in P}
            if len(L) < 4: continue
            M = dict(L)
            for s in ALL:
                if s not in RUNGS:
                    y = og(wl, s, th, ag)
                    if y is not None: M[s] = y
                    else: miss += 1
            pr = {s: score_rc.to_wall(adv.db_time(s, th, ag), ag, effective_cores(s, th, ag)) / score_rc.BIAS_K for s in ALL}
            for s in M:
                if s not in RUNGS: err.append(pr[s] / M[s] - 1)
            complete = all(s in M for s in ALL)
            for k in score_rc.KS:
                T = base * k
                # (a) 网格
                truly = any(abs(v / T - 1) <= TOL for v in L.values())
                d, s = min((abs(pr[s] / T - 1), s) for s in L); mf = d <= TOL
                res["grid"][1] += 1; res["grid"][0] += (truly and mf and abs(L[s] / T - 1) <= TOL) or (not truly and not mf)
                # (b) 全部整数(只评实测齐全的情形)
                if not complete: continue
                truly = any(abs(v / T - 1) <= TOL for v in M.values())
                d, s = min((abs(pr[s] / T - 1), s) for s in ALL); mf = d <= TOL
                ok = (truly and mf and abs(M[s] / T - 1) <= TOL) or (not truly and not mf)
                res["all"][1] += 1; res["all"][0] += ok
                # 同一批情形上的 (a) 对照
                truly_g = any(abs(v / T - 1) <= TOL for v in L.values())
                d, s = min((abs(pr[s] / T - 1), s) for s in L); mf = d <= TOL
                res["all_complete"][1] += 1; res["all_complete"][0] += (truly_g and mf and abs(L[s] / T - 1) <= TOL) or (not truly_g and not mf)
f = lambda a: f"{a[0]}/{a[1]} = {100*a[0]/a[1]:.1f}%" if a[1] else "-"
print("网格外缺测点", miss, "/ 224")
print("(a) 只在网格挑, 全部 224 目标:", f(res["grid"]))
print("(b) 遍历全部整数, 实测齐全的情形:", f(res["all"]), "  同批情形上 (a):", f(res["all_complete"]))
if err:
    ae = sorted(abs(x) for x in err)
    print(f"网格外点预测误差: n={len(err)} 中位|误差|={100*st.median(ae):.1f}%  ±15%内 {100*sum(x<=TOL for x in ae)/len(ae):.1f}%  偏快(pred<meas) {100*sum(x<0 for x in err)/len(err):.1f}%")
