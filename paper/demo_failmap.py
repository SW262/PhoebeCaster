#!/usr/bin/env python3
"""网页"找错"面板的数据: 224 个机械生成目标逐个的对错, 判定逻辑逐行照抄 check_prod_v2.py
(生产路径 phoebe_cli._load, 与 advise.py / 网页同一条)。合计必须等于 check_prod_v2 的 189/224,
不等就 assert 失败, 不落盘。   python3 demo_failmap.py  -> demo_failmap.json
2026-11-11: 候选与判定改为 2..16 全部整数核数(网格外实测 testv2og), 与 advise.py 现行扫描同口径;
  合计须等于 testv2og_eval.py / allint_numbers.py all 的 186/224。旧(只网格)版 demo_failmap.py.bak-allint。"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))
from paths import DATA, RESULTS

import io, sys, os, json, importlib.util
from types import SimpleNamespace
import score_rc, eval_testv2
from phoebe_predictor import effective_cores
D = DATA
sp = importlib.util.spec_from_file_location("cli", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src", "phoebe_cli.py"))
cli = importlib.util.module_from_spec(sp); sp.loader.exec_module(cli)
TOL = 0.15; RUNGS = tuple(range(2, 17))
def og(wl, s, th, ag):
    p = f"{D}/testv2og/{wl}/s{s}_l4_t{th}_a{ag}_b0/point.json"
    d = json.load(open(p)); assert d["steps"] == d["passes"] * d["questions"], p
    return d["completion_seconds"]
out = []
for wl, nm in (("test", "bal_test"), ("test2", "test2")):
    e = sys.stderr; sys.stderr = io.StringIO()
    try: adv = cli._load(SimpleNamespace(offline=f"{D}/offline_{nm}_v2.json", anchors=f"{D}/anchors_{nm}_v2.json",
                                         zero_shot_quantile=0.0, anchor_quantile=None))
    finally: sys.stderr = e
    _, _, P = eval_testv2.build(wl)
    base = P[(16, 2, 224)]["wall"]
    for th in (2, 1):
        for ag in eval_testv2.AGS:
            L = {s: P[(s, th, ag)]["wall"] for s in RUNGS if (s, th, ag) in P}
            if len(L) < 4: continue
            L.update({s: og(wl, s, th, ag) for s in RUNGS if s not in L})
            for k in score_rc.KS:
                T = base * k
                truly = any(abs(v / T - 1) <= TOL for v in L.values())
                c = [(abs(score_rc.to_wall(adv.db_time(s, th, ag), ag, effective_cores(s, th, ag)) / score_rc.BIAS_K / T - 1), s) for s in L]
                preds = {s_: score_rc.to_wall(adv.db_time(s_, th, ag), ag, effective_cores(s_, th, ag)) / score_rc.BIAS_K for s_ in L}
                d, srv = min(c); mf = d <= TOL; hit = abs(L[srv] / T - 1) <= TOL
                ok = (truly and mf and hit) or (not truly and not mf)
                kind = ("right" if ok else
                        "wrong_core" if (truly and mf and not hit) else
                        "missed_fit" if (truly and not mf) else "false_fit")
                out.append(dict(workload=nm, ht="on" if th == 2 else "off", agents=ag, target=T, k=k,
                                feasible=truly, said_fit=mf, srv=srv, measured=L[srv],
                                pred_dev=d, ok=ok, kind=kind,
                                rungs=[dict(srv=s_, pred=preds[s_], measured=L[s_]) for s_ in sorted(L)]))
n_ok = sum(r["ok"] for r in out)
assert (n_ok, len(out)) == (186, 224), f"{n_ok}/{len(out)} != 186/224(11-11 全部整数; 只网格时 189)"
json.dump(dict(cmd="python3 demo_failmap.py", tol=TOL, ok=n_ok, n=len(out), targets=out),
          open(f"{RESULTS}/demo_failmap.json", "w"), ensure_ascii=False)
import collections
print(n_ok, len(out), collections.Counter(r["kind"] for r in out))
