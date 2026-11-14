#!/usr/bin/env python3
"""Section 4 numbers that depend on the core-selection rule; 'all' = every core count 2..16 (the paper), 'grid' = the earlier 7-count grid.
   python3 allint_numbers.py grid  -> 只在网格 7 档挑/判(= 现论文, 用来自检)
   python3 allint_numbers.py all   -> 在 2..16 全部整数挑/判(工具现行规则; 网格外实测 testv2og)
预测走生产路径(phoebe_cli._load + advise.to_wall, 同 advise.py); Amdahl 同 demo_usl_baseline.py;
截止时间规则同 demo_usl_baseline.py / demo_deadline_bias.py。"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))
from paths import DATA, RESULTS

import io, sys, os, json, importlib.util, statistics as st
from types import SimpleNamespace
import score_rc, eval_testv2, advise
from phoebe_predictor import effective_cores
D = DATA
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
TOL = 0.15; RUNGS = (2, 4, 6, 8, 11, 13, 16)
CAND = RUNGS if MODE == "grid" else tuple(range(2, 17))
sp = importlib.util.spec_from_file_location("cli", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src", "phoebe_cli.py")); cli = importlib.util.module_from_spec(sp); sp.loader.exec_module(cli)
def og(wl, s, th, ag):
    p = f"{D}/testv2og/{wl}/s{s}_l4_t{th}_a{ag}_b0/point.json"
    if not os.path.exists(p): return None
    d = json.load(open(p)); return d["completion_seconds"] if d["steps"] == d["passes"] * d["questions"] else None
rows = []; fwd = {"ours": [], "am": [], "amh": []}; bias = {"ours": [], "am": [], "amh": []}
for wl, nm in (("test", "bal_test"), ("test2", "test2")):
    e = sys.stderr; sys.stderr = io.StringIO()
    try: adv = cli._load(SimpleNamespace(offline=f"{D}/offline_{nm}_v2.json", anchors=f"{D}/anchors_{nm}_v2.json", zero_shot_quantile=0.0, anchor_quantile=None))
    finally: sys.stderr = e
    _, _, P = eval_testv2.build(wl); base = P[(16, 2, 224)]["wall"]
    ours = lambda s, th, ag: advise.to_wall(adv.db_time(s, th, ag), ag, effective_cores(s, th, ag))
    def amdahl(hmode):
        n_ = lambda s, th, ag: min(ag, effective_cores(s, th, ag) if hmode else s)
        (n1, t1), (n2, t2) = [(n_(*k), P[k]["wall"]) for k in ((16, 2, 224), (8, 2, 8))]
        v = (t1 * n1 - t2 * n2) / (n1 - n2); u = t1 * n1 - v * (n1 - 1)
        return lambda s, th, ag: (u + v * (n_(s, th, ag) - 1)) / n_(s, th, ag)
    M = {"ours": ours, "am": amdahl(False), "amh": amdahl(True)}
    for th in (2, 1):
        for ag in eval_testv2.AGS:
            G = {s: P[(s, th, ag)]["wall"] for s in RUNGS if (s, th, ag) in P}
            if len(G) < 4: continue
            L = dict(G)
            if MODE == "all":
                for s in CAND:
                    if s not in L: L[s] = og(wl, s, th, ag)
                assert all(L[s] is not None for s in CAND), (wl, th, ag)
            for s in L:
                for m, f in M.items(): bias[m].append(f(s, th, ag) / L[s] - 1)   # 偏差: 同 demo_deadline_bias.py, 含锚点
                if (s, th, ag) in ((16, 2, 224), (8, 2, 8)): continue
                for m, f in M.items(): fwd[m].append(abs(f(s, th, ag) / L[s] - 1))
            for k in score_rc.KS:
                T = base * k; truly = any(abs(L[s] / T - 1) <= TOL for s in CAND)
                r = dict(wl=nm, ht="on" if th == 2 else "off", ag=ag, k=k, feasible=truly)
                for m, f in M.items():
                    d, s = min((abs(f(s, th, ag) / T - 1), s) for s in CAND); fits = d <= TOL
                    hit = abs(L[s] / T - 1) <= TOL
                    ok = (fits and hit) if truly else (not fits)
                    kind = None if ok else ("none_fits" if not fits else ("under" if L[s] > T * (1 + TOL) else "over" if L[s] < T * (1 - TOL) else "pick_but_infeasible"))
                    cand = [x for x in CAND if f(x, th, ag) <= T]; meet = [x for x in CAND if L[x] <= T]
                    dl_ok = (L[cand[0]] <= T) if cand else (not meet)
                    dl_fewest = (cand[0] == min(meet)) if (cand and L[cand[0]] <= T) else None
                    r[m] = dict(ok=ok, kind=kind, srv=s, fits=fits, dl_ok=dl_ok, dl_fewest=dl_fewest)
                rows.append(r)
pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None
o = [r["ours"] for r in rows]
out = dict(mode=MODE, n=len(rows),
  fwd={m: dict(n=len(v), median=round(100 * st.median(v), 1), within15=pct([x <= TOL for x in v]), bias_median=round(100 * st.median(bias[m]), 1)) for m, v in fwd.items()},
  ours_all=pct([x["ok"] for x in o]),
  feasible=(sum(r["feasible"] for r in rows), pct([r["ours"]["ok"] for r in rows if r["feasible"]])),
  infeasible=(sum(not r["feasible"] for r in rows), pct([r["ours"]["ok"] for r in rows if not r["feasible"]])),
  wrong=len([x for x in o if not x["ok"]]), kinds={k: sum(x["kind"] == k for x in o) for k in ("under", "over", "none_fits", "pick_but_infeasible")},
  bins={b: (len(v), pct([x["ok"] for x in v])) for b, v in {
      "2-4": [x for x in o if x["srv"] <= 4], "5": [x for x in o if x["srv"] == 5], "6-13": [x for x in o if 6 <= x["srv"] <= 13], "14-15": [x for x in o if 14 <= x["srv"] <= 15], "16": [x for x in o if x["srv"] == 16]}.items()},
  wrong_bal=sum(1 for r in rows if not r["ours"]["ok"] and r["wl"] == "bal_test"),
  ht={h: (sum(r["ht"] == h for r in rows), pct([r["ours"]["ok"] for r in rows if r["ht"] == h])) for h in ("on", "off")},
  bal_off=pct([r["ours"]["ok"] for r in rows if r["ht"] == "off" and r["wl"] == "bal_test"]),
  by_wl={w: {m: pct([r[m]["ok"] for r in rows if r["wl"] == w]) for m in ("ours", "amh")} for w in ("bal_test", "test2")},
  amdahl=pct([r["am"]["ok"] for r in rows]), amdahl_h=pct([r["amh"]["ok"] for r in rows]),
  deadline={m: pct([r[m]["dl_ok"] for r in rows]) for m in ("ours", "am", "amh")},
  deadline_overprov={m: (lambda v: round(100 - pct(v), 0) if v else None)([r[m]["dl_fewest"] for r in rows if r[m]["dl_fewest"] is not None]) for m in ("ours", "am", "amh")})
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(f"{RESULTS}/allint_numbers_{MODE}.json", "w"), ensure_ascii=False, indent=1)
