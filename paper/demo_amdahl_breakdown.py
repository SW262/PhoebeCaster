#!/usr/bin/env python3
"""Where Amdahl's law + h(N) and PhoebeCaster differ: forward error and recommendation accuracy by core range and workload.
先跑 demo_failmap.py 与 demo_usl_baseline.py none / h。   python3 demo_amdahl_breakdown.py
  ① 前向误差(192 点, 去锚点): 中位 / ±15% 内, 按核数段(2–4, 6–13, 16)与负载拆
  ② 推荐正确率(224 目标), 按"实测最接近 T 的网格核数"分段(模型无关的分组)与负载拆; 配对 McNemar
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))
from paths import DATA, RESULTS

import os, json, statistics as st
from math import comb
D = DATA
F = json.load(open(f"{RESULTS}/demo_failmap.json"))["targets"]
ANCH = {(16, "on", 224), (8, "on", 8)}
ours = {}
for t in F:
    for r in t["rungs"]:
        k = (t["workload"], r["srv"], t["ht"], t["agents"])
        if (r["srv"], t["ht"], t["agents"]) not in ANCH: ours[k] = abs(r["pred"] / r["measured"] - 1)
OK = {(t["workload"], t["ht"], t["agents"], t["k"]): t["ok"] for t in F}
seg = lambda s: "2-4" if s <= 4 else ("16" if s == 16 else "6-13")
pct = lambda xs: 100 * sum(xs) / len(xs) if xs else float("nan")
R = {}
for m, lab in (("none", "Amdahl"), ("h", "Amdahl+h(N)")):
    U = json.load(open(f"{RESULTS}/demo_usl_baseline_{m}.json"))
    fu = {(p["workload"], p["srv"], p["ht"], p["agents"]): p["err"] for p in U["fwd_points"]}
    assert set(fu) == set(ours) and len(fu) == 192
    R[lab] = U
    print(f"== {lab}")
    print("① 前向误差  中位(ours/base)  ±15%内(ours/base)")
    groups = [("all", lambda k: True)] + [(g, (lambda g: lambda k: seg(k[1]) == g)(g)) for g in ("2-4", "6-13", "16")]\
             + [(w, (lambda w: lambda k: k[0] == w)(w)) for w in ("bal_test", "test2")]
    for g, f in groups:
        ks = [k for k in fu if f(k)]
        print(f"  {g:9s} n={len(ks):3d}  {100*st.median([ours[k] for k in ks]):5.1f}/{100*st.median([fu[k] for k in ks]):5.1f}"
              f"   {pct([ours[k] <= .15 for k in ks]):5.1f}/{pct([fu[k] <= .15 for k in ks]):5.1f}")
    print("② 推荐正确率(按实测最接近 T 的核数分段)  ours/base   只ours对/只base对")
    T = U["targets"]
    g2 = [("all", lambda t: True)] + [(g, (lambda g: lambda t: seg(t["best"]) == g)(g)) for g in ("2-4", "6-13", "16")]\
         + [(w, (lambda w: lambda t: t["workload"] == w)(w)) for w in ("bal_test", "test2")]
    for g, f in g2:
        ts = [t for t in T if f(t)]
        a = sum(t["ok_ours"] and not t["ok"] for t in ts); b = sum(t["ok"] and not t["ok_ours"] for t in ts); n = a + b
        p = min(1, 2 * sum(comb(n, i) for i in range(0, min(a, b) + 1)) / 2 ** n) if n else 1
        print(f"  {g:9s} n={len(ts):3d}  {pct([t['ok_ours'] for t in ts]):5.1f}/{pct([t['ok'] for t in ts]):5.1f}   {a}/{b}  p={p:.3f}")
