#!/usr/bin/env python3
"""Numbers for the low-core extrapolation paragraph of Section 4.
  ① 每个核档的预测偏差中位(墙钟, 与网页找错面板同源 demo_failmap.json; 先跑 demo_failmap.py)
  ② 2->8 核的核数斜率 d log(时间)/d log(有效核), 实测 vs 生产模型(phoebe_cli._load, 同 demo_failmap.py)
  ③ 训练 10 库的平均单条成本与 8->16 核 SQL 斜率(train_set_v2.json), 以及 bal_test / test2 的平均成本
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))
from paths import DATA, RESULTS

import io, sys, os, json, math, importlib.util, statistics as st
from types import SimpleNamespace
import numpy as np
import eval_testv2
from phoebe_predictor import effective_cores
D = DATA

# ①
T = json.load(open(f"{RESULTS}/demo_failmap.json"))["targets"]
seen, err = set(), {}
for t in T:
    for r in t["rungs"]:
        k = (t["workload"], t["ht"], t["agents"], r["srv"])
        if k in seen: continue
        seen.add(k); err.setdefault((t["workload"], r["srv"]), []).append(r["pred"] / r["measured"] - 1)
print("① 预测偏差中位(墙钟, 负 = 预测偏快)")
print("  2..4 核(低锚点以下, 两负载合并): %+.1f%%" % (100 * st.median([x for (w, s_), v in err.items() if s_ <= 4 for x in v])))
for w in ("bal_test", "test2"):
    print(f"  {w:9s}" + "  ".join(f"s{s}={100*st.median(err[(w, s)]):+.1f}%" for s in sorted({k[1] for k in err})))
bad = {}
for t in T:
    if t["srv"] <= 4: bad.setdefault(t["workload"], [0, 0]); bad[t["workload"]][0] += not t["ok"]; bad[t["workload"]][1] += 1
print("  选 <=4 核的目标 错/总:", bad)

# ②
sp = importlib.util.spec_from_file_location("cli", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src", "phoebe_cli.py"))
cli = importlib.util.module_from_spec(sp); sp.loader.exec_module(cli)
print("② 核数斜率 2->8 核(中位, 14 个 (HT, 并发) 组合)")
for wl, nm in (("test", "bal_test"), ("test2", "test2")):
    e = sys.stderr; sys.stderr = io.StringIO()
    try: adv = cli._load(SimpleNamespace(offline=f"{D}/offline_{nm}_v2.json", anchors=f"{D}/anchors_{nm}_v2.json",
                                         zero_shot_quantile=0.0, anchor_quantile=None))
    finally: sys.stderr = e
    m = adv.m
    assert hasattr(m, "cal_") and not hasattr(m, "cal_sql_")         # 生产路径: 总量标定, 两分量同乘一个因子
    def parts(s, th, ag):
        sq, ns = m._parts(adv.cost, adv.steps, ag, effective_cores(s, th, ag))
        f = math.exp(m.cal_[0] * math.log(ag) + m.cal_[1]); return sq * f, ns * f
    cost, _, P = eval_testv2.build(wl)
    R = {k: [] for k in ("ms", "mn", "ps", "pn", "sh_m", "sh_p")}
    for (s, th, ag), p in P.items():
        if s != 2 or (8, th, ag) not in P: continue
        q = P[(8, th, ag)]; L = math.log(p["C"] / q["C"])
        a1, b1 = parts(2, th, ag); a2, b2 = parts(8, th, ag)
        R["ms"].append(math.log(p["sql"] / q["sql"]) / L); R["mn"].append(math.log(p["ns"] / q["ns"]) / L)
        R["ps"].append(math.log(a1 / a2) / L); R["pn"].append(math.log(b1 / b2) / L)
        R["sh_m"].append(q["ns"] / (q["sql"] + q["ns"])); R["sh_p"].append(b2 / (a2 + b2))
    M = {k: st.median(v) for k, v in R.items()}
    print(f"  {nm:9s} SQL 实测 {M['ms']:+.2f} 模型 {M['ps']:+.2f} | 非SQL 实测 {M['mn']:+.2f} 模型 {M['pn']:+.2f} | "
          f"8 核非SQL占比 实测 {100*M['sh_m']:.0f}% 模型 {100*M['sh_p']:.0f}% | 平均成本 {np.mean(cost):.1f} ms")

# ③
print("③ 训练 10 库: 平均成本 vs 8->16 核 SQL 斜率")
C, S = [], []
for w in json.load(open(f"{D}/train_set_v2.json"))["workloads"]:
    Q = {(p["srv"], p["th"], p["ag"]): p for p in w["points"]}
    sl = [math.log(Q[k]["sql"] / Q[(16,) + k[1:]]["sql"]) / math.log(Q[k]["C"] / Q[(16,) + k[1:]]["C"])
          for k in Q if k[0] == 8 and (16,) + k[1:] in Q and Q[k]["sql"] > 0 and Q[(16,) + k[1:]]["sql"] > 0]
    C.append(float(np.mean(w["cost"]))); S.append(st.median(sl))
    print(f"  {w['lib']:28s} 成本 {C[-1]:5.2f} ms  斜率 {S[-1]:+.2f}")
rk = lambda x: np.argsort(np.argsort(x))
print(f"  最大成本 {max(C):.1f} ms; 斜率中位 {st.median(S):+.2f}; Spearman(成本, 斜率) = {np.corrcoef(rk(C), rk(S))[0,1]:+.2f}")
