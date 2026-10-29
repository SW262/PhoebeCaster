#!/usr/bin/env python3
"""demo 论文第 2–4 节新增的数字, 一条命令重算:  python3 rev1_numbers.py -> rev1_numbers.json
口径全部同论文: testv2 两个留出负载, 224 个目标(demo_failmap.json, 生产路径), 7 个实测核档, ±15% 双向。
  split   84.4% 拆成 可行 / 不可行目标, 错误再分 资源不足 / 过度配置 / 漏报 / 误报
  sla     另一种规则(探索性, 看过 84.4% 之后才算): 选"预测 ≤ T 的最少核数"; 实测 ≤ T 才算达标
  wall    db_time -> 墙钟换算本身的误差: 用实测 db_time 换算, 与实测墙钟比
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))
from paths import DATA, RESULTS

import json, os, glob, re, statistics as st, collections
import eval_testv2, badpoints
from advise import to_wall
from phoebe_predictor import effective_cores
D = DATA; TOL = 0.15
T = json.load(open(f"{RESULTS}/demo_failmap.json"))["targets"]; assert len(T) == 224
R = {}
fe = [t for t in T if t["feasible"]]; inf = [t for t in T if not t["feasible"]]
R["split"] = dict(feasible=len(fe), feasible_ok=sum(t["ok"] for t in fe),
                  infeasible=len(inf), infeasible_ok=sum(t["ok"] for t in inf))
err = collections.Counter()
for t in T:
    if t["ok"]: continue
    if t["kind"] == "missed_fit": err["missed_fit"] += 1           # 有解却说无解
    else:                                                          # 给了档但实测没落进 ±15%
        err["under_provisioned" if t["measured"] > t["target"] else "over_provisioned"] += 1
R["errors"] = dict(err)
# SLA 规则: 最少核数且 预测 ≤ T; 没有就报无解。对 = 选了档且实测 ≤ T, 或报无解且网格里确实没有档实测 ≤ T
sla = collections.Counter(); extra = []
for t in T:
    rg = t["rungs"]; T_ = t["target"]
    pick = next((r for r in rg if r["pred"] <= T_), None)
    can = [r["srv"] for r in rg if r["measured"] <= T_]
    if pick is None: sla["right_none" if not can else "missed"] += 1
    elif pick["measured"] <= T_:
        sla["met"] += 1; extra.append(pick["srv"] - min(can))     # 比"实测达标的最少核数"多几核
    else: sla["violated"] += 1
R["sla"] = dict(n=len(T), **sla, ok=sla["met"] + sla["right_none"],
                extra_cores_when_met=dict(collections.Counter(extra)))
# 墙钟换算误差
E = []
for wl in ("test", "test2"):
    _, _, P = eval_testv2.build(wl)
    for (s, th, a), p in P.items():
        E.append(abs(to_wall(p["db"], a, effective_cores(s, th, a)) / p["wall"] - 1))
R["wall"] = dict(n=len(E), median=st.median(E), p90=sorted(E)[int(0.9 * len(E))], max=max(E))
# 探测耗时(估算): 6 遍 x 逐条中位离线代价之和; 13 个负载(2 留出 + 11 训练)。不含报错/超时与连接开销。
pr = {w: 6 * sum(json.load(open(f"{D}/offline_{w}_v2.json"))["cost_ms"]) / 1000 for w in ("bal_test", "test2")}
acc = collections.defaultdict(float)
for x in json.load(open(f"{D}/train_label_noise.json")):
    if x.get("median", 0) > 0 and "pdb_" + x["db"] not in badpoints.EXCLUDED_LIBS: acc[x["db"]] += x["median"]
pr.update({w: 6 * v / 1000 for w, v in acc.items()})
v = sorted(pr.values())
R["probe"] = dict(n=len(v), seconds={w: round(x, 1) for w, x in sorted(pr.items(), key=lambda t: t[1])},
                  heldout=[round(pr["test2"], 1), round(pr["bal_test"], 1)], median=st.median(v),
                  min=v[0], max=v[-1], under20=sum(x < 20 for x in v))
# 每个留出负载的一次性测量代价(秒) = 探测(估算) + 两次锚点运行的实测墙钟
cost = {}
for wl, pk in (("test", "bal_test"), ("test2", "test2")):
    _, _, P = eval_testv2.build(wl)
    cost[pk] = pr[pk] + P[(16, 2, 224)]["wall"] + P[(8, 2, 8)]["wall"]
R["cost"] = dict(seconds={k: round(v) for k, v in cost.items()}, min=min(cost.values()), max=max(cost.values()))
# 图 3: 各分量每步耗时随并发的膨胀(相对 8 并发), 两个留出负载, 16 核 / HT 开, testv2 实测
CMP = ("vec_mem", "vec_value", "sql", "tp_txn")
dil = {}
for wl, pk in (("test", "bal_test"), ("test2", "test2")):
    ms = {}                                                        # 同 eval_testv2.build 的完整点筛选
    for f in glob.glob(f"{D}/testv2/{wl}/s16_l4_t2_a*_b0/point.json"):
        d = json.load(open(f))
        if d["steps"] != d["passes"] * d["questions"]: continue
        ms[int(re.search(r"_a(\d+)_b0", f)[1])] = {c: d["db_time_breakdown_s"].get(c, 0) / d["steps"] * 1000 for c in CMP}
    dil[pk] = {"agents": sorted(ms), **{c: [ms[a][c] / ms[8][c] for a in sorted(ms)] for c in CMP}}
    dil[pk]["spread224"] = max(dil[pk][c][-1] for c in CMP) / min(dil[pk][c][-1] for c in CMP)
R["dilation"] = dil
# 错在哪里: 按所选核数分三段(低锚点 8 核以下 / 中间 / 网格顶 16 核), 以及按负载
seg = lambda s: "2-4" if s <= 4 else ("16" if s == 16 else "6-13")
by = collections.defaultdict(lambda: [0, 0])
for t in T:
    by[seg(t["srv"])][0] += 1; by[seg(t["srv"])][1] += t["ok"]
    by[t["workload"]][0] += 1; by[t["workload"]][1] += t["ok"]
R["where"] = {k: dict(n=n, ok=o, rate=o / n) for k, (n, o) in by.items()}
# 预测偏差(有符号, 预测/实测 − 1 的中位), 按核数段; 每个 (负载, HT, 并发) 的各核档只计一次
_seen, _b = set(), collections.defaultdict(list)
for t in T:
    k = (t["workload"], t["ht"], t["agents"])
    if k in _seen: continue
    _seen.add(k)
    for r in t["rungs"]: _b[seg(r["srv"])].append(r["pred"] / r["measured"] - 1)
R["bias"] = {k: st.median(v) for k, v in _b.items()}
# 并发指数范围(训练负载, 同 PAPER_NUMBERS_v2.md 的命令)
import eval_parts
from phoebe_predictor import workload_ag_exponents
ex = [float(x) for x in workload_ag_exponents(list(eval_parts.build().values()))]
R["exponent"] = dict(n=len(ex), min=min(ex), max=max(ex))
# 慢查询筛查(sql_screen.json, 只覆盖训练库): 剔除负载之外, 能执行的轮次里单独 >1 s 被排除的; 都与 BIRD 标准答案不符
sc = json.load(open(f"{D}/sql_screen.json"))
rows = [r for r in sc["rows"] if "pdb_" + r["db"] not in badpoints.EXCLUDED_LIBS]
exe = [r for r in rows if r.get("error") is None and r.get("ms") is not None]
exc = [r for r in rows if r["excluded"]]
R["screen"] = dict(executable=len(exe), median_ms=st.median(r["ms"] for r in exe),
                   excluded=[dict(qid=r["qid"], db=r["db"], round=r["round"], ms=r["ms"]) for r in exc])
# 留出负载的筛查(test_/test2_sql_screen.json, 2026-10-29 起回放才真正生效): 900086 答案与标准答案一致(74,299)
hx = [r for f in ("test_sql_screen.json", "test2_sql_screen.json") for r in json.load(open(f"{D}/{f}"))["rows"]]
R["screen_heldout"] = dict(executable=sum(r.get("error") is None and r.get("ms") is not None for r in hx),
                           excluded=[dict(qid=r["qid"], db=r["db"], round=r["round"], ms=r["ms"]) for r in hx if r["excluded"]])
json.dump(R, open(f"{RESULTS}/rev1_numbers.json", "w"), indent=1); print(json.dumps(R, indent=1))
