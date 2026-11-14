#!/usr/bin/env python3
"""Accuracy split by hyper-threading, run time of all 210 configurations, the h(N) table, and statement timeouts.
  ① 推荐正确率按超线程开/关拆分(demo_failmap.json, 先跑 demo_failmap.py)
  ② 完整网格的运行墙钟合计(testv2/<wl>/s*_l4_t*_a*_b0/point.json 的 completion_seconds; 不含建库/清缓存)
  ③ 超线程增益 h(N) 表(ht_gain_fit.json)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))
from paths import DATA, RESULTS

import json, glob, os, collections
D = DATA
T = json.load(open(f"{RESULTS}/demo_failmap.json"))["targets"]
c = collections.defaultdict(lambda: [0, 0])
for t in T:
    for k in (t["ht"], (t["workload"], t["ht"])):
        c[k][0] += t["ok"]; c[k][1] += 1
print("① 正确率")
for k in ("on", "off", ("bal_test", "on"), ("bal_test", "off"), ("test2", "on"), ("test2", "off")):
    print(f"  {str(k):22s} {c[k][0]}/{c[k][1]} = {100*c[k][0]/c[k][1]:.1f}%")
print("② 完整网格运行墙钟")
for wl, nm in (("test", "bal_test"), ("test2", "test2")):
    # 2026-11-11: 全部 2..16 核 = 网格 testv2/ + 网格外 testv2og/(对照点 *_ctl* 不算)
    P = glob.glob(f"{D}/testv2/{wl}/s*_l4_t*_a*_b0/point.json") + glob.glob(f"{D}/testv2og/{wl}/s*_l4_t*_a*_b0/point.json")
    s = sum(json.load(open(p))["completion_seconds"] for p in P)
    print(f"  {nm:9s} {len(P)} 个配置, 合计 {s/3600:.1f} h")
print("③ h(N) =", json.load(open(f"{D}/ht_gain_fit.json"))["table"])
# ④ 2026-11-11: 被 30 s statement_timeout 取消过查询的运行(sql_errors 超出该负载固定基线 = 超时次数;
#    testv2og 的 PG 日志逐点核对过, 与超出数逐个相等)。全部 2..16 核 420 次运行(testv2/ + testv2og)。
import re
BASE = {"test": 450, "test2": 982}; TO = []
for d in ("testv2", "testv2og"):
    for p in glob.glob(f"{D}/{d}/*/s*_l4_t*_a*_b0/point.json"):
        wl = p.split("/")[-3]; j = json.load(open(p)); x = j["sql_errors"] - BASE[wl]
        if x > 0: m = re.match(r"s(\d+)_l4_t(\d)_a(\d+)", p.split("/")[-2]); TO.append((int(m[1]), int(m[3]), x))
print(f"④ 有语句超时的运行: {len(TO)} / 420; 核数 {min(t[0] for t in TO)}–{max(t[0] for t in TO)}; 并发 {min(t[1] for t in TO)}–{max(t[1] for t in TO)}; 每次 {min(t[2] for t in TO)}–{max(t[2] for t in TO)} 条")
