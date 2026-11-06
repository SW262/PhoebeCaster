#!/usr/bin/env python3
"""Accuracy split by hyper-threading, full-grid run time, and the h(N) table.
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
    P = glob.glob(f"{D}/testv2/{wl}/s*_l4_t*_a*_b0/point.json")
    s = sum(json.load(open(p))["completion_seconds"] for p in P)
    print(f"  {nm:9s} {len(P)} 个配置, 合计 {s/3600:.1f} h")
print("③ h(N) =", json.load(open(f"{D}/ht_gain_fit.json"))["table"])
