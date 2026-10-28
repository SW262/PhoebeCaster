#!/usr/bin/env python3
"""在新定义测试集(testv2, b0 与训练同定义)上评估三套模型。

两种口径都报:
  严格  : 有解时推荐档实测落进 ±15% 才算对; 无解时模型必须自己说无解。超线程开/关都算(224 组),
          与 score2 同范围。(2026-10-15 之前只算了超线程开, 没有理由, 已改。)
  score2: 与 score2_wall_r.py 同规则 —— 目标 = 锚点1(16核x224)实测墙钟 x k,
          k ∈ [0.8,1,1.3,1.8,2.5,3.5,5,7], 并发 [8,16,32,64,96,128,224], 超线程开/关都评;
          "确实无解"(网格里本来没有配置落进 ±15%)算对。
锚点: 16核/HT开/224 + 8核/HT开/8, 取 testv2 的实测(带 sql/ns 分量)。
"""
import json, glob, os, re, math, collections, statistics as st
import numpy as np
from phoebe_predictor import PhoebePredictor, effective_cores
from sql_model2 import SQLPredictor
import eval_parts, eval_old, score_rc
TOL = 0.15; KS = score_rc.KS; AGS = [8, 16, 32, 64, 96, 128, 224]
from paths import DATA
LAB = {"test": f"{DATA}/test_label_noise.json", "test2": f"{DATA}/test2_label_noise.json"}


def build(wl):
    cost = np.array([x["median"] for x in json.load(open(LAB[wl])) if x.get("median", 0) > 0])
    pts = {}; steps = None
    for p in glob.glob(f"{DATA}/testv2/{wl}/s*/point.json"):
        m = re.match(r"s(\d+)_l4_t(\d)_a(\d+)_b0$", os.path.basename(os.path.dirname(p)))
        if not m: continue
        d = json.load(open(p)); bk = d.get("db_time_breakdown_s") or {}
        if d["steps"] != d["passes"] * d["questions"]: continue
        steps = d["steps"]
        s, th, a = int(m[1]), int(m[2]), int(m[3])
        pts[(s, th, a)] = dict(C=effective_cores(s, th, a), ag=a, srv=s, th=th, sql=bk.get("sql", 0),
                               ns=bk.get("vec_mem", 0)+bk.get("vec_value", 0)+bk.get("tp_txn", 0),
                               db=d["db_time_s"], wall=d["completion_seconds"])
    return cost, steps, pts


def run(train, cls, wl):
    cost, steps, P = build(wl)
    anc = [P[(16, 2, 224)], P[(8, 2, 8)]]
    A = [(a["C"], a["ag"], a["sql"], a["ns"]) for a in anc]
    M = cls().fit(list(train.values()), anchor_rows=PhoebePredictor.anchor_as_rows(cost, steps, A))
    M.calibrate(cost, steps, A)
    pred = lambda s, th, a: M.predict(cost, steps, s, th, a)
    E = []
    for k, p in P.items():
        if k in ((16, 2, 224), (8, 2, 8)): continue
        pr = pred(*k); E.append(abs(pr/p["db"] - 1))
    base = P[(16, 2, 224)]["wall"]
    strict = [0, 0]; s2 = collections.Counter()
    for th in (2, 1):
        for ag in AGS:
            L = {s: P[(s, th, ag)]["wall"] for s in (2, 4, 6, 8, 11, 13, 16) if (s, th, ag) in P}
            if len(L) < 4: continue
            for k in KS:
                T = base * k
                truly = any(abs(v/T - 1) <= TOL for v in L.values())
                c = [(abs(score_rc.to_wall(pred(s, th, ag), ag, effective_cores(s, th, ag))/T - 1), s) for s in L]
                d, srv = min(c); mf = d <= TOL; hit = abs(L[srv]/T - 1) <= TOL
                s2["命中" if hit else ("确实无解" if not truly else "错")] += 1
                strict[1] += 1
                if (truly and mf and hit) or (not truly and not mf): strict[0] += 1
    return E, strict, s2


if __name__ == "__main__":
    OLD, NEW = eval_old.build_old(), eval_parts.build()
    CFG = [("v1  老网格 + 现模型 (线上)", OLD, PhoebePredictor),
           ("    老网格 + 模型3", OLD, SQLPredictor),
           ("    rc3+rc4 + 现模型", NEW, PhoebePredictor),
           ("v2  rc3+rc4 + 模型3", NEW, SQLPredictor)]
    for wl in ("test", "test2"):
        _, _, P = build(wl)
        print(f"  {wl}: {len(P)} 个配置")
    print(f"\n  {'':<28}{'严格 bal':>10}{'严格 t2':>10}{'严格合计':>12}{'score2 合计':>14}{'误差中位 bal/t2':>18}")
    for nm, tr, cls in CFG:
        R = {wl: run(tr, cls, wl) for wl in ("test", "test2")}
        sb, st2 = R["test"][1], R["test2"][1]
        g = sum(R[w][2]["命中"]+R[w][2]["确实无解"] for w in R); n = sum(sum(R[w][2].values()) for w in R)
        print(f"  {nm:<28}{100*sb[0]/sb[1]:>9.1f}%{100*st2[0]/st2[1]:>9.1f}%"
              f"{sb[0]+st2[0]:>6}/{sb[1]+st2[1]}={100*(sb[0]+st2[0])/(sb[1]+st2[1]):.1f}%"
              f"{g:>6}/{n}={100*g/n:.1f}%"
              f"{100*st.median(R['test'][0]):>9.1f}% /{100*st.median(R['test2'][0]):>5.1f}%")
