#!/usr/bin/env python3
"""把 score2_wall_r.py 的口径搬到 rc3+rc4 / 老网格上, 做真正可比的对照。

score2 的口径(逐条照搬):
  * 目标 = 锚点1(16核x224并发) 的【实测墙钟】x k,  k ∈ [0.8,1,1.3,1.8,2.5,3.5,5,7]
  * 预测值也换成墙钟: wall = BIAS_K · db_time / 并发 / r(并发, 有效核)
  * 选档 = "最近档"(|预测/目标| 最接近 1), 预测落在 ±15% 内才算 ok
  * 三分类: 推荐命中 / 确实无解(网格里本来就没有配置能落进 ±15%) / 本来有解但没命中
  * 达标率 = (命中 + 确实无解) / 组数
模型侧: 留一库交叉验证 + 2 锚点, 与 eval_parts.py 相同(score2 用的是定稿模型,
这里必须留一, 否则就是拿测试数据训练)。
"""
import json, glob, os, re, math, collections, sys
import statistics as st
import numpy as np
from phoebe_predictor import PhoebePredictor, effective_cores, QP
from ns_model2 import NSPredictor
import eval_parts, eval_old

from paths import DATA as D
TOL = 0.15
RM = json.load(open(f"{D}/r_model.json"))
KS = [0.8, 1, 1.3, 1.8, 2.5, 3.5, 5, 7]
import phoebe_advisor
BIAS_K = getattr(phoebe_advisor, "BIAS_K", 1.0)


def to_wall(db, ag, C):
    return BIAS_K * db / ag / math.exp(RM["a"]*math.log(ag) + RM["b"]*math.log(C) + RM["c"])


def bench(W, walls):
    """walls: {(lib,srv,ag): 实测墙钟}"""
    return walls


def load_walls(which):
    out = {}
    import badpoints; NOM = badpoints.nominal_steps()
    if which == "new":
        tmp = collections.defaultdict(list)
        for run in ("rc3", "rc4"):
            for p in glob.glob(f"{D}/{run}/pdb_*/*/point.json"):
                lib, tag = p.split(os.sep)[-3], p.split(os.sep)[-2]
                m = re.match(r"s(\d+)_l4_t2_a(\d+)_b0_x\d+_k\d+$", tag)
                if not m: continue
                d = json.load(open(p))
                if d.get("steps", 0) != NOM.get(lib, -1): continue
                if d.get("completion_seconds"):
                    tmp[(lib, int(m[1]), int(m[2]))].append(d["completion_seconds"])
        out = {k: st.mean(v) for k, v in tmp.items()}
    else:
        for p in glob.glob(f"{D}/pdb_*_grid/s*/point.json"):
            lib = os.path.basename(os.path.dirname(os.path.dirname(p)))[:-5]
            m = re.match(r"s(\d+)_l4_t2_a(\d+)_b0\.2$", os.path.basename(os.path.dirname(p)))
            if not m: continue
            d = json.load(open(p))
            if d.get("completion_seconds"):
                out[(lib, int(m[1]), int(m[2]))] = d["completion_seconds"]
    return out


def score(W, walls, cls):
    libs = sorted(W); cat = collections.Counter()
    for held in libs:
        w = W[held]; anc = eval_parts.anchors(w)
        if len(anc) < 2: continue
        A = [(a["C"], a["ag"], a["sql"], a["ns"]) for a in anc]
        M = cls().fit([W[l] for l in libs if l != held],
                      anchor_rows=PhoebePredictor.anchor_as_rows(w["cost"], w["steps"], A))
        M.calibrate(w["cost"], w["steps"], A)
        a1 = max(anc, key=lambda p: p["ag"])                       # 锚点1 = 最高并发那个
        base = walls.get((held, a1["srv"], a1["ag"]))
        if not base: continue
        srvs = sorted({p["srv"] for p in w["points"]})
        for ag in sorted({p["ag"] for p in w["points"]}):
            L = {s: walls[(held, s, ag)] for s in srvs if (held, s, ag) in walls}
            if len(L) < 4: continue
            for k in KS:
                T = base * k
                truly = any(abs(v/T - 1) <= TOL for v in L.values())
                cand = []
                for s in srvs:
                    try: pd_ = M.predict(w["cost"], w["steps"], s, 2, ag)
                    except Exception: continue
                    if pd_ and pd_ > 0:
                        cand.append((abs(math.log(to_wall(pd_, ag, effective_cores(s, 2, ag))/T)), s))
                if not cand: continue
                dmin, srv = min(cand)
                ok = dmin <= math.log(1+TOL)
                y = L.get(srv)
                if y is None: cat["推荐超出可实测范围"] += 1; continue
                hit = abs(y/T - 1) <= TOL
                cat["对: 推荐命中" if hit else ("对: 确实无解" if not truly else "错: 本来有解但没命中")] += 1
    return cat


if __name__ == "__main__":
    for tag, W, walls in (("老网格 b0.2", eval_old.build_old(), load_walls("old")),
                          ("rc3+rc4 b0 x5", eval_parts.build(), load_walls("new"))):
        print(f"\n{'='*58}\n{tag}   (score2_wall_r 口径, 留一库)\n{'='*58}")
        for cls, nm in ((PhoebePredictor, "现模型"), (NSPredictor, "改进 ns")):
            c = score(W, walls, cls); n = sum(c.values())
            if not n: print(f"  {nm}: 无数据"); continue
            good = sum(v for kk, v in c.items() if kk.startswith("对"))
            den = n - c["推荐超出可实测范围"]
            print(f"  【{nm}】 共 {n} 组")
            for kk, v in c.most_common(): print(f"      {kk:<22}{v:>5}{100*v/n:>7.1f}%")
            print(f"      {'达标率':<22}{good}/{den} = {100*good/max(den,1):.1f}%")
