#!/usr/bin/env python3
"""分项评估: 只看 sql 项 / 只看非 sql 项 / 合起来 db_time。

同一套留一库交叉验证、同样的 2 锚点(跨最低/最高并发)、锚点格点不计分。
calibrate 在 2 锚点下本来就把 sql 与非 sql 分开修正 (cal_sql_ / cal_ns_),
所以分项打分用的就是生产路径里那两个分量, 不是另做一套。
"""
import json, glob, os, re, math, collections, sys
import statistics as st
import numpy as np
import badpoints
from phoebe_predictor import PhoebePredictor, effective_cores, QP
from ns_model2 import NSPredictor

D = os.path.dirname(os.path.abspath(__file__)); NOM = badpoints.nominal_steps(); TOL = 0.15


def build():
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in ("rc3", "rc4"):
        for p in glob.glob(f"{D}/{run}/pdb_*/*/point.json"):
            lib, tag = p.split(os.sep)[-3], p.split(os.sep)[-2]
            m = re.match(r"s(\d+)_l4_t2_a(\d+)_b0_x\d+_k\d+$", tag)
            if not m: continue
            try: d = json.load(open(p))
            except Exception: continue
            if d.get("steps", 0) != NOM.get(lib, -1): continue
            bk = d.get("db_time_breakdown_s") or {}
            if (bk.get("sql", 0) * 1000.0 / max(d.get("sql_exec_n") or 1, 1)
                    > (d.get("sql_exec_p99_ms") or 1e18)): continue
            g[lib][(int(m[1]), int(m[2]))].append(dict(
                sql=bk.get("sql", 0),
                ns=bk.get("vec_mem", 0)+bk.get("vec_value", 0)+bk.get("tp_txn", 0),
                db=d["db_time_s"], **{f"q{q}": d.get(f"sql_exec_p{q}_ms") for q in QP}))
    lab = [x for x in json.load(open(f"{D}/train_label_noise.json")) if x.get("median", 0) > 0]
    c0 = collections.defaultdict(list)
    for x in lab: c0[int(x["qid"])].append(x["median"])
    qc = {q: float(np.mean(v)) for q, v in c0.items()}
    W = {}
    for lib, grid in g.items():
        ids = [q["question_id"] for q in json.load(open(f"{D}/{lib}_questions.json"))]
        cost = np.array([qc[q] for q in ids if q in qc])
        if not len(cost): continue
        W[lib] = dict(cost=cost, steps=NOM[lib], points=[
            dict(C=effective_cores(s, 2, a), ag=a, srv=s, th=2,
                 sql=st.mean([y["sql"] for y in v]), ns=st.mean([y["ns"] for y in v]),
                 db=st.mean([y["db"] for y in v]),
                 **{f"q{q}": st.mean([y[f"q{q}"] for y in v if y.get(f"q{q}")] or [0]) for q in QP})
            for (s, a), v in grid.items()])
    return W


def anchors(w):
    ags = sorted({p["ag"] for p in w["points"]})
    pick = lambda ag, pref: min([p for p in w["points"] if p["ag"] == ag],
                                key=lambda p: abs(p["srv"] - pref), default=None)
    return [p for p in (pick(ags[-1], 16), pick(ags[0], 8)) if p]


def run(cls, W):
    libs = sorted(W)
    E = {"sql": [], "ns": [], "db": []}
    R = {k: [0, 0] for k in E}
    per_lib = collections.defaultdict(list)
    for held in libs:
        w = W[held]; anc = anchors(w)
        A = [(a["C"], a["ag"], a["sql"], a["ns"]) for a in anc]
        M = cls().fit([W[l] for l in libs if l != held],
                      anchor_rows=PhoebePredictor.anchor_as_rows(w["cost"], w["steps"], A))
        M.calibrate(w["cost"], w["steps"], A)
        ks = float(np.exp(M.cal_sql_[0] * 0 + 0)) if hasattr(M, "cal_sql_") else 1.0
        def pred(p):
            s, n = M._parts(w["cost"], w["steps"], p["ag"], p["C"])
            if hasattr(M, "cal_sql_"):
                s *= float(np.exp(M.cal_sql_[0]*np.log(p["ag"]) + M.cal_sql_[1]))
                n *= float(np.exp(M.cal_ns_[0]*np.log(p["ag"]) + M.cal_ns_[1]))
            return s, n
        for p in w["points"]:
            if any(p["srv"] == a["srv"] and p["ag"] == a["ag"] for a in anc): continue
            s, n = pred(p)
            for k, pv, mv in (("sql", s, p["sql"]), ("ns", n, p["ns"]), ("db", s+n, p["db"])):
                if pv > 0 and mv > 0:
                    E[k].append(abs(pv/mv - 1)); per_lib[(k, held)].append(abs(pv/mv - 1))
        srvs = sorted({p["srv"] for p in w["points"]})
        for ag in sorted({p["ag"] for p in w["points"]}):
            pts = [p for p in w["points"] if p["ag"] == ag]
            if len(pts) < 4: continue
            for k in E:
                row = {p["srv"]: (p["sql"] if k == "sql" else p["ns"] if k == "ns" else p["db"]) for p in pts}
                base = st.median(list(row.values()))
                for mu in (0.6, 0.8, 1.0, 1.3, 1.8):
                    tgt = base*mu; R[k][1] += 1
                    cand = []
                    for s_ in srvs:
                        q = next((x for x in pts if x["srv"] == s_), None)
                        if not q: continue
                        sv, nv = pred(q)
                        pv = sv if k == "sql" else nv if k == "ns" else sv+nv
                        cand.append((abs(pv-tgt), s_))
                    if not cand: continue
                    b = min(cand)[1]
                    if b in row and abs(row[b]/tgt - 1) <= TOL: R[k][0] += 1
    return E, R, per_lib


if __name__ == "__main__":
    W = build()
    for cls, nm in ((PhoebePredictor, "现模型"), (NSPredictor, "改进 ns 模型")):
        E, R, PL = run(cls, W)
        print(f"\n【{nm}】  留一库交叉验证, 2 锚点")
        print(f"  {'目标量':<14}{'n':>5}{'误差中位':>10}{'p90':>9}{'≤15%':>8}{'推荐±15% 达标':>16}")
        for k, nm2 in (("sql", "只看 sql 项"), ("ns", "只看非 sql 项"), ("db", "db_time 合计")):
            e = E[k]; g, d = R[k]
            print(f"  {nm2:<14}{len(e):>5}{100*st.median(e):>9.1f}%{100*np.percentile(e,90):>8.1f}%"
                  f"{100*sum(1 for x in e if x<=TOL)/len(e):>7.0f}%{g:>9}/{d}{100*g/d:>6.1f}%")
    E, R, PL = run(NSPredictor, W)
    print(f"\n  只看 sql 项, 按库:")
    for l in sorted(W):
        v = PL[("sql", l)]
        if v: print(f"    {l:<30}{100*st.median(v):>7.1f}%")
