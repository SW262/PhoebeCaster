#!/usr/bin/env python3
"""把 eval_parts.py 的协议原封不动跑在【老网格】上, 用来分离"协议差异"与"数据差异"。

老网格: {lib}_grid/s*_l4_t2_a*_b0.2, 每配置 1 次, 2026-09-23~10-01。
除了数据来源, 留一库交叉验证 / 2 锚点 / 评分方式 与 eval_parts.py 逐行相。
"""
import json, glob, os, re, math, collections, sys
import statistics as st
import numpy as np
from phoebe_predictor import PhoebePredictor, effective_cores, QP
from ns_model2 import NSPredictor
import eval_parts

D = os.path.dirname(os.path.abspath(__file__)); TOL = 0.15


def build_old():
    lab = [x for x in json.load(open(f"{D}/train_label_noise.json")) if x.get("median", 0) > 0]
    c0 = collections.defaultdict(list)
    for x in lab: c0[int(x["qid"])].append(x["median"])
    qc = {q: float(np.mean(v)) for q, v in c0.items()}
    meta = json.load(open(f"{D}/pdb_meta.json"))
    W = {}
    for lib, m in meta.items():
        ids = [q["question_id"] for q in json.load(open(f"{D}/{lib}_questions.json"))]
        cost = np.array([qc[q] for q in ids if q in qc])
        if not len(cost): continue
        pts = []
        for p in glob.glob(f"{D}/{lib}_grid/s*/point.json"):
            mm = re.match(r"s(\d+)_l4_t2_a(\d+)_b0\.2$", os.path.basename(os.path.dirname(p)))
            if not mm: continue
            d = json.load(open(p)); bk = d.get("db_time_breakdown_s") or {}
            if not bk: continue
            if (bk.get("sql", 0) * 1000.0 / max(d.get("sql_exec_n") or 1, 1)
                    > (d.get("sql_exec_p99_ms") or 1e18)): continue
            pts.append(dict(C=effective_cores(int(mm[1]), 2, int(mm[2])), ag=int(mm[2]),
                            srv=int(mm[1]), th=2, sql=bk.get("sql", 0),
                            ns=bk.get("vec_mem", 0)+bk.get("vec_value", 0)+bk.get("tp_txn", 0),
                            db=d["db_time_s"],
                            **{f"q{q}": d.get(f"sql_exec_p{q}_ms") for q in QP}))
        if pts: W[lib] = dict(cost=cost, steps=m["steps"], points=pts)
    return W


if __name__ == "__main__":
    for tag, W in (("老网格 b0.2 2026-09-23~10-01", build_old()),
                   ("rc3+rc4 b0 x5 2026-10-10~14", eval_parts.build())):
        print(f"\n{'='*78}\n{tag}   {len(W)} 库 {sum(len(w['points']) for w in W.values)} 格点\n{'='*78}")
        for cls, nm in ((PhoebePredictor, "现模型"), (NSPredictor, "改进 ns")):
            E, R, _ = eval_parts.run(cls, W)
            print(f"  【{nm}】 {'目标量':<12}{'误差中位':>9}{'p90':>8}{'≤15%':>7}{'推荐±15%':>13}")
            for k, n2 in (("sql", "只看 sql"), ("ns", "只看非sql"), ("db", "db_time")):
                e = E[k]; g, d = R[k]
                print(f"           {n2:<12}{100*st.median(e):>8.1f}%{100*np.percentile(e,90):>7.1f}%"
                      f"{100*sum(1 for x in e if x<=TOL)/len(e):>6.0f}%{g:>7}/{d}{100*g/d:>6.1f}%")
        sh = st.median([p["sql"]/p["db"] for w in W.values() for p in w["points"]])
        print(f"  sql 占 db_time 中位 {100*sh:.0f}%")
