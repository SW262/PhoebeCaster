#!/usr/bin/env python3
"""网页 demo 的三个解释面板用的数据, 与 advise.py 走同一条生产路径(phoebe_cli._load)。

  python3 demo_panels.py --workload test2 --agents 64 [--max-cores 16]   -> JSON 到 stdout

输出:
  pipeline  测量流水线: 探针(查询数/离线代价)、两个锚点(实测墙钟/db_time/分量)、
            标定出的 (κ0, κ1)(SQL 与非SQL 各一对)、三条护栏的值与是否通过
  sweep     每个实测核档(HT 开/关): 两锚点预测墙钟、零实测预测墙钟、实测墙钟,
            以及预测分量(SQL / 非SQL, db_time 秒)与实测四分量(db_time 秒)
所有数都是现算或读实测文件, 不写死。
"""
import argparse, io, json, os, sys, importlib.util
from types import SimpleNamespace
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
from paths import DATA as D
TRACE = {"bal_test": "test_trace.json", "test2": "test2_trace.json"}
RUNGS = [2, 4, 6, 8, 11, 13, 16]
MAX_K0 = 0.30          # 护栏 (i), 与 phoebe_predictor.MAX_EXP_CORRECTION 一致
MIN_RHO = 0.5          # 护栏 (ii)
MAX_FAIL = 0.20        # 护栏 (iii)


def _imp(n):
    sp = importlib.util.spec_from_file_location(n, os.path.join(SRC, n + ".py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workload", choices=list(TRACE), required=True)
    p.add_argument("--agents", type=int, required=True)
    p.add_argument("--max-cores", type=int, default=16)
    a = p.parse_args()
    os.environ.setdefault("ADVISE_LANG", "en")
    adv_mod = _imp("advise"); cli = _imp("phoebe_cli"); pp = _imp("phoebe_predictor")
    off = os.path.join(D, f"offline_{a.workload}_v2.json")
    anc = os.path.join(D, f"anchors_{a.workload}_v2.json")
    e = sys.stderr; sys.stderr = io.StringIO()
    try:
        adv = cli._load(SimpleNamespace(offline=off, anchors=anc, zero_shot_quantile=0.0, anchor_quantile=None))
        adv0 = cli._load(SimpleNamespace(offline=off, anchors=None, zero_shot_quantile=0.0, anchor_quantile=None))
    finally:
        sys.stderr = e
    m = adv.m
    O = json.load(open(off)); A = json.load(open(anc))
    cost = np.array(O["cost_ms"], float); steps = adv.steps
    BK = getattr(_imp("phoebe_advisor"), "BIAS_K", 1.0)

    # ── 流水线 ──
    # 生产路径默认对总量标定(cal_, 一对 κ0/κ1, 两分量同乘); COMP_CAL=1 时才分量各自标定
    cal = {}
    if getattr(m, "cal_", None) is not None: cal["total"] = [float(m.cal_[0]), float(m.cal_[1])]
    for tag in ("sql", "ns"):
        c = getattr(m, f"cal_{tag}_", None)
        if c is not None: cal[tag] = [float(c[0]), float(c[1])]
    anchors = []
    for x in A:
        C = pp.effective_cores(x["srv"], x.get("th", 2), x["agents"])
        pt = os.path.join(D, x["source"], "point.json") if x.get("source") else None
        bd = json.load(open(pt)).get("db_time_breakdown_s") if pt and os.path.exists(pt) else None
        anchors.append(dict(srv=x["srv"], th=x.get("th", 2), agents=x["agents"], eff_cores=C,
                            rho=x["agents"] / C, wall_s=x["wall_s"], db_time_s=x["db_time_s"],
                            sql_s=x.get("sql_s"), ns_s=x.get("ns_s"), breakdown=bd))
    tr = json.load(open(os.path.join(D, TRACE[a.workload])))
    nv = sum(1 for q in tr if not any(r.get("sql") and not str(r.get("outcome", "")).startswith("ERROR")
                                      for r in q["rounds"]))
    lo = min(anchors, key=lambda z: z["agents"])
    k0s = [abs(v[0]) for v in cal.values() if v]
    guards = [
        dict(id="i", name="Anchors agree", value=max(k0s) if k0s else None, op="≤", threshold=MAX_K0,
             passed=bool(k0s) and max(k0s) <= MAX_K0 and not getattr(m, "cal_warn_", None), unit="|κ₀|",
             note="exponent correction fitted by the two anchors"),
        dict(id="ii", name="Low anchor loaded", value=lo["rho"], op="≥", threshold=MIN_RHO,
             passed=lo["rho"] >= MIN_RHO, unit="ρ = N/C",
             note=f"lower anchor: {lo['agents']} agents on {lo['eff_cores']:.1f} effective cores"),
        dict(id="iii", name="Queries valid", value=nv / len(tr), op="≤", threshold=MAX_FAIL,
             passed=nv / len(tr) <= MAX_FAIL, unit="share of questions with no valid query",
             note=f"{nv} of {len(tr)} questions"),
    ]
    probe = dict(n_queries=len(cost), sum_cost_ms=float(cost.sum()), median_cost_ms=float(np.median(cost)),
                 max_cost_ms=float(cost.max()), steps=int(steps), repeats=6,
                 offline_cost_s=float(cost.sum()) * 6 / 1000.0)

    # ── 扫描: 预测分量 / 零实测 / 实测 ──
    sweep = []
    for th, lab in ((2, "on"), (1, "off")):
        for srv in RUNGS:
            if srv > a.max_cores: continue
            C = pp.effective_cores(srv, th, a.agents)
            s_, n_ = m._parts(cost, steps, a.agents, C)
            f = lambda k: float(np.exp(cal[k][0] * np.log(a.agents) + cal[k][1]))
            if "sql" in cal and "ns" in cal: s_, n_ = s_ * f("sql"), n_ * f("ns")
            elif "total" in cal: s_, n_ = s_ * f("total"), n_ * f("total")
            db = adv.db_time(srv, th, a.agents)
            db0 = adv0.db_time(srv, th, a.agents)
            row = dict(ht=lab, srv=srv, pred=adv_mod.to_wall(db, a.agents, C),
                       pred_zero=adv_mod.to_wall(db0, a.agents, C),
                       pred_db=db, pred_sql_db=BK * s_, pred_ns_db=BK * n_,
                       measured=adv_mod.measured(a.workload, srv, th, a.agents), meas_breakdown=None)
            pt = os.path.join(D, adv_mod.MEAS_V2[a.workload], f"s{srv}_l4_t{th}_a{a.agents}_b0", "point.json")
            if os.path.exists(pt):
                d = json.load(open(pt))
                if d.get("steps") == d.get("passes", 0) * d.get("questions", 0):
                    row["meas_breakdown"] = d.get("db_time_breakdown_s")
                    row["meas_db"] = d.get("db_time_s")
            sweep.append(row)
    prior = dict(n_workloads=len(json.load(open(os.path.join(D, "train_set_v2.json")))["workloads"]))
    print(json.dumps(dict(workload=a.workload, agents=a.agents, probe=probe, prior=prior, anchors=anchors,
                          calibration=cal, calibration_discarded=bool(getattr(m, "cal_warn_", None)), guards=guards, sweep=sweep, rungs=[r for r in RUNGS if r <= a.max_cores]),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
