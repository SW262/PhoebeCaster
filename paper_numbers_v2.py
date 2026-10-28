#!/usr/bin/env python3
"""demo 论文的全部数字, 一条命令重算(2026-10-15, 口径 = v2)。

    python3 paper_numbers_v2.py            # 打印 + 写 paper_numbers_v2.json

口径(全文统一, 不与老网格/老测试集混报, 见 TODO #11):
  训练  rc3+rc4(b0, 11 个训练库, 只有超线程开), eval_parts.build
  模型  v2 = SQLPredictor(SQL 成本指数随 ρ 变 + 非SQL 核指数随并发变)
  测试  testv2(b0, 与训练同定义, 196 点; bal_test / test2 两个测试负载)
  锚点  16核/HT开/224 并发 + 8核/HT开/8 并发, 取 testv2 实测
  标定  与生产(phoebe_cli._load, COMP_CAL 未设)一致: 锚点并进训练行 + τ 地板(_attach_tau) + 联合标定总量
        (prereg_unify.json)。check_prod_v2.py 证实本口径 = 工具实际输出(189/224)。
  达标  |实测墙钟 − 目标| ≤ 15%, 双向
"""
import json, glob, os, re, math, collections, statistics as st
import numpy as np
from phoebe_predictor import PhoebePredictor, effective_cores
from sql_model2 import SQLPredictor
import eval_parts, eval_testv2, score_rc
import importlib.util, io, sys
from types import SimpleNamespace
_sp = importlib.util.spec_from_file_location("phoebe_cli", os.path.join(os.path.dirname(os.path.abspath(__file__)), "phoebe_cli.py"))
cli = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(cli)
WLF = {"test": "bal_test", "test2": "test2"}

D = os.path.dirname(os.path.abspath(__file__))
TOL = 0.15
ANC = ((16, 2, 224), (8, 2, 8))
OUT = collections.OrderedDict()
med = lambda xs: float(np.median(xs)) if len(xs) else float("nan")


def put(key, val, note=""):
    OUT[key] = dict(value=val, note=note)
    v = f"{val:.4g}" if isinstance(val, float) else str(val)
    print(f"  {key:<44s} {v:>14s}   {note}")


def load_test(wl):
    """testv2 每个点: 分量、墙钟、逐查询耗时。只收 steps == passes*questions 的完整点。"""
    P = {}
    for p in glob.glob(f"{D}/testv2/{wl}/s*/point.json"):
        m = re.match(r"s(\d+)_l4_t(\d)_a(\d+)_b0$", os.path.basename(os.path.dirname(p)))
        if not m: continue
        d = json.load(open(p))
        if d["steps"] != d["passes"] * d["questions"]: continue
        P[(int(m[1]), int(m[2]), int(m[3]))] = d
    return P


def fitted(wl, train, cls, anchors=True):
    cost, steps, pts = eval_testv2.build(wl)
    A = [(pts[k]["C"], pts[k]["ag"], pts[k]["sql"], pts[k]["ns"]) for k in ANC]
    if anchors:
        M = cls().fit(list(train.values()), anchor_rows=PhoebePredictor.anchor_as_rows(cost, steps, A))
        # τ 地板: 与生产同一个函数、同一顺序(先装地板再标定)
        e = sys.stderr; sys.stderr = io.StringIO()
        try: cli._attach_tau(SimpleNamespace(m=M), SimpleNamespace(
                 anchors=f"{D}/anchors_{WLF[wl]}_v2.json", offline=f"{D}/offline_{WLF[wl]}_v2.json"))
        finally: sys.stderr = e
        M.calibrate(cost, steps, [(c, a, s + n) for c, a, s, n in A])   # 联合标定, 同生产
    else:
        M = cls().fit(list(train.values()))
    return M, cost, steps, pts


def train_points():
    """rc3+rc4 原始点(含墙钟与分量), 用于数据集规模、锚点成本、分量膨胀。"""
    NOM = eval_parts.NOM
    R = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in ("rc3", "rc4"):
        for p in glob.glob(f"{D}/{run}/pdb_*/*/point.json"):
            lib, tag = p.split(os.sep)[-3], p.split(os.sep)[-2]
            m = re.match(r"s(\d+)_l4_t2_a(\d+)_b0_x\d+_k\d+$", tag)
            if not m: continue
            try: d = json.load(open(p))
            except Exception: continue
            if d.get("steps", 0) != NOM.get(lib, -1): continue
            R[lib][(int(m[1]), int(m[2]))].append(d)
    return R


def amp_q(M, c, cost, steps, ag, C):
    """v2 SQL 放大 amp(c_i, B, N, C), B 用整个负载的平均离线代价(_amp_vec 同式)。"""
    b = M.ba2_; lo = np.log(np.maximum(np.asarray(c, float), 1e-12))
    la, lc = math.log(float(ag)), math.log(float(C))
    lB = math.log(max(float(np.mean(cost)) * steps / 1000.0, 1e-12))
    return np.exp(b[0]*lo + b[1]*lo*(la-lc) + b[2]*lB + b[3]*la + b[4]*lc + b[5])


def comp(d):
    b = d["db_time_breakdown_s"]; s = d["steps"]
    return {k: b.get(k, 0) / s * 1000 for k in ("vec_mem", "vec_value", "sql", "tp_txn")}  # ms/步


if __name__ == "__main__":
    TRAIN = eval_parts.build()
    RAW = train_points()
    T = {wl: load_test(wl) for wl in ("test", "test2")}

    print("\n[数据集]")
    put("train.workloads", len(TRAIN), "rc3+rc4 有好点的训练库")
    put("train.runs", sum(len(v) for g in RAW.values() for v in g.values()), "rc3+rc4 完整运行(含重复)")
    put("train.configs", sum(len(g) for g in RAW.values()), "rc3+rc4 不同(库,核,并发)配置")
    put("test.workloads", len(T), "bal_test, test2")
    put("test.runs", sum(len(v) for v in T.values()), "testv2 完整点")
    put("test.questions.bal_test", T["test"][ANC[0]]["questions"], "point.json questions")
    put("test.questions.test2", T["test2"][ANC[0]]["questions"], "")
    put("test.steps.bal_test", T["test"][ANC[0]]["steps"], "")
    put("test.steps.test2", T["test2"][ANC[0]]["steps"], "")
    steps_tr = sorted(eval_parts.NOM[l] for l in TRAIN)
    put("train.steps.min", steps_tr[0], ""); put("train.steps.max", steps_tr[-1], "")
    put("train.steps.median", med(steps_tr), "")

    print("\n[前向预测: testv2, 锚点以外的配置]")
    for nm, cls in (("v2", SQLPredictor), ("v1model", PhoebePredictor)):
        for anc in (True, False):
            ew, ed, n, ewl = [], [], 0, {}
            for wl in ("test", "test2"):
                M, cost, steps, pts = fitted(wl, TRAIN, cls, anc); e1 = []
                for k, p in pts.items():
                    if k in ANC: continue
                    pr = M.predict(cost, steps, *k)
                    ed.append(abs(pr / p["db"] - 1))
                    e1.append(abs(score_rc.to_wall(pr, p["ag"], p["C"]) / p["wall"] - 1))
                ew += e1; ewl[wl] = e1
            tag = f"fwd.{nm}.{'2anchor' if anc else 'zeroshot'}"
            for wl in ewl: put(f"{tag}.wall_median_pct.{wl}", 100 * med(ewl[wl]), "")
            put(f"{tag}.n", len(ew), "")
            put(f"{tag}.wall_median_pct", 100 * med(ew), "墙钟完成时间误差中位")
            put(f"{tag}.wall_within15_pct", 100 * float(np.mean(np.array(ew) <= TOL)), "墙钟误差 ≤15% 的占比")
            put(f"{tag}.db_median_pct", 100 * med(ed), "db_time 误差中位")

    print("\n[反推: 224 个目标, 严格口径, 与生产同路径]")
    RUNGS = (2, 4, 6, 8, 11, 13, 16)
    for nm, cls in (("v2", SQLPredictor), ("v1model", PhoebePredictor)):
        ok = n = truly_n = 0; per = {}
        for wl in ("test", "test2"):
            M, cost, steps, P = fitted(wl, TRAIN, cls, True)
            base = P[(16, 2, 224)]["wall"]; o = m = 0
            for th in (2, 1):
                for ag in eval_testv2.AGS:
                    L = {s_: P[(s_, th, ag)]["wall"] for s_ in RUNGS if (s_, th, ag) in P}
                    if len(L) < 4: continue
                    for k in score_rc.KS:
                        Tg = base * k
                        truly = any(abs(v / Tg - 1) <= TOL for v in L.values())
                        c = [(abs(score_rc.to_wall(M.predict(cost, steps, s_, th, ag), ag,
                                                   effective_cores(s_, th, ag)) / Tg - 1), s_) for s_ in L]
                        d, srv = min(c); mf = d <= TOL; hit = abs(L[srv] / Tg - 1) <= TOL
                        m += 1; o += (truly and mf and hit) or (not truly and not mf); truly_n += not truly
            per[wl] = 100 * o / m; ok += o; n += m
        put(f"inv.{nm}.ok", ok, "")
        put(f"inv.{nm}.n", n, "目标数")
        put(f"inv.{nm}.strict_pct", 100 * ok / n, "有解时推荐档实测落±15%; 无解时须自己说无解")
        put(f"inv.{nm}.strict_bal_test_pct", per["test"], ""); put(f"inv.{nm}.strict_test2_pct", per["test2"], "")
        put(f"inv.{nm}.truly_infeasible", truly_n, "网格里本来就没有配置落进 ±15%")

    print("\n[逐查询 Q-error: testv2, v2; 同 qerror_perquery.py 口径]")
    # 真值 = sql_ms_by_qid[qid] (该题本次运行全部执行的延迟中位); 预测 = c_i·amp·SQL 标定系数;
    # c_i 取 offline_byqid_*.json (离线探针逐题代价)。真值或预测为 0 的跳过(该题本次全部出错)。
    allq, cfgmed, per_ag = [], {}, collections.defaultdict(list)
    for wl, off in (("test", "bal_test"), ("test2", "test2")):
        M, cost, steps, pts = fitted(wl, TRAIN, SQLPredictor, True)
        qc = json.load(open(f"{D}/offline_byqid_{off}.json"))["cost_by_qid"]
        for k, d in T[wl].items():
            if k in ANC: continue
            s, th, a = k; C = effective_cores(s, th, a)
            KS = getattr(M, "cal_", None)                                 # 联合标定系数
            f = float(np.exp(KS[0] * math.log(a) + KS[1])) if KS is not None else 1.0
            byq = d.get("sql_ms_by_qid") or {}
            q = [x for x in byq if str(x) in qc and byq[x] > 0 and qc[str(x)] > 0]
            if not q: continue
            c = np.array([qc[str(x)] for x in q], float)
            pred = c * amp_q(M, c, cost, steps, a, C)
            if getattr(M, "tau_", None) is not None:                  # 同 _parts: 先取地板再乘标定
                pred = np.maximum(pred, M.tau_[0] * float(a) ** M.tau_[1])
            pred = pred * f
            meas = np.array([byq[x] for x in q], float)
            qe = np.maximum(pred / meas, meas / pred)
            allq += list(qe); per_ag[a] += list(qe); cfgmed[f"{wl}/s{s}t{th}a{a}"] = float(np.median(qe))
    put("qerr.n_query_config", len(allq), "(查询, 配置) 对")
    put("qerr.n_configs", len(cfgmed), "")
    put("qerr.median", med(allq), "")
    put("qerr.p90", float(np.percentile(allq, 90)), "")
    cm = sorted(cfgmed.values())
    put("qerr.config_median.min", cm[0], ""); put("qerr.config_median.max", cm[-1], "")
    for a in sorted(per_ag): put(f"qerr.median.ag{a}", med(per_ag[a]), f"n={len(per_ag[a])}")

    print("\n[分量膨胀: 16核/HT开, 8 -> 224 并发, 每步耗时之比]")
    def dil(d8, d224):
        a, b = comp(d8), comp(d224)
        return {k: b[k] / a[k] for k in a if a[k] > 0}
    rows = {}
    for wl in ("test", "test2"):
        rows[f"test:{wl}"] = dil(T[wl][(16, 2, 8)], T[wl][(16, 2, 224)])
    for lib, g in RAW.items():
        if (16, 8) in g and (16, 224) in g:
            mean = lambda L: {k: st.mean(comp(x)[k] for x in L) for k in ("vec_mem", "vec_value", "sql", "tp_txn")}
            a, b = mean(g[(16, 8)]), mean(g[(16, 224)])
            rows[f"train:{lib}"] = {k: b[k] / a[k] for k in a if a[k] > 0}
    for k, r in rows.items():
        spread = max(r.values()) / min(r.values())
        print(f"    {k:<34s} " + "  ".join(f"{c}={v:5.1f}x" for c, v in r.items()) + f"   最快/最慢={spread:5.1f}x")
        OUT[f"dil.{k}"] = dict(value={**r, "spread": spread}, note="")
    sp = [max(r.values()) / min(r.values()) for r in rows.values()]
    put("dil.spread.min", min(sp), "各负载内 最快/最慢 分量膨胀之比")
    put("dil.spread.max", max(sp), "")
    put("dil.spread.median", med(sp), "")
    # european_football_2: 64 并发起 SQL 出错从 1300 升到 ~1490 条(最重的查询超时), SQL 每步耗时
    # "变快"到 0.8x 是污染不是真实膨胀(TODO), 论文的膨胀范围剔除它并注明。
    sp2 = [max(r.values()) / min(r.values()) for k, r in rows.items() if "european_football_2" not in k]
    put("dil.spread.min_excl_efb2", min(sp2), "剔除 european_football_2")
    put("dil.spread.max_excl_efb2", max(sp2), "剔除 european_football_2")
    put("dil.n_excl_efb2", len(sp2), "")

    print("\n[锚点成本: 两次锚点运行的墙钟之和]")
    cost_min = {}
    for wl in ("test", "test2"):
        cost_min[f"test:{wl}"] = sum(T[wl][k]["completion_seconds"] for k in ANC) / 60
    for lib, g in RAW.items():
        if (16, 224) in g and (8, 8) in g:
            cost_min[f"train:{lib}"] = (st.mean(x["completion_seconds"] for x in g[(16, 224)])
                                        + st.mean(x["completion_seconds"] for x in g[(8, 8)])) / 60
    for k, v in sorted(cost_min.items(), key=lambda x: x[1]): print(f"    {k:<34s} {v:6.2f} min")
    put("anchor.workloads", len(cost_min), "")
    put("anchor.minutes.median", med(list(cost_min.values())), "")
    put("anchor.minutes.max", max(cost_min.values()), "")
    put("anchor.minutes.test_max", max(cost_min[f"test:{w}"] for w in ("test", "test2")), "")

    print("\n[SQL / 非SQL 跨负载跨度: 16核x224 每步耗时 max/min]")
    sq, ns = [], []
    for lib, g in RAW.items():
        if (16, 224) in g:
            c = [comp(x) for x in g[(16, 224)]]
            sq.append(st.mean(x["sql"] for x in c)); ns.append(st.mean(x["vec_mem"]+x["vec_value"]+x["tp_txn"] for x in c))
    put("span.sql", max(sq) / min(sq), "训练库, 16核x224")
    put("span.nonsql", max(ns) / min(ns), "")

    print("\n[离线代价 vs SQL 占比: 训练 + 测试, 16核x224]")
    xs, ys = [], []
    for lib, w in TRAIN.items():
        g = RAW[lib].get((16, 224))
        if not g: continue
        c = [comp(x) for x in g]; tot = [sum(x.values()) for x in c]
        xs.append(math.log(float(np.mean(w["cost"])))); ys.append(st.mean(x["sql"] / t for x, t in zip(c, tot)))
    for wl in ("test", "test2"):
        cost, _, _ = eval_testv2.build(wl); c = comp(T[wl][(16, 2, 224)])
        xs.append(math.log(float(np.mean(cost)))); ys.append(c["sql"] / sum(c.values()))
    put("costshare.n", len(xs), "负载数")
    put("costshare.r_logcost", float(np.corrcoef(xs, ys)[0, 1]), "log(平均离线代价) 与 SQL 占比的 Pearson r")

    print("\n[模型]")
    put("model.params.sql", 6, "log amp = b0·log c + b1·log c·log(N/C) + b2·log B + b3·log N + b4·log C + b5")
    put("model.params.nonsql", 6, "log ns = 二次曲面 in (log N, log C)")
    put("model.params.calib_per_component", 2, "每个分量 κ0·log N + κ1")

    json.dump(OUT, open(f"{D}/paper_numbers_v2.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n写入 paper_numbers_v2.json ({len(OUT)} 项)")
