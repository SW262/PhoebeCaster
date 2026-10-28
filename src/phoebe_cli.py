#!/usr/bin/env python3
"""PhoebeDB 资源顾问 —— 端到端入口。

客户侧三步, 总计约 20 分钟实测:

  步骤 1  量离线成本(约 9 分钟)
      python3 phoebe_cli.py measure-offline --dsn <DSN> --sql-file queries.txt \
              --out offline.json
      对每条 SQL 串行跑 6 次取中位(丢第一次), 产出逐条稳态耗时。
      必须是稳态中位, 不能用单次值 —— 单次值系统性偏慢 20.3%,
         曾因此让整个跨库评估失效 8 个百分点。

  步骤 2  跑 2 个锚点
      在客户自己的库上跑两个配置, 一个低并发、一个最高并发。两档可选:

        档位     低锚点        高锚点        墙钟   前瞻集      bal_test     最差
                                              中位       中位
        快速档  ag16/最大机器  ag224/最大机器  178s   3.4%      6.2%      33.5%
        稳健档  ag8 /最小机器  ag224/最小机器 1055s   9.8%      8.1%      20.9%

      快速档中位最好、墙钟省 6 倍; 稳健档中位差一倍但尾部最好
      (最差 20.9% vs 33.5%, ±20% 覆盖 97% vs 76%)。不能接受长尾就选稳健档。
      两档都在两个独立评估集上验证过(16 个前瞻点 + 36 个 bal_test 点), 。

      真正的失效条件: 低锚点的负载强度 ρ = 并发 / 有效核数 落进 [0.5, 3]。
         ρ<0.5  几乎空载, 放大倍数≈1, 测不到拥塞 —— ag8/C19.8(ρ=0.40)两个集上
                都是最差: 前瞻 11.7%, bal_test 24.1%。
         ρ>4    进入饱和, 也变差(ρ=6.45 -> 13.1%, ρ=8.0 -> 21.2%)。
         误差对 ρ 是 U 型, 两个评估集的最优点一致。
         注意这推翻了  写的"低锚点用最小机器、目标 ρ≈3~12" ——
            那条是只在训练库留一上选出来的, 没有转移。优先把核数开到最大。

  步骤 3  问
      python3 phoebe_cli.py advise --offline offline.json --anchors anchors.json \
              --target-seconds 3000 --agents 64
      或 --isoline / --max-agents

精度(两套口径, 对外报保守的那个):
  test 4 个陌生库 / 36 点:      零实测 40.9%, 2 锚点中位 8.2%, ±20% 83.4%
  11 个单库负载留一 / 各 12 点:  零实测 22.4%, 2 锚点中位 21.0%, 最差 30.2%
  后者点少、锚点占掉 1/6, 更脆弱。对外应报 21%, 不是 8.2%。
  反推(选型)比点预测稳: 推荐配置 94~98% 实际达标(≤1.2× 目标)。

坏锚点护栏: 若两个锚点解出的指数修正 |cal| > 0.30 即判为不自洽, 完全不校正。
  依据: 相关 r=+0.882; 一个坏锚点曾把误差从 31% 放大到 129%(退回常数校正更糟, 177%)。
"""
import argparse, json, os, sys, time
import numpy as np


def cmd_measure_offline(a):
    import psycopg2
    sqls = [l.strip() for l in open(a.sql_file) if l.strip() and not l.startswith("--")]
    print(f"待测 {len(sqls)} 条, 每条 {a.repeats} 次取中位(丢第一次)", file=sys.stderr)
    c = psycopg2.connect(a.dsn); c.autocommit = True
    cur = c.cursor(); cur.execute(f"SET statement_timeout={a.timeout_ms}")
    out, bad = [], 0
    t0 = time.time()
    for i, s in enumerate(sqls):
        ts = []
        try:
            for _ in range(a.repeats):
                t = time.perf_counter(); cur.execute(s); cur.fetchall()
                ts.append((time.perf_counter() - t) * 1000.0)
        except Exception:
            bad += 1
            continue
        out.append(float(np.median(ts[1:] or ts)))       # 丢第一次(冷启动)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(sqls)}  {time.time()-t0:.0f}s", file=sys.stderr)
    json.dump(dict(cost_ms=out, n_sql=len(sqls), failed=bad,
                   steps_hint=a.steps), open(a.out, "w"))
    print(f"写出 {a.out}: {len(out)} 条成功 / {bad} 条失败, 用时 {time.time()-t0:.0f}s", file=sys.stderr)
    if not out:
        sys.exit("!! 全部失败, 无法继续")
    print(f"  离线总量 B = {np.mean(out)*a.steps/1000:.1f}s (按 {a.steps} 步)", file=sys.stderr)
    # 失败的查询被静默丢弃会让 B 被低估, 而 B 是模型唯一的负载侧输入 —— 必须吼出来
    fr = bad / max(len(sqls), 1)
    if fr > 0.20:
        print(f"!! 失败率 {100*fr:.0f}% 过高 —— B 会被低估约 {100*fr:.0f}%, 预测随之偏低。"
              f"\n   请先修掉失败的 SQL 再来, 否则推荐的配置会不够用。", file=sys.stderr)
    elif fr > 0.05:
        print(f"!! 失败率 {100*fr:.0f}% —— B 可能被低估, 建议核对失败的 SQL。", file=sys.stderr)
    # 成本分布的形状直接决定预测质量, 打出来供核对
    q = np.percentile(out, [50, 90, 99])
    print(f"  逐条耗时: 中位 {q[0]:.2f}ms  p90 {q[1]:.2f}ms  p99 {q[2]:.2f}ms  "
          f"偏度(p99/中位) {q[2]/max(q[0],1e-9):.1f}", file=sys.stderr)


# 2026-10-09(prereg_unify.json md5 5f4b3350): 锚点标定改回联合标定。
#    当初选"分量各自标定", 是在 τ 地板(2026-10-07, phoebe_predictor.py:194)之前测的。
#   装了地板之后, 分量标定在 224 个预注册目标上 89.3% -> 74.2%, 正向中位 5.9% -> 10.7%,
#   bal_test 正向中位更是 26.6%、有符号 -20.0%, 并产生 38 个"推荐超出可实测范围"。
#   原因: 两个分量各带一条自己倾斜的幂律, 离开锚点后发散得比单条快, 而地板又只压 SQL 一侧。
#   COMP_CAL=1 可复现  的旧行为。
_USE_COMP_CAL = os.environ.get("COMP_CAL") == "1"


def _en():
    """输出语言。默认英文(VLDB demo); ADVISE_LANG=zh 恢复中文。"""
    return os.environ.get("ADVISE_LANG", "en") != "zh"

def _load(a):
    import importlib.util, os
    d = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, d)
    from paths import DATA as _DATA
    def imp(n):
        sp = importlib.util.spec_from_file_location(n, os.path.join(d, n + ".py"))
        m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
    pp, pa = imp("phoebe_predictor"), imp("phoebe_advisor")
    off = json.load(open(a.offline))
    cost = np.array(off["cost_ms"], float); steps = off.get("steps_hint") or pp.STEPS_REF
    # 2026-10-14 切到 v2: 训练集 = rc3+rc4 (train_set_v2.json), 模型 = sql_model2.SQLPredictor
    #   (非SQL 核指数随并发变 + SQL 成本指数随 ρ=ag/C 变)。依据见 MODEL_IMPROVEMENT.md:
    #   两测试集推荐成功率 81.7% -> 90.0%, 测试集预测误差中位 5.7% -> 5.0%。
    #   PHOEBE_MODEL=v1 原样复现老结果(老网格 + PhoebePredictor)。
    # 2026-10-14 默认退回 v1: 在两个测试集上 v1 仍最好(生产口径 89.3% vs v2 83.0%,
    #   严格口径 92.5% vs 90.0%)。测试集是老负载定义(b0.2, 不放大)采的, 与老网格同定义、
    #   与 rc3+rc4 不同 —— 新数据训练在它们上面天然吃亏。v2 / v1m3 保留为开关。
    # 2026-10-15 默认改回 v2: 在新定义测试集 testv2(rc3+rc4 同定义)上 v2 最好 ——
    #   严格 85.3% vs v1 82.6%, score2 98.2% vs 96.0%(eval_testv2.py)。论文统一用此口径。
    _V = os.environ.get("PHOEBE_MODEL", "v2")
    if _V == "v1":
        WLS = pp.load_repo_workloads(os.path.join(d)); _Cls = pp.PhoebePredictor
    elif _V == "v1m3":      # 老网格训练集 + 模型3 —— 对照用
        WLS = pp.load_repo_workloads(os.path.join(d)); _Cls = imp("sql_model2").SQLPredictor
    else:
        _tv = json.load(open(os.path.join(_DATA, "train_set_v2.json")))
        WLS = [dict(cost=np.array(w["cost"], float), steps=w["steps"], points=w["points"])
               for w in _tv["workloads"]]
        _Cls = imp("sql_model2").SQLPredictor
    # 诊断用开关(2026-10-09): 把某个负载排出主干拟合。默认不启用, 生产行为不变。
    _drop = os.environ.get("DROP_WL")
    if _drop:
        import json as _j
        import badpoints as _bp
        _nm = [w for w in _j.load(open(os.path.join(_DATA, "pdb_meta.json"))) if w not in _bp.EXCLUDED_LIBS]
        _norm = lambda z: z[4:] if z.startswith("pdb_") else z
        _want = _norm(_drop)
        WLS = [x for _i, x in enumerate(WLS) if _norm(_nm[_i]) != _want]
        assert len(WLS) == len(_nm) - 1, f"DROP_WL={_drop} 未匹配到任何负载 ({len(WLS)}/{len(_nm)})"
    if a.anchors:
        A = json.load(open(a.anchors))
        anc = [(pp.effective_cores(x["srv"], x.get("th", 2), x["agents"]), x["agents"], x["db_time_s"]) for x in A]
        # : 先把锚点并进训练集重拟合(Hilprecht 官方实现的 few-shot 做法),
        #   再做事后两参数修正。两者叠加才有效 ——
        #   1 锚点: 仅 cal 35.7% / 仅 refit 28.7% / 叠加 19.5%(两个评估集一致);
        #   2 锚点: 中位持平但 ±10% 从 60.0% 升到 77.5%、最差 33.5%→26.5%。
        #   锚点若缺分量拆解(db_time_breakdown_s), 退回纯事后修正。
        arows = None
        if all(("sql_s" in x or "breakdown" in x) for x in A):
            ar = [(pp.effective_cores(x["srv"], x.get("th", 2), x["agents"]), x["agents"],
                   x.get("sql_s") or (x.get("breakdown") or {}).get("sql", 0),
                   x.get("ns_s") or sum((x.get("breakdown") or {}).get(k, 0)
                                        for k in ("vec_mem", "vec_value", "tp_txn")))
                  for x in A]
            # /: 只有恰好 1 个锚点时, 改用与训练行同形的分位行。
            #   单锚点的事后修正只能定常数(k0≡0), 分位行是唯一能约束成本指数的信息:
            #   出族实测 中位 11.7%→8.6%, p90 29.2%→19.8%, 最差 45.6%→29.5%, ±20% 68.8%→90.0%。
            #   两锚点不用它(k0 已修过指数, 再拉成本指数是重复修正, 实测 7.1%→9.6% 变差),
            #   所以 2 锚点及缺分位数时逐位走原路径。
            pctq = [{pc: x.get(f"sql_exec_p{pc}_ms") or (x.get("pct") or {}).get(pc)
                     for pc in pp.QP} for x in A]
            if len(A) == 1 and all(v for v in pctq[0].values()):
                arows = pp.PhoebePredictor.anchor_as_rows_pct(
                    cost, steps, [(c, a, s, n, pctq[0]) for (c, a, s, n) in ar])
                print(f"!! 单锚点已启用**分位行**(): p50/p95/p99 = "
                      f"{'/'.join(f'{pctq[0][pc]:.0f}ms' for pc in pp.QP)}。\n"
                      f"   实测中位误差 11.7% -> **8.6%**, 最差 45.6% -> 29.5%。", file=sys.stderr)
            else:
                arows = pp.PhoebePredictor.anchor_as_rows(cost, steps, ar)
                if len(A) == 1:
                    print("!! 提示: 锚点缺 sql_exec_p50/p95/p99_ms, 走原单行路径。"
                          "带上这三个分位可让单锚点从 11.7% 降到 8.6%() —— "
                          "采集脚本本来就产出, 直接填进来即可。", file=sys.stderr)
        else:
            print("!! 提示: 锚点未带分量拆解(sql_s/ns_s 或 breakdown), 退回纯事后修正。\n"
                  "   带上分量可让 1 锚点从 35.7% 降到 19.5%() —— "
                  "采集脚本本来就产出 db_time_breakdown_s, 直接填进来即可。", file=sys.stderr)
        M = _Cls().fit(WLS, anchor_rows=arows)
        ba = M.ba_.copy()
        aq = getattr(a, "anchor_quantile", None)
        ags = {x["agents"] for x in A}
        if aq is not None:
            if len(A) != 1:
                print("!! 忽略 --anchor-quantile: 只在**恰好 1 个锚点**时生效"
                      "(2 个锚点已能定出指数, 用不着悲观先验)。", file=sys.stderr)
            else:
                ia = pp.PhoebePredictor.FA.index("ag")
                ex = pp.workload_ag_exponents(WLS)
                old_e = ba[ia]; ba[ia] = float(np.quantile(ex, float(aq)))
                print(f"!! 已启用单锚点悲观指数 q={aq}: ag 指数 {old_e:.3f} -> {ba[ia]:.3f}"
                      f"(训练负载 {ex.min():.3f}~{ex.max():.3f})。\n"
                      f"   实测 bal_test: 中位误差 16.1% -> **7.5%**, p90 29.2% -> 26.7%, "
                      f"少用 3.84 个核; 但达标率 95.1% -> 90.2%()。\n"
                      f"   注意这是**降误差、减保守度**的取舍 —— 若你更怕配置买小了, 不要开。",
                      file=sys.stderr)
        # : 恰好 1 个锚点且带 db_blks_hit 时, 用它反推该负载的并发指数。
        #   单锚点的事后修正只能定常数(k0≡0), 指数一直只能用训练集的全局拟合值(0.759);
        #   而锚点那次运行本来就产出这个字段, 之前整个扔掉了。
        #   实测(两个出族集各一次): 1 锚点中位 11.7%->5.2% / 9.9%->8.9%,
        #   bal_test 上甚至优于现行 2 锚点(7.1%)。
        if len(A) == 1:
            bh = A[0].get("db_blks_hit") or (A[0].get("sys") or {}).get("db_blks_hit")
            if bh:
                pe, note = pp.anchor_exponent(bh, A[0]["srv"], A[0].get("th", 2),
                                              A[0]["agents"], os.path.join(d))
                if pe is not None:
                    ba[pp.PhoebePredictor.FA.index("ag")] = pe
                    print(f"!! 已用锚点系统指标反推并发指数: {note}\n"
                          f"   实测 1 锚点中位误差 11.7%->**5.2%**(bal_test) / 9.9%->**8.9%**(test2)。",
                          file=sys.stderr)
                else:
                    print(f"!! 未能从锚点反推并发指数({note}), 用训练集全局值。", file=sys.stderr)
            else:
                print("!! 提示: 锚点未带 db_blks_hit —— 带上它可让单锚点中位误差从 11.7% 降到 5.2%"
                      "()。采集脚本本来就产出该字段, 直接填进来即可。", file=sys.stderr)
        # 2026-10-14: 必须与拟合用的类一致, 并拷全部拟合参数(含 v2 的 ba2_/bn2_)。
        #   原来写死 PhoebePredictor 且只拷 ba_/bn_ —— 切 v2 时改进项会被静默丢掉。
        #   注: 单锚点的指数微调只改 ba_, 对 v2 的 SQL 放大项(ba2_)不起作用; 协议是两锚点, 不受影响。
        m = _Cls(); m.__dict__.update({k: v for k, v in M.__dict__.items() if k.endswith("_")})
        m.ba_, m.bn_ = ba, M.bn_
        if len(ags) < 2:
            if arows is not None:
                print("!! 提示: 只有一个并发档 —— 事后修正定不出并发指数(只能定常数), "
                      "但锚点已并进训练集重拟合。\n"
                      "   出族实测(81 配置, /): 单锚点中位 **8.6%**(带分位行)"
                      " / 11.7%(不带), 最差 29.5% / 45.6%。\n"
                      "   再加一个**不同并发档**的锚点 -> 中位 7.1%、±20% 命中 89.9%"
                      "(已接近两参数修正的天花板 7.5%, )。",
                      file=sys.stderr)
            else:
                print("!! 警告: 锚点并发档相同且未带分量拆解 —— 只能定常数, "
                      "实测中位误差 **35.7%**()。\n"
                      "   两条改进任选其一: 带上分量拆解(→19.5%), 或补一个不同并发档的锚点(→4.9%)。",
                      file=sys.stderr)
        # 负载强度检查(已修正  的错误阈值)。
        # 误差对低锚点的 rho 呈 U 型, 两个独立评估集一致:
        #   rho=0.40 -> 前瞻 11.7% / bal_test 24.1%   (空载, 测不到拥塞)
        #   rho=0.81 -> 前瞻  3.4% / bal_test  6.2%   最优
        #   rho=3.23 -> 前瞻  5.1% / bal_test  9.5%
        #   rho=6.45 -> 前瞻  7.2% / bal_test 13.1%   (开始饱)
        #   rho=8.00 -> 前瞻 11.9% / bal_test 21.2%
        lo = min(anc, key=lambda z: z[1])            # 并发最低的那个锚点
        rho = lo[1] / lo[0]
        if len(ags) < 2:
            # 单锚点用自己的规则, 不能套两锚点的 ρ∈[0.5,3]。
            #    实测(11 训练库 × 每库 12 个位置): 单锚点最好的位置是
            #   最高并发 × 最大机器(ρ≈11.3, 中位 15.8%), 最差是最低并发 × 最大机器
            #   (ρ=0.40, 63.7%)。两锚点的规则说的是"低并发那个锚点"该落哪, 与此无关。
            hi_ag, hi_C = max(a[1] for a in anc), max(a[0] for a in anc)
            if anc[0][1] < hi_ag or anc[0][0] < hi_C:
                pass
            if rho < 1.0:
                print(f"!! 警告: 单锚点 ρ={rho:.2f} < 1 —— 这台机器对该并发几乎空载, "
                      f"测不到拥塞。实测最低并发×最大机器(ρ=0.40)中位 63.7%, "
                      f"而最高并发×最大机器(ρ≈11)只有 15.8%()。\n"
                      f"   修法: **换成最高并发档 + 你能给的最大机器**。", file=sys.stderr)
        elif rho < 0.5:
            print((f"!! WARNING: the low-concurrency anchor ({lo[1]} agents, {lo[0]:.1f} effective "
                   f"cores) has rho={rho:.2f} < 0.5 -- this machine is nearly idle at that "
                   f"concurrency, amplification ~1, so it witnesses no congestion.\n"
                   f"   Measured error in this regime is 11.7%-24.1%, against 3.4%-6.2% at rho~0.8.\n"
                   f"   Fix: **raise this anchor's concurrency** (do not lower the cores) so that "
                   f"rho falls in [0.5, 3].") if _en() else
                  (f"!! 警告: 低并发锚点 (并发 {lo[1]}, 有效核 {lo[0]:.1f}) 的 ρ={rho:.2f} < 0.5 —— "
                   f"这台机器对这个并发几乎是空载, 放大倍数≈1, 测不到拥塞。\n"
                   f"   实测这一档误差 11.7%~24.1%, 而 ρ≈0.8 时是 3.4%~6.2%。\n"
                   f"   修法: **提高这个锚点的并发**(核数不要降), 让 ρ 落进 [0.5, 3]。"),
                  file=sys.stderr)
        elif rho > 4.0:
            print((f"!! WARNING: the low-concurrency anchor has rho={rho:.2f} > 4 -- already in "
                   f"saturation, where measured error rises to 13%-21%. Fix: move this anchor to a "
                   f"**larger machine** so that rho falls in [0.5, 3].") if _en() else
                  (f"!! 警告: 低并发锚点 ρ={rho:.2f} > 4 —— 已进入饱和区, 实测误差升到 "
                   f"13%~21%。修法: 把这个锚点挪到**更大的机器**上, 让 ρ 落进 [0.5, 3]。"),
                  file=sys.stderr)
        # : 锚点带分量实测时走分量各自标定
        anc4 = [(pp.effective_cores(x["srv"], x.get("th", 2), x["agents"]), x["agents"],
                 x.get("sql_s") or (x.get("breakdown") or {}).get("sql", 0),
                 x.get("ns_s") or sum((x.get("breakdown") or {}).get(k, 0)
                                      for k in ("vec_mem", "vec_value", "tp_txn")))
                for x in A]
        m.calibrate(cost, steps, anc4 if (_USE_COMP_CAL and all(s > 0 and n > 0 for _, _, s, n in anc4)) else anc)
    else:
        M = _Cls().fit(WLS)
        m = _Cls(); m.__dict__.update({k: v for k, v in M.__dict__.items() if k.endswith("_")})
        print("!! 提示: 无锚点 = 零实测。**预测误差约 40%**(15 个网格外配置的前瞻实测: 中位 39.8%),\n"
              "   而且偏差方向**不可预测** —— 两个陌生负载实测为 −42.1% 与 +18.9%()。\n"
              "   后果: 推荐的机器实测只用掉目标时间的约 66%, 即多买约 1.5 倍。\n"
              "   补 1 个锚点(最高并发 × 最大机器, **约 81 秒**)即可把预测误差降到 **10.0%**,\n"
              "     推荐配置贴到目标的 0.86x —— 前瞻实测里 6 次有 5 次直接命中'最小够用'的那台机器。\n"
              "   --zero-shot-quantile 0.9 曾是默认, 现已关闭: 它只在恰好被低估的负载上有用,\n"
              "     在被高估的负载上会把达标率从 70.8% 压到 33.3%、预测误差推到 150.6%()。",
              file=sys.stderr)
    adv = pa.PhoebeAdvisor(m, cost, steps)
    adv.m = m
    q = getattr(a, "zero_shot_quantile", None)
    if q is not None and float(q) > 0:      # 0 = 关闭(默认)
        if a.anchors:
            # 默认值是 0.9, 所以只有用户显式传了才提示, 否则是噪音。
            if any(x.startswith("--zero-shot-quantile") for x in sys.argv):
                print("!! 忽略 --zero-shot-quantile: 已有锚点, 悲观指数只用于**零实测**。",
                      file=sys.stderr)
        else:
            ex = pp.workload_ag_exponents(WLS)
            adv.set_zero_shot(ex, float(q), pp.PhoebePredictor.FA.index("ag"))
            print(f"!! 已启用零实测悲观指数 q={q}: 训练负载指数 "
                  f"{ex.min():.3f}~{ex.max():.3f}, 取 {float(__import__('numpy').quantile(ex,float(q))):.3f} "
                  f"(拟合值 {m.ba_[pp.PhoebePredictor.FA.index('ag')]:.3f})。\n"
                  f"   ρ 自适应余量已**自动关闭** —— 两者针对同一个误差来源, 叠加是重复计费()。\n"
                  f"   注意: q **不是**'把握度'。'名义 q ≈ 达标率'这个对应关系**不成立**\n"
                  f"   (: |实际−名义| 中位 0.134, 单库侧 q=0.5 时达标率就已 84.7%)。\n"
                  f"   q 只是**单调的保守程度旋钮**; 实际达标率见  的表, 不要按 q 解读成概率。",
                  file=sys.stderr)
    # 2026-10-07 逐查询地板 tau: 若锚点带逐题实测延迟(point.json 的 sql_ms_by_qid)
    #   与逐题离线代价(offline_byqid_*.json), 就从最便宜那撮查询解出固定开销,
    #   两个锚点正好定 tau(N)=k*N^e 的两个参数。是测量, 不含训练参数。
    #   缺任一文件则不启用, 模型退回原样(已验证完全等价)。
    _tau_before = getattr(m, "tau_", None)
    try:
        _attach_tau(adv, a)
    except Exception as _e:
        print(f"!! tau 地板未启用: {_e}", file=sys.stderr)
    # 2026-10-09 修 BUG(顺序错): 上面的 calibrate 是在装 tau 地板之前跑的,
    #   而 tau 地板在 _parts 里把 SQL 抬高约 2.8x(phoebe_predictor.py:194)。
    #   结果是陈旧的 cal_sql_ 乘在被地板抬高过的 SQL 上 ——
    #   实测 bal_test 在它自己的两个锚点上就偏 4.24x / 2.61x(应为 1.00x)。
    #   计分脚本(score2_wall_r.py 等)之所以没这个毛病, 是因为它们在 _load 返回后
    #   删掉标定重标了一次, 那时 tau 已经在了 —— 所以论文的数是对的, 只有工具是错的。
    #   修法: tau 一旦装上就重标一次, 让标定看得见地板。
    if a.anchors and getattr(m, "tau_", None) is not None and _tau_before is None:
        for _c in ("cal_", "cal_sql_", "cal_ns_"):
            if hasattr(m, _c): delattr(m, _c)
        m.calibrate(cost, steps, anc4 if (_USE_COMP_CAL and all(s > 0 and n > 0 for _, _, s, n in anc4)) else anc)
    return adv


def _attach_tau(adv, a, cheap_pct=20):
    import math, glob as _g
    if not getattr(a, "anchors", None): return
    A = json.load(open(a.anchors))
    base = os.path.basename(a.offline).replace("offline_", "").replace(".json", "")
    # 2026-10-15: v2 文件名带 _v2 后缀(offline_test2_v2.json), 逐题离线代价与老的是同一份
    #   (cost 数组逐位相), 去掉后缀再找; 否则地板被静默跳过 —— 逐题 Q-error 1.38 -> 1.64 的原因。
    base = base[:-3] if base.endswith("_v2") else base
    qcf = os.path.join(os.path.dirname(os.path.abspath(a.offline)), f"offline_byqid_{base}.json")
    if not os.path.exists(qcf): return
    qc = json.load(open(qcf))["cost_by_qid"]
    pts = []
    for x in A:
        srv, th, ag = x["srv"], x.get("th", 2), x["agents"]
        hit = None
        # 2026-10-15: 锚点文件若写明来源点(v2 锚点取自 testv2), 直接读那一个, 不按目录名猜。
        if x.get("source"):
            _sp = os.path.join(os.path.dirname(qcf), x["source"], "point.json")
            if os.path.exists(_sp):
                hit = json.load(open(_sp)).get("sql_ms_by_qid") or None
        # 2026-10-14 修 BUG: 原来去全仓库 glob 同配置名的点, 取第一个带 sql_ms_by_qid 的。
        #   而 ndb_* 与 bal_test 的 qid 都从 900000 起编号 —— 于是 bal_test 的地板是拿
        #   ndb_authors 的实测延迟配 bal_test 的离线成本解出来的(qid "重合" 63/63, 全是撞号)。
        #   修法: 只在【该负载自己的测量目录】里找; 找不到就不装地板, 而不是借别人的。
        _own = {"bal_test": ["bal_test", "testset_grid", "prospective_grid", "ladder_test"],
                "test2": ["testset2_grid", "prospective2b", "ladder_test2"]}.get(base, [base])
        for _od in ([] if hit else _own):
            for d in _g.glob(os.path.join(os.path.dirname(qcf), _od, f"s{srv}_l4_t{th}_a{ag}_b0.2/point.json")):
                j = json.load(open(d))
                if j.get("sql_ms_by_qid"): hit = j["sql_ms_by_qid"]; break
            if hit: break
        if not hit: continue
        pr = sorted((qc[k], v) for k, v in hit.items() if k in qc)
        if len(pr) < 5: continue
        k = max(3, len(pr) * cheap_pct // 100)
        pts.append((float(ag), float(np.median([v for _, v in pr[:k]]))))
    if len(pts) < 2 or pts[0][0] == pts[1][0]: return
    (n1, f1), (n2, f2) = pts[:2]
    e = (math.log(f1) - math.log(f2)) / (math.log(n1) - math.log(n2))
    adv.m.tau_ = (f1 / n1 ** e, e)
    print((f"!! per-query floor enabled: tau(N) = {adv.m.tau_[0]:.4g}*N^{e:.3f} ms "
           f"(measured at the anchors: N={n1:.0f}->{f1:.3f}ms, N={n2:.0f}->{f2:.3f}ms)") if _en() else
          (f"!! tau 地板已启用: tau(N) = {adv.m.tau_[0]:.4g}*N^{e:.3f} ms "
           f"(锚点实测 N={n1:.0f}->{f1:.3f}ms, N={n2:.0f}->{f2:.3f}ms)"), file=sys.stderr)


def cmd_advise(a):
    adv = _load(a)
    if a.isoline:
        lv = [int(x) for x in a.isoline.split(",")]
        print(f"目标 db_time <= {a.target_seconds}s 的等值线(th={a.th}):")
        for ag, n in adv.isoline(a.target_seconds, lv, th=a.th, max_cores=a.max_cores):
            cap = min(a.max_cores, adv.useful_cores(ag, a.th))
            print(f"  并发 {ag:4d}  ->  " + (f"{n} 核" if n else
                  (f"达不到(该并发下最多用得上 {cap} 核, 再加核也压不下 db_time)"
                   if cap < a.max_cores else f"{a.max_cores} 核也达不到")))
    elif a.max_agents_for_cores:
        for srv in [int(x) for x in a.max_agents_for_cores.split(",")]:
            r, note = adv.max_agents(a.target_seconds, srv, th=a.th)
            print(f"  {srv:3d} 核  ->  " + (f"最多 {r} 并发" if r else "连 1 并发都达不到"))
            if note and r:
                print(f"           {note}")
    else:
        r = adv.recommend(a.target_seconds, a.agents, th=a.th, safety=a.safety,
                          max_cores=a.max_cores)
        if not r["ok"]:
            print(f"达不到: {r['reason']}"); return
        print(f"推荐: {r['srv']} 核 (th={r['th']}), 并发 {r['agents']}")
        print(f"  预测 db_time {r['pred_db_time']:.0f}s, 目标 {r['target']:.0f}s, 余量 {r['margin']:.2f}x")
        print(f"  {r['note']}")
        if getattr(adv.m, "cal_warn_", None):
            print("  锚点未被采用(见上方警告), 实际精度退化到零实测水平"
                  "(两个陌生负载实测 18.9% / 42.1%, 偏差方向相反, ),"
                  "\n     建议按 2x 余量重新选型, 或复测锚点后重来。")
        elif not hasattr(adv.m, "cal_"):
            print("  未提供锚点: 实际精度为零实测水平 —— 15 个网格外配置的前瞻实测"
                  "中位 **39.8%**, 推荐机器实测只用掉目标时间约 66%(多买约 1.5 倍)。\n"
                  "     补 1 个锚点(81 秒)-> 误差 10.0%, 贴到 0.86x。")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure-offline", help="步骤1: 量逐条 SQL 的稳态离线耗时")
    m.add_argument("--dsn", required=True); m.add_argument("--sql-file", required=True)
    m.add_argument("--out", default="offline.json"); m.add_argument("--repeats", type=int, default=6)
    m.add_argument("--timeout-ms", type=int, default=30000); m.add_argument("--steps", type=int, default=24500)
    m.set_defaults(func=cmd_measure_offline)
    v = sub.add_parser("advise", help="步骤3: 反推配置")
    v.add_argument("--offline", required=True)
    v.add_argument("--anchors", help="锚点 JSON: [{srv,th,agents,db_time_s,sql_s,ns_s}, ...] "
                                     "带 sql_s/ns_s 可让 1 锚点从 35.7%% 降到 19.5%%")
    v.add_argument("--zero-shot-quantile", type=float, default=0.0, dest="zero_shot_quantile",
                   help="零实测时用训练负载并发指数的 q 分位(悲观)替代拟合值。"
                        "**默认 0 = 关闭()**。曾默认 0.9, 被 15 个网格外配置的**前瞻实测**推翻: "
                        "预测误差 150.6%%(不开是 39.8%%), 推荐的机器实测只用掉目标时间的 36%%"
                        "(等于买 3 倍), 还拒答了一个实测可达的目标。"
                        "它只在**恰好被低估**的负载上有用(bal_test); 在被高估的负载上(test2)"
                        "把真实达标率从 70.8%% 压到 33.3%%。仅在**无锚点**时生效。")
    v.add_argument("--anchor-quantile", type=float, default=None, dest="anchor_quantile",
                   help="**只在恰好 1 个锚点时生效**(): 把 ag 指数换成训练负载指数分布的 q 分位, "
                        "再用锚点标定常数。实测 q=0.9: bal_test 中位误差 16.1%%->**7.5%%**、"
                        "p90 29.2%%->26.7%%、少用 3.84 个核, 代价是达标率 95.1%%->90.2%%。"
                        "默认关闭 —— 这是**降误差、减保守度**的取舍, 要不要换由你定。")
    v.add_argument("--target-seconds", type=float, required=True)
    v.add_argument("--agents", type=int, default=64); v.add_argument("--th", type=int, default=2)
    # : 默认必须是 None, 否则 recommend 里的 `auto = safety is None` 恒为 False,
    #   / 建好并验证过的 rho 自适应余量就成了死代码 —— 生产一直在用固定 1.20x。
    #   实测后果: 1 锚点本可贴到 0.86x, 被固定余量压到 0.67x(多买约 19%)。
    #   传具体数值(如 --safety 1.2)可回到固定余量。
    v.add_argument("--safety", type=float, default=None,
                   help="选核时的余量倍数。**默认 1.05**(2026-09-29 由用户定)。"
                        "旧默认 None(rho 自适应)已废弃: 其系数 expected_err(rho)="
                        "0.0805+0.0289|log(rho/2.40)| 是在**含 bal_test** 的 52 点上拟的(泄露); "
                        "只用 11 个训练库重拟得 R^2=0.142(线性 0.139, 常数 0), rho 的 U 型不复现。\n"
                        "余量越大, 预测就越是刻意低于客户目标: 训练库上 预测/目标 = "
                        "1.00->0.86x, 1.10->0.77x, 1.20->0.73x, 1.50->0.60x —— "
                        "客户说 8000, 余量 1.272 的工具只去够 6289。\n"
                        "而余量**买不到预测精度**: |预测-实测| 从 1.00 的 12.5% 到 1.50 的 11.3%, 基本持平。\n"
                        "1.05 的实测(112 个预注册目标, 2 锚点): 实测/目标 0.87x/0.86x "
                        "(整数核物理下限 0.92x/0.91x), |预测-实测| 中位 5.7%/4.8%, "
                        "比最优解多 0.37/0.40 核; 达标 35/38 与 35/35。")
    v.add_argument("--max-cores", type=int, default=64)
    v.add_argument("--isoline", help="逗号分隔的并发档, 输出等值线")
    v.add_argument("--max-agents-for-cores", help="逗号分隔的核数, 反问能撑多少并发")
    v.set_defaults(func=cmd_advise)
    a = p.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
