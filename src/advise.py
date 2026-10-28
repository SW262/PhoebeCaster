#!/usr/bin/env python3
"""按客户目标推荐核数,开/关超线程各给一个。

目标量是真实完成时间(墙钟,秒):客户说"这批活要多久跑完"。

内部走 db_time 再换算(2026-10-07, 224 组预注册检验, 目标与真值两头都是实测墙钟):
    预测 db_time 再换算成墙钟  -> 推荐达标率 87.9%, 墙钟预测误差中位 5.75%   <- 采用
    直接以实测墙钟为 target    -> 推荐达标率 82.3%, 墙钟预测误差中位 5.34%
  db_time 在核数维度上区分度更锐利, 而选档靠的就是核数维度; 墙钟混进了与核数关系弱的
  客户端排队与调度尾部, 相邻核档更难分开, 更容易错一档。对客户等价: 给的是真实时间,
  拿到的推荐也按真实时间验证, 中间走不走 db_time 看不。

  python3 advise.py 120 64 --workload test2
  python3 advise.py 22000 224 --workload bal_test
  python3 advise.py 8000 96 --offline my.json --anchors my_anchors.json --max-cores 24
"""
import argparse, json, os, sys, io, importlib.util
from types import SimpleNamespace
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
from paths import DATA as D

TOL = 0.15          # 与 phoebe_advisor.TOL 一致
GRID_MIN_CORES = 2  # 训练网格采过的最小物理核数; 低于即外推
GRID_MAX_CORES = 16 # 训练网格采过的最大物理核数; 超过即外推, 本机也无法实测
# r = db_time / (并发 x 墙钟)。11 个训练库拟合, 两个测试库不参与。
_RM = json.load(open(os.path.join(D, "r_model.json")))
def to_wall(db_time_s, agents, eff_cores):
    import math
    r = math.exp(_RM["a"]*math.log(agents) + _RM["b"]*math.log(eff_cores) + _RM["c"])
    return db_time_s / (agents * r)
MEAS_DIRS = {"bal_test":["bal_test","testset_grid","prospective_grid","ladder_test"],
             "test2":["testset2_grid","prospective2b","ladder_test2"]}
# 2026-10-15 口径统一为 v2(论文同口径): 模型 v2(rc3+rc4 + 模型3), 离线代价/锚点/实测真值
#   都取新定义测试集 testv2(b0, 与训练同定义)。PHOEBE_MODEL=v1 退回老口径(老网格 + b0.2 实测)。
V2 = os.environ.get("PHOEBE_MODEL", "v2") == "v2"
MEAS_V2 = {"bal_test": "testv2/test", "test2": "testv2/test2"}
# 网页"现场实测"按钮采的点(live_measure.sh)。单独目录, 只用来显示, 不进任何论文数字。
MEAS_LIVE = {"bal_test": "testv2_live/test", "test2": "testv2_live/test2"}

def measured_live(wl, srv, th, agents):
    """现场实测的完成时间(秒); 没测过返回 None。闸同 testv2: steps = passes x questions。"""
    p = os.path.join(D, MEAS_LIVE.get(wl, ""), f"s{srv}_l4_t{th}_a{agents}_b0", "point.json")
    try:
        d = json.load(open(p))
        if d["steps"] == d["passes"] * d["questions"]: return d["completion_seconds"]
    except Exception: pass
    return None

def measured(wl, srv, th, agents):
    """该配置的实测完成时间(秒, 墙钟); 没采过返回 None。"""
    if V2:
        p = os.path.join(D, MEAS_V2.get(wl, ""), f"s{srv}_l4_t{th}_a{agents}_b0", "point.json")
        if os.path.exists(p):
            try:
                d = json.load(open(p))
                if d["steps"] == d["passes"] * d["questions"]: return d["completion_seconds"]
            except Exception: pass
        return None
    for r in MEAS_DIRS.get(wl, []):
        p = os.path.join(D, r, f"s{srv}_l4_t{th}_a{agents}_b0.2", "point.json")
        if os.path.exists(p):
            try: return json.load(open(p))["completion_seconds"]
            except Exception: pass
    return None

def busy():
    """采集是否正在占机。返回占机进程的 pidfile 名或 None。"""
    for f in ("fill_ladder.pid", "chain_ladder.pid", "run_prospective2b.pid"):
        p = os.path.join(D, f)
        if os.path.exists(p):
            try:
                if os.path.exists(f"/proc/{int(open(p).read().split()[0])}"): return f
            except Exception: pass
    return None

def run_measure(wl, todo, agents):
    import subprocess, time
    key = "test" if wl == "bal_test" else "test2"
    lines = "".join(f"{key} {n} {th} {agents}\n" for n, th in todo)
    q = os.path.join(D, "advise_todo.txt")
    open(q, "a").write(lines)
    b = busy()
    if b:
        print(f"  采集正在占机({b}), 已排队 {len(todo)} 个点 -> advise_todo.txt")
        print(f"  等它跑完后执行: ./fill_ladder.sh advise_todo.txt\n")
        return
    est = sum(6 if agents <= 8 else 3 if agents <= 16 else 2 for _ in todo)
    print(f"  开始实测 {len(todo)} 个点, 预计 {est} 分钟...\n")
    subprocess.run([os.path.join(SRC,"fill_ladder.sh"), q], cwd=D)
    for n, th in todo:
        y = measured(wl, n, th, agents)
        print(f"  {n} 核 / 超线程{'开' if th==2 else '关'} / {agents} 并发 -> "
              + (f"实测 {y:,.0f}s" if y else "采集失败, 看 fill_ladder.log"))
    print()

# 2026-10-21(用户定): 默认只在训练网格实采的 7 档核数里挑, 与论文 84.4%(189/224)的评分
#   口径(check_prod_v2.py / demo_failmap.py)逐位相同 —— 工具输出 = 论文评的东西。
#   挑法同评分: 预测离目标的 |pred/T-1| 最小的一档; <= TOL 算达标, 否则如实说无解。
#   --any-core 退回旧行为(1..max 任意整数核, 那些档多数没实测, 不在 224 的评分里)。
RUNGS = [2, 4, 6, 8, 11, 13, 16]
def rung_pick(adv, wl, target, agents, th, max_cores):
    # 候选 = 网格 7 档 + 网页上现场实测过的核数(2026-10-22 用户定: "测过的就能推荐")。
    # 论文 224 个目标的评分只读 testv2/, 没有现场点, 所以那边仍逐位 = 7 档口径。
    live = [n for n in range(GRID_MIN_CORES, max_cores + 1)
            if n not in RUNGS and measured_live(wl, n, th, agents) is not None]
    rs = sorted([r for r in RUNGS if r <= max_cores] + live) or [RUNGS[0]]
    preds = {r: adv.db_time(r, th, agents) for r in rs}
    best = min(rs, key=lambda r: abs(preds[r] / target - 1))
    y = measured(wl, best, th, agents)
    if y is None: y = measured_live(wl, best, th, agents)
    near = dict(srv=best, pred=preds[best], dev=preds[best] / target - 1, measured=y,
                meas_dev=(y / target - 1) if y else None, extrapolated=False, live=best in live)
    if abs(preds[best] / target - 1) <= TOL:
        return dict(ok=True, srv=best, pred_db_time=preds[best], live=best in live), None, near
    # 实测档里没有落进 ±TOL 的: 再看网格【之间】的整数核(网格范围内, 不外推)。
    # 有预测落进 ±TOL 的就作为"未实测候选"一并给出 —— 只是提示, 不改判定(仍是无解),
    # 所以 224 个目标的评分口径不变。可在网页上点"现场实测"确认。
    between = [n for n in range(max(GRID_MIN_CORES, rs[0]), rs[-1] + 1) if n not in rs]   # rs 已含现场点
    unv = None
    if between:
        bp = {n: adv.db_time(n, th, agents) for n in between}
        nb = min(between, key=lambda n: abs(bp[n] / target - 1))
        if abs(bp[nb] / target - 1) <= TOL:
            yl = measured_live(wl, nb, th, agents)
            unv = dict(srv=nb, pred=bp[nb], dev=bp[nb] / target - 1, live_measured=yl,
                       live_dev=(yl / target - 1) if yl else None,
                       live_met=(None if yl is None else abs(yl / target - 1) <= TOL))
    lo, up = min(preds.values()), max(preds.values())
    kind = ("too_tight" if lo > target * (1 + TOL) else
            "too_loose" if up < target * (1 - TOL) else "gap")
    # gap: 给出夹住目标的相邻两档
    opts = []
    if kind == "gap":
        for a_, b_ in zip(rs, rs[1:]):
            if (preds[a_] - target) * (preds[b_] - target) <= 0:
                opts = [dict(srv=x, pred=preds[x], dev=preds[x] / target - 1) for x in (a_, b_)]; break
    return dict(ok=False, infeasible_gap=True, options=opts, unverified=unv), kind, near

def nearest_cfg(adv, wl, target, agents, th, hi):
    """无解时给客户的"最接近的配置": 1..hi 核里预测完成时间离目标最近(对数距离)的一档,
    以及无解的类型 —— too_tight: 给满 hi 核也慢于目标; too_loose: 1 核都快于目标;
    gap: 目标卡在相邻两档之间。只读模型预测, 不改推荐逻辑。"""
    import math
    preds = {cc: adv.db_time(cc, th, agents) for cc in range(1, int(hi) + 1)}
    best = min(preds, key=lambda cc: abs(math.log(preds[cc] / target)))
    lo = min(preds.values()); up = max(preds.values())
    kind = ("too_tight" if lo > target * (1 + TOL) else
            "too_loose" if up < target * (1 - TOL) else "gap")
    y = measured(wl, best, th, agents)
    return kind, dict(srv=best, pred=preds[best], dev=preds[best] / target - 1, measured=y,
                      meas_dev=(y / target - 1) if y else None,
                      extrapolated=bool(best < GRID_MIN_CORES or best > GRID_MAX_CORES))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", type=float, help="客户要求的**完成时间**上限(秒)")
    p.add_argument("agents", type=int, help="并发 agent 数")
    p.add_argument("--workload", choices=["bal_test","test2"], help="用内置负载(自动带 offline+anchors, 并查实测真值)")
    p.add_argument("--lang", choices=["en","zh"], default=os.environ.get("ADVISE_LANG","en"),
                   help="输出语言, 默认 en(VLDB demo 用); zh 为原中文输出")
    p.add_argument("--anchors", help="锚点 JSON")
    p.add_argument("--offline", default=os.path.join(D,"offline.json"), help="离线 SQL 成本")
    # 2026-10-07: 默认从 64 改为 16。两个硬约束, 都是实测边界:
    #   ① 训练网格的核数档最高 16(2,4,6,8,11,13,16), 更高是外推且从未验证;
    #   ② 采集的绑核布局里 server 从 0 号核起、load 固定占 16-19 号, 所以
    #      server 要到 17 核就会与 load 重叠、互相抢核 —— 本机采都采不出
    #      >16 核的干净数据点。默认 64 会不声不响给出无法验证的推荐。
    p.add_argument("--max-cores", type=int, default=16,
                   help="最多能给多少物理核(默认 16 = 训练网格与本机绑核布局的上限)")
    # --json: 给 Web UI 用。文字输出一个字不变, 只是改成不打到 stdout,
    # 结构化结果单独收一份。这样 CLI 与网页共用同一条推荐路径, 不会出现
    # "网页上的数和命令行的数对不上" 这种事。
    p.add_argument("--json", action="store_true", help="输出 JSON(给 Web UI), 文字表格不变只是不打印")
    p.add_argument("--any-core", action="store_true",
                   help="旧行为: 在 1..max 任意整数核里挑(多数没实测, 不在论文 224 个目标的评分口径里)")
    p.add_argument("--measure", action="store_true",
                   help="推荐的配置若没跑过, 就真去跑一遍(约 1~6 分钟/点)。采集在占机时自动排队。")
    a = p.parse_args()
    JOUT = {"target": None, "agents": None, "workload": None, "anchors": [],
            "warnings": [], "rows": [], "tol": TOL, "max_cores": None}
    _real_stdout = sys.stdout
    if a.json: sys.stdout = io.StringIO()
    if a.workload:
        _sfx = "_v2" if V2 else ""
        a.offline = os.path.join(D, f"offline_{a.workload}{_sfx}.json")
        if not a.anchors: a.anchors = os.path.join(D, f"anchors_{a.workload}{_sfx}.json")
    if not os.path.exists(a.offline):
        sys.exit(f"找不到 {a.offline}。先跑 phoebe_cli.py measure-offline,或用 --workload")
    sp = importlib.util.spec_from_file_location("phoebe_cli", os.path.join(SRC,"phoebe_cli.py"))
    cli = importlib.util.module_from_spec(sp); sp.loader.exec_module(cli)
    # 2026-10-07: 以前这里把 stderr 整个换成 StringIO 丢弃, 结果三项守卫的告警
    #   全被吞掉 —— 而守卫告警正是往 stderr 打的(phoebe_predictor 的锚点自洽、
    #   phoebe_cli 的负载强度 rho 与失败率)。只过滤掉加载期的噪声, 以 "!!" 开头的
    #   告警行必须原样交给用户。
    e = sys.stderr; buf = io.StringIO(); sys.stderr = buf
    os.environ["ADVISE_LANG"] = a.lang          # 让 phoebe_cli / predictor 的告警跟随 --lang
    try:
        adv = cli._load(SimpleNamespace(offline=a.offline, anchors=a.anchors,
                                        zero_shot_quantile=0.0, anchor_quantile=None))
    finally:
        sys.stderr = e
    _warns = [ln for ln in buf.getvalue().splitlines() if ln.lstrip().startswith("!!")]
    # 对外一律是完成时间: 内部预测 db_time, 输出前除以 (并发 x r)。
    _ps = importlib.util.spec_from_file_location("pp", os.path.join(SRC, "phoebe_predictor.py"))
    _pp = importlib.util.module_from_spec(_ps); _ps.loader.exec_module(_pp)
    _dbt = adv.db_time
    adv.db_time = (lambda srv, th, agents, _f=_dbt, _m=_pp:
                   to_wall(_f(srv, th, agents), agents, _m.effective_cores(srv, th, agents)))

    JOUT.update(target=a.target, agents=a.agents, max_cores=a.max_cores,
                workload=os.path.basename(a.offline), warnings=[w.strip() for w in _warns])
    if a.anchors and os.path.exists(a.anchors):
        JOUT["anchors"] = json.load(open(a.anchors))
    EN = (a.lang == "en")
    if EN:
        print(f"\nTarget: finish in {a.target:,.0f} s @ {a.agents} agents "
              f"(met = measured within {100*TOL:.0f}% of target, either way)   workload: "
              f"{os.path.basename(a.offline)}")
    else:
        print(f"\n客户目标: 完成时间 {a.target:,.0f} 秒 @ {a.agents} 并发 "
              f"(达标 = 实测与目标相差 <= {100*TOL:.0f}%, 双向)   负载: "
              f"{os.path.basename(a.offline)}")
    if a.anchors:
        A = json.load(open(a.anchors))
        if EN:
            print("Anchors: " + "  +  ".join(
                f"{x['srv']} cores/HT {'on' if x.get('th',2)==2 else 'off'}/{x['agents']} agents" for x in A))
        else:
            print("锚点: " + "  +  ".join(
                f"{x['srv']}核/超线程{'开' if x.get('th',2)==2 else '关'}/{x['agents']}并发" for x in A))
    else:
        print("Anchors: none (zero measurement; error 20-45% and its sign is not predictable; "
              "one anchor, about a minute, brings it to 5-8%)" if EN else
              "锚点: 无(零实测, 误差 20~45% 且方向不可预测; 补 1 个约 1 分钟可降到 5~8%)")
    if _warns:                      # 守卫告警: 必须让用户看, 不能吞掉
        for ln in _warns:
            print("  " + ln.strip())
    print()
    if EN:
        print(f"  {'HT':<6s} {'cores':>7s} {'pred':>11s} {'vs target':>10s} {'measured':>11s} {'vs target':>10s}")
    else:
        print(f"  {'超线程':<6s} {'推荐核数':>7s} {'预测':>11s} {'预测离目标':>10s} {'实测':>11s} {'实测离目标':>10s}")
    print("  " + "-"*60)
    ok = False; todo = []
    for th, lab in ((2, "on" if EN else "开"), (1, "off" if EN else "关")):
        if not a.any_core:
            r, _kind, _near = rung_pick(adv, a.workload, float(a.target), a.agents, th, a.max_cores)
            _cap = a.max_cores
        else:
            r = adv.recommend(float(a.target), a.agents, th=th, max_cores=a.max_cores)
            if not r.get("ok"):
                # 与 recommend 同一个搜索上界: 低并发时超过 useful_cores 加核压不下时间(轻载地板护栏)
                _cap = int(adv.useful_cores(a.agents, th)); _hi = min(a.max_cores, _cap)
                _kind, _near = nearest_cfg(adv, a.workload, float(a.target), a.agents, th, _hi)
        if not r.get("ok"):
            if r.get("infeasible_gap"):
                _why = {"too_tight": ("target is faster than any allowed core count can reach",
                                      "目标比允许的最多核数还快"),
                        "too_loose": ("even the fewest allowed cores finish well before the target",
                                      "最少的核数都远早于目标完成"),
                        "gap": ("target falls between two adjacent core counts",
                                "目标卡在相邻两档之间")}[_kind]
                if EN:
                    print(f"  {lab:<6s} {'none':>7s}   {_why[0]}; no core count lands within "
                          f"\u00b1{100*TOL:.0f}%. Nearest: {_near['srv']} cores. Bracketing:")
                else:
                    print(f"  {lab:<6s} {'无解':>7s}   {_why[1]}, 没有核数能落进 ±{100*TOL:.0f}%。最接近: {_near['srv']} 核。两侧:")
                for o in r["options"]:
                    y2 = measured(a.workload, o["srv"], th, a.agents)
                    ys2 = ((f"  measured {y2:,.0f}s ({100*(y2/a.target-1):+.1f}%)" if EN else
                            f"  实测 {y2:,.0f}s ({100*(y2/a.target-1):+.1f}%)") if y2 else "")
                    print(f"  {'':<6s} {str(o['srv'])+('' if EN else ' 核'):>7s} {o['pred']:10,.0f}s {100*o['dev']:+9.1f}%{ys2}")
                _u = r.get("unverified")
                if _u:
                    _ls = ((f", measured live {_u['live_measured']:,.0f}s ({100*_u['live_dev']:+.1f}%)" if EN else
                            f", 现场实测 {_u['live_measured']:,.0f}s ({100*_u['live_dev']:+.1f}%)") if _u["live_measured"] else
                           (", not measured" if EN else ", 未实测"))
                    print((f"  {'':<6s} unverified: {_u['srv']} cores (between measured counts) predicted "
                           f"{_u['pred']:,.0f}s ({100*_u['dev']:+.1f}%){_ls}") if EN else
                          (f"  {'':<6s} 未实测候选: {_u['srv']} 核(在实测档之间) 预测 "
                           f"{_u['pred']:,.0f}s ({100*_u['dev']:+.1f}%){_ls}"))
                JOUT["rows"].append(dict(ht=("on" if th==2 else "off"), ok=False,
                                         reason="infeasible_gap", unverified=_u,
                                         options=[dict(srv=o["srv"], pred=o["pred"], dev=o["dev"],
                                                       measured=measured(a.workload, o["srv"], th, a.agents))
                                                  for o in r["options"]],
                                         kind=_kind, nearest=_near, cap=_cap if _cap < a.max_cores else None))
            else:
                print(f"  {lab:<6s} {('n/a' if EN else '达不到'):>7s}   {r['reason'].splitlines()[0][:60]}")
                JOUT["rows"].append(dict(ht=("on" if th==2 else "off"), ok=False,
                                         reason=r["reason"].splitlines()[0][:200], options=[],
                                         kind=_kind, nearest=_near, cap=_cap if _cap < a.max_cores else None))
                _ys = (f"  measured {_near['measured']:,.0f}s" if EN else f"  实测 {_near['measured']:,.0f}s") if _near["measured"] else ""
                print(f"  {'':<6s} {('nearest ' if EN else '最接近 ')+str(_near['srv']):>7s} {_near['pred']:10,.0f}s {100*_near['dev']:+9.1f}%{_ys}")
            continue
        ok = True
        n, pr = r["srv"], r["pred_db_time"]
        extrap = ("  !extrapolated" if EN else "  ⚠外推") if n > GRID_MAX_CORES else ""
        y = measured(a.workload, n, th, a.agents)
        if y is None and r.get("live"): y = measured_live(a.workload, n, th, a.agents)
        ys = f"{y:10,.0f}s" if y else f"{('not run' if EN else '未跑过'):>11s}"
        yd = f"{100*(y/a.target-1):+9.1f}%" if y else f"{'-':>10s}"
        mark = "" if not y else ("  ✓" if abs(y/a.target-1) <= TOL else "  ✗")
        if y is None: todo.append((n, th))
        print(f"  {lab:<6s} {str(n)+('' if EN else ' 核'):>7s} {pr:10,.0f}s {100*(pr/a.target-1):+9.1f}% "
              f"{ys} {yd}{mark}{extrap}")
        JOUT["rows"].append(dict(ht=("on" if th==2 else "off"), ok=True, srv=n, pred=pr,
                                 pred_dev=pr/a.target-1, measured=y,
                                 meas_dev=(y/a.target-1) if y else None,
                                 met=(None if y is None else abs(y/a.target-1) <= TOL),
                                 extrapolated=bool(n > GRID_MAX_CORES), live=bool(r.get("live"))))
    if not ok:
        print("  -> relax the target, raise concurrency, or allow more cores (--max-cores)" if EN else
              "  -> 放宽目标、提高并发, 或给更多核(--max-cores)")
    print()
    if a.json:
        # 核数扫描: 每个训练网格采过的核档各预测一次。推荐只给一个数, 扫描让人
        # 看见目标线落在哪两档之间 —— demo 里"为什么是 3 核不是 4 核"一眼就清楚。
        # 档位就是网格实采的 2/4/6/8/11/13/16, 不插值、不外推。
        RUNGS = [2, 4, 6, 8, 11, 13, 16]
        sweep = []
        for th2, lab2 in ((2, "on"), (1, "off")):
            for srv2 in RUNGS:
                if srv2 > a.max_cores: continue
                try: pv = adv.db_time(srv2, th2, a.agents)
                except Exception: continue
                sweep.append(dict(ht=lab2, srv=srv2, pred=pv,
                                  measured=measured(a.workload, srv2, th2, a.agents)))
        JOUT["sweep"] = sweep
        JOUT["rungs"] = [r for r in RUNGS if r <= a.max_cores]
        txt = sys.stdout.getvalue(); sys.stdout = _real_stdout
        JOUT["feasible"] = ok; JOUT["text"] = txt
        print(json.dumps(JOUT, ensure_ascii=False))
        return
    if a.measure and todo:
        if not a.workload: sys.exit("  --measure 需要 --workload(要知道用哪个库和题目)")
        run_measure(a.workload, todo, a.agents)

if __name__ == "__main__": main()
