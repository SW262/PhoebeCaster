#!/usr/bin/env python3
"""PhoebeDB 资源配置 -> 性能 预测器(定稿)。

用法(客户侧两步):
  1. 串行量一遍离线成本  -> 每条 SQL 的稳态中位耗时(ms), 约 9 分钟
  2. 跑 2 个锚点         -> 必须一个在最低并发、一个在最高并发, 约 10 分钟
  之后任意 (核数, 并发) 的 db_time 都能预测。

锚点规则是有依据的(枚举 test 上全部合规组合, 取中位, 不挑最好):
  规则                      墙钟    误差中位   ±10%    ±20%   最差组合
  零实测                      0s    40.9%    0.0%    2.8%     40.9%
  任意 1 点                 172s    11.5%   37.1%   67.9%     30.1%
  任意 2 点(跨并发档)         601s    11.0%   41.0%   69.8%    121.1%
  2 点·都在高并发            564s    15.6%   31.6%   57.8%    121.1%
  2 点·跨最低/最高并发       586s     8.2%   51.9%   83.4%     24.1%
  3 点·低/中/高各一          828s     8.3%   53.4%   86.3%     22.0%

  "最差组合"这一列是关键: 任意取 2 点最差能到 121%, 跨最低/最高的规则把最差封在 24.1%。
  注意"选高并发省墙钟"是错的(15.6%) —— 省下的时间换来一倍误差。

为什么必须有锚点:
  误差几乎全在 sql 部分, 而其中九成来自并发指数 ag_exp ——
  把它换成真值, 误差从 43.3% 掉到 9.5%。该指数跨负载在 0.427~0.962 之间,
  用成本分布统计/schema 规模/BAL/访问集重叠四类特征做留一预测, 最好的 MAE 0.106,
  而指数标准差是 0.148 —— 几乎零解释力。它是负载的内禀属性, 静态量测不出来。
  两个锚点必须跨并发档: 同档两点只能定常数, 定不出指数。

曾报过"2 锚点 6.9%" —— 那是从高并发组合里挑出的最好一对, 用测试集选锚点, 已作废。
   凡是"挑最好"的数字都不能当成绩( 第 2 条教训, 这是第二次栽在同一条上)。
"""
import json, glob, os, re, collections
import numpy as np

QP = [50, 95, 99]
HT_GAIN = 1.24
STEPS_REF = 24500


# (2026-10-01, 预注册 prereg_htgain.json): 超线程增益随并发变, 不是常数。
#   11 个训练库补齐 th=1 网格后(308 点)实测, 308 对同配置 th1/th2:
#       ag=8 -> 1.072   ag=16 -> 1.181   ag=64 -> 1.262   ag=224 -> 1.293
#   低并发填不满超线程。嵌套留一(只用训练库)选形式:
#       H0 常数 1.24(现行)  中位 14.1%  p90 34.2%
#       H1 常数 1.234       中位 14.2%  p90 34.4%
#       H2 按并发分档       中位 13.5%  p90 32.7%   <- 采用
#       H3 线性 log(ag)     中位 13.4%  p90 33.2%
#   H2 与 H3 都过判据, 选 H2 是因为 p90 更低 —— 反推达标率由误差尾部决定
#   (误差 >10% 时达标率从 85~95% 崩到 38%), 中位那 0.1 个点无关紧要。
#   档间按 log(ag) 线性插值, 两端取端值, 因此有界。
_HT_TAB = None
def _ht_gain(agents):
    """关闭超线程相对开启的减速倍数。agents 为 None 时退回常数(向后兼容)。"""
    global _HT_TAB
    if agents is None: return HT_GAIN
    if _HT_TAB is None:
        try:
            import os as _o
            t = json.load(open(_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))),
                                            "data", "ht_gain_fit.json")))["table"]
            _HT_TAB = sorted((float(k), float(v)) for k, v in t.items())
        except Exception:
            _HT_TAB = []
    if not _HT_TAB: return HT_GAIN
    ks = [k for k, _ in _HT_TAB]
    if agents <= ks[0]:  return _HT_TAB[0][1]
    if agents >= ks[-1]: return _HT_TAB[-1][1]
    for (a0, g0), (a1, g1) in zip(_HT_TAB, _HT_TAB[1:]):
        if a0 <= agents <= a1:
            t = (np.log(agents) - np.log(a0)) / (np.log(a1) - np.log(a0))
            return float((1 - t) * g0 + t * g1)
    return HT_GAIN


def effective_cores(srv, threads_per_core, agents=None):
    return srv * (_ht_gain(agents) if threads_per_core == 2 else 1.0)


class PhoebePredictor:
    """db_time = sum_i cost_i * amp(cost_i, B, ag, C) * (steps/n) + nonsql(ag, C, B)"""

    FA = ["cost", "B", "ag", "C"]
    # 2026-10-07: B 已从非SQL项移除(预注册 prereg_dropB.json, md5 fd97616b)。
    #   理由: B = mean(cost)×steps, 而全部已测点的 steps 相差 0.4%, 故 B 与 mean(cost)
    #   完全共线, 拟出的指数 +0.0574 无法归因。步数线性性实验(prereg_steplin)实测总量
    #   指数 1.019(两配置, R²>0.9998), 分量级非SQL指数 0.90/1.13 覆盖 1.0 —— 不支持 0.0574。
    FN = ["ag", "C"]

    def fit(self, workloads, anchor_rows=None):
        """workloads: [{cost: ndarray(ms), steps: int, points: [{C, ag, sql, ns, q50, q95, q99}]}]

        anchor_rows: (sql 行, ns 行) —— 目标负载的锚点转成的训练行, 见 `anchor_as_rows`。
            依据: 这是 Hilprecht VLDB22 官方实现的 few-shot 做法 ——
              `read_workload_runs` 把目标域的少量样本直接并进训练集、权重 1、照常训练。
              我们原有的 `calibrate` 是事后两参数修正, 是另一回事。
              两者叠加才有效: 1 锚点 35.7%(仅 cal) / 28.7%(仅 refit) -> 19.5%(叠加)。
              权重经嵌套验证选出 W*=1, 即论文的权重 —— 加权是多余的。
        """
        rows, nrows = [], []
        for wi, w in enumerate(workloads):
            B = float(np.mean(w["cost"])) * w["steps"] / 1000.0
            for pc in QP:
                oc = float(np.percentile(w["cost"], pc))
                for p in w["points"]:
                    v = p.get(f"q{pc}")
                    if v: rows.append((oc, B, p["ag"], p["C"], v / oc, wi))
            for p in w["points"]:
                if p.get("ns", 0) > 0:
                    nrows.append((p["ag"], p["C"], p["ns"] * STEPS_REF / w["steps"], wi))
        ar = anchor_rows or ([], [])
        self.ba_ = self._wls(rows, len(self.FA), extra=list(ar[0]))
        self.bn_ = self._wls(nrows, len(self.FN), extra=list(ar[1]))
        return self

    @staticmethod
    def anchor_as_rows(cost, steps, anchors, gid=10_000):
        """把锚点变成可并进训练集的行。anchors: [(C, ag, sql_s, ns_s)]。

        锚点采集本来就产出 `db_time_breakdown_s`, 所以 sql / 非 SQL 能分开。
        sql 行用 mean(cost) 作成本特征, 有效放大倍数 = sql实测 / (mean(cost)/1000 · steps)。
        """
        cost = np.asarray(cost, float)
        B = float(np.mean(cost)) * steps / 1000.0
        mc = float(np.mean(cost))
        R, N = [], []
        for C, ag, sql_s, ns_s in anchors:
            if sql_s and sql_s > 0:
                R.append((mc, B, float(ag), float(C), sql_s / (mc / 1000.0 * steps), gid))
            if ns_s and ns_s > 0:
                N.append((float(ag), float(C), ns_s * STEPS_REF / steps, gid))
        return R, N

    @staticmethod
    def anchor_as_rows_pct(cost, steps, anchors, gid=10_000):
        """只用于单锚点路径。锚点按 QP 分位生成多行, 与 `base_rows` 同形。

        为什么只给单锚点: 单锚点的事后修正只能定常数(k0≡0), 分位行是唯一能约束
        成本指数的信息 -> 纯增益(出族 11.7%→8.6%, p90 29.2%→19.8%, 最差 45.6%→29.5%)。
        两锚点已用 k0 修过指数, 分位行再拉成本指数 = 重复修正 -> 反而变差(7.1%→9.6%),
        所以两锚点保持原 `anchor_as_rows` 不变。

        anchors: [(C, ag, sql_s, ns_s, {pc: 实测 sql_exec_p{pc}_ms})]
        """
        cost = np.asarray(cost, float)
        B = float(np.mean(cost)) * steps / 1000.0
        R, N = [], []
        for C, ag, sql_s, ns_s, q in anchors:
            for pc in QP:
                v = (q or {}).get(pc)
                oc = float(np.percentile(cost, pc))
                if v and v > 0 and oc > 0:
                    R.append((oc, B, float(ag), float(C), v / oc, gid))
            if ns_s and ns_s > 0:
                N.append((float(ag), float(C), ns_s * STEPS_REF / steps, gid))
        return R, N

    @staticmethod
    def _wls(rows, nf, extra=None):
        """每个负载等权 —— 否则点数多的负载(如 dev 1512 点)会占 97% 的权重。

        extra: 目标负载的锚点行。每行权重固定为 1.0, 不参与"每个负载等权"的分组折算。
            这一条是实测逼出来的: 起初把锚点行直接并进 rows, 于是它们被当成一个
            "负载"做等权折算, 权重变成 sqrt(1/锚点行数) —— 1 个锚点时恰好也是 1.0,
            2 个锚点时降到 0.707, 结果从 5.1% 退到 6.4%( 的验证值对不上)。
            1 锚点重合掩盖了这个差异, 只有做生产路径自查才发现。
        """
        a = np.array([r[:nf] for r in rows], float)
        y = np.log(np.array([r[nf] for r in rows], float))
        g = np.array([r[nf + 1] for r in rows])
        cnt = collections.Counter(g.tolist())
        wt = np.sqrt(1.0 / np.array([cnt[x] for x in g], float))
        if extra:
            ea = np.array([r[:nf] for r in extra], float)
            ey = np.log(np.array([r[nf] for r in extra], float))
            a = np.vstack([a, ea]); y = np.concatenate([y, ey])
            wt = np.concatenate([wt, np.ones(len(extra))])      # 锚点行权重恒 1.0
        X = np.column_stack([np.log(np.maximum(a[:, i], 1e-12)) for i in range(nf)] + [np.ones(len(a))])
        return np.linalg.lstsq(X * wt[:, None], y * wt, rcond=None)[0]

    def _parts(self, cost, steps, ag, C):
        """返回 (SQL 分量, 非SQL 分量), 与 _raw 的两项一致。"""
        B = float(np.mean(cost)) * steps / 1000.0
        vals = {"cost": np.asarray(cost, float), "B": B, "ag": float(ag), "C": float(C)}
        amp = np.exp(self.ba_[-1]) * np.ones_like(vals["cost"])
        for i, k in enumerate(self.FA):
            amp = amp * np.maximum(np.asarray(vals[k], float), 1e-9) ** self.ba_[i]
        # 2026-10-07 逐查询地板(预注册 prereg_tau2.json): 单条查询的预测延迟
        #   不得低于该并发下的固定开销 tau —— 解析、计划、协议往返这些不随查询变贵,
        #   但同样排队。实测: 离线 0-0.3ms 的查询在 224 并发下被抬到 3-4ms(放大 30-40 倍),
        #   而幂律外推到 c->0 必然预测趋于 0, 该档单查询 Q-error 2.95。
        #   tau 由两个锚点运行的"最便宜那撮查询的实测延迟中位"解出 —— 是测量,
        #   不是从训练集学来的参数。对贵查询完全不影响(max 的性质)。
        _c = np.asarray(cost, float) / 1000.0
        _per = _c * amp
        if getattr(self, "tau_", None) is not None:
            _per = np.maximum(_per, self.tau_[0] * float(ag) ** self.tau_[1] / 1000.0)
        sql = float(np.sum(_per)) * steps / len(cost)
        ns = np.exp(self.bn_[-1])
        for i, k in enumerate(self.FN):
            ns = ns * float({"ag": ag, "C": C, "B": B}[k]) ** self.bn_[i]
        return sql, ns * steps / STEPS_REF

    def _raw(self, cost, steps, ag, C):
        B = float(np.mean(cost)) * steps / 1000.0
        vals = {"cost": np.asarray(cost, float), "B": B, "ag": float(ag), "C": float(C)}
        amp = np.exp(self.ba_[-1]) * np.ones_like(vals["cost"])
        for i, k in enumerate(self.FA):
            amp = amp * np.maximum(np.asarray(vals[k], float), 1e-9) ** self.ba_[i]
        sql = float(np.sum(np.asarray(cost, float) / 1000.0 * amp)) * steps / len(cost)
        ns = np.exp(self.bn_[-1])
        for i, k in enumerate(self.FN):
            ns = ns * float({"ag": ag, "C": C, "B": B}[k]) ** self.bn_[i]
        return sql + ns * steps / STEPS_REF

    # 指数修正的合理上限。依据: 11 个负载留一, |指数修正| 中位 0.111、p90 0.234,
    # 唯一超过 0.3 的是 european_football_2(0.491), 它的 2 锚点误差 128.7%,
    # 而其余 10 个负载中位仅 18.0%。|指数修正| 与最终误差相关 r=+0.882。
    MAX_EXP_CORRECTION = 0.30

    def calibrate(self, cost, steps, anchors, strict=True):
        """anchors: [(C, ag, measured_db_time)] —— 必须至少两个不同并发档才能定指数。

        strict=True 时, 若解出的指数修正过大(锚点彼此不自洽), 完全不校正。
        依据: 坏锚点的伤害远大于不用锚点 —— european_football_2 零实测 31.3%,
        用它的锚点做指数校正 128.7%, 退回常数校正 176.9%(更糟, 因为锚点值本身就坏),
        不校正才是对的。
        """
        import sys
        # (2026-09-29, 预注册 prereg_compcal2.json): 锚点若带分量实测
        #   (C, ag, sql_s, ns_s), 则两个分量各自标定, 不再只对总量标定。
        #   原因: 总量标定被占比大的分量主导, 占比小的被牺牲 —— 实测 bal_test
        #   非SQL侧 标定前 23.3% -> 标定后 58.3%。
        #   11 个训练库留一: 两侧误差最大值 20.6% -> 19.3%(主判据), 合计中位 +0.12 点、
        #   p90 -0.87 点(护栏)。不增加实测成本(分量与总量出自同一次运行的同一批计时),
        #   不增加任何从训练数据学来的参数。
        if anchors and len(anchors[0]) == 4:
            ags = np.array([a for _, a, _, _ in anchors], float)
            if len(anchors) >= 2 and len(set(ags.tolist())) >= 2:
                Z = np.column_stack([np.log(ags), np.ones(len(ags))])
                for tag, idx in (("sql", 2), ("ns", 3)):
                    p = np.array([self._parts(cost, steps, a, c)[0 if tag == "sql" else 1]
                                  for c, a, _, _ in anchors])
                    yv = np.array([x[idx] for x in anchors], float)
                    ok = (p > 0) & (yv > 0)
                    if ok.sum() >= 2:
                        c_ = np.linalg.lstsq(Z[ok], np.log(yv[ok] / p[ok]), rcond=None)[0]
                        if abs(c_[0]) <= self.MAX_EXP_CORRECTION:
                            setattr(self, f"cal_{tag}_", c_)
                if hasattr(self, "cal_sql_") and hasattr(self, "cal_ns_"):
                    self.cal_warn_ = None
                    return self
            anchors = [(c, a, s + n) for c, a, s, n in anchors]      # 退回总量标定
        p0 = np.array([self._raw(cost, steps, a, c) for c, a, _ in anchors])
        y = np.array([m for _, _, m in anchors], float)
        ags = np.array([a for _, a, _ in anchors], float)
        self.cal_warn_ = None
        if len(anchors) >= 2 and len(set(ags)) >= 2:
            Z = np.column_stack([np.log(ags), np.ones(len(ags))])
            cal = np.linalg.lstsq(Z, np.log(y / p0), rcond=None)[0]         # (指数修正, 常数)
            if strict and abs(cal[0]) > self.MAX_EXP_CORRECTION:
                _en_ = os.environ.get("ADVISE_LANG", "en") != "zh"
                self.cal_warn_ = ((f"anchor exponent correction {cal[0]:+.3f} exceeds the "
                                   f"+/-{self.MAX_EXP_CORRECTION} threshold -- the two anchors are "
                                   f"inconsistent with each other, so the calibration is **discarded "
                                   f"entirely**. Re-measure the anchors (check the low-concurrency one "
                                   f"first: if db_time grows with cores, a parallel plan is inflating it).")
                                  if _en_ else
                                  (f"锚点指数修正 {cal[0]:+.3f} 超过阈值 "
                                   f"±{self.MAX_EXP_CORRECTION} —— 两个锚点彼此不自洽, "
                                   f"已**完全不校正**。建议复测锚点(尤其检查低并发那个: "
                                   f"若'核越多 db_time 越大', 多半是并行计划把 db_time 抬高了)。"))
                print("!! " + self.cal_warn_, file=sys.stderr)
                return self          # 完全不校正
            else:
                self.cal_ = cal
        else:
            self.cal_ = np.array([0.0, float(np.mean(np.log(y / p0)))])      # 只能定常数
        return self

    def predict(self, cost, steps, srv, threads_per_core, agents):
        C = effective_cores(srv, threads_per_core, agents)
        if hasattr(self, "cal_sql_") and hasattr(self, "cal_ns_"):
            s_, n_ = self._parts(cost, steps, agents, C)
            return (s_ * float(np.exp(self.cal_sql_[0] * np.log(agents) + self.cal_sql_[1]))
                    + n_ * float(np.exp(self.cal_ns_[0] * np.log(agents) + self.cal_ns_[1])))
        v = self._raw(cost, steps, agents, C)
        if hasattr(self, "cal_"):
            v = v * float(np.exp(self.cal_[0] * np.log(agents) + self.cal_[1]))
        return v


# ── 从本仓库的采集结果装配训练集 ────────────────────────────────────────────
def workload_ag_exponents(workloads):
    """每个训练负载自己的并发指数 —— 供零实测的悲观分位数策略用。

    做法: 固定核数档内对并发做 log-log 回归, 跨核数档取中位。
    实测 11 个单库负载得 0.401~0.953(标准差 0.150), 而这个指数无法从静态特征预测
    (五类特征全灭), 所以只能把它的分布当先验。
    """
    out = []
    for w in workloads:
        # 按物理核数分组, 不能按有效核 ——  之后 HT 增益随并发变,
        #   同一物理核数在不同并发下的有效核不, 按有效核分组会把每组打散成单点。
        byC = collections.defaultdict(list)
        for p in w["points"]:
            if p.get("sql"):
                byC[(p.get("srv", round(p["C"], 2)), p.get("th", 2))].append(
                    (p["ag"], p["sql"] + p["ns"]))
        es = []
        for _, v in byC.items():
            if len(v) < 3: continue
            v.sort()
            es.append(float(np.polyfit(np.log([x[0] for x in v]),
                                       np.log([x[1] for x in v]), 1)[0]))
        if es: out.append(float(np.median(es)))
    return np.array(out, float)


# ── 从锚点的系统指标反推该负载的并发指数 ──────────────────────
#   受 URSA(arXiv:1912.11963)启发: 它不"预测"scaling surface, 而是用一个 base config
#   上实测的系统指标把负载归类, 套该类的曲面。我们一直做的是"用静态特征预测并发指数",
#   五类特征全灭(, 最好 LOO MAE 0.106 ≈ 常数基线)。
#   而锚点那一次运行本来就产出 97 个字段, 我们此前只用了 db_time, 其余全扔了。
#
#   实测(嵌套留一, 内层只在训练库之间选特征): MAE 0.0582, 常数基线 0.1048(−44%)。
#   选中的是 `db_blks_hit`(缓冲池命中次数)。步数固定 24500 时它正比于"每步触碰的缓冲页数",
#   缓冲流量越大 -> 并发下争用越强 -> 并发指数越高, 机制上讲得通。
#
#   关系式必须在与客户锚点相同的配置上拟合 —— db_blks_hit 在同一负载内跨配置
#     变化 1.38~7.07 倍, 与跨负载差异同量级; 只有 like-for-like 比较才有意义。
#     实测 28 个锚点位置里 27 个可用(MAE 0.030~0.079), 仅 16核x8并发 边缘不过(0.0802)。
SYSIDX_FEATURE = "db_blks_hit"

def anchor_exponent(metric_value, srv, th, agents, root="."):
    """用锚点实测的系统指标反推并发指数。返回 (指数, 说明) 或 (None, 原因)。"""
    import glob as _g, re as _re, os as _os
    if not metric_value or metric_value <= 0:
        return None, f"锚点缺 {SYSIDX_FEATURE}"
    meta = json.load(open(f"{root}/pdb_meta.json"))
    ex = workload_ag_exponents(load_repo_workloads(root))
    # 收集训练库在各个配置上的该指标
    M = collections.defaultdict(dict)
    for i, w in enumerate(meta):
        for p in _g.glob(f"{root}/{w}_grid/s*/point.json"):
            mm = _re.match(r"s(\d+)_l4_t(\d)_a(\d+)_b0.2$", _os.path.basename(_os.path.dirname(p)))
            if not mm: continue
            d = json.load(open(p)); v = d.get(SYSIDX_FEATURE)
            if v: M[(int(mm[1]), int(mm[2]), int(mm[3]))][i] = float(v)
    # 找与客户锚点最近的训练配置(log 距离), 且要覆盖全部训练库
    full = {k: v for k, v in M.items() if len(v) == len(meta)}
    if not full: return None, "训练库缺该指标"
    key = min(full, key=lambda k: (abs(np.log(max(k[0],1)/max(srv,1)))
                                   + abs(np.log(max(k[2],1)/max(agents,1)))
                                   + (0.0 if k[1] == th else 0.5)))
    xs = np.array([np.log(full[key][i]) for i in range(len(meta))])
    ys = np.array([ex[i] for i in range(len(meta))])
    if xs.std() < 1e-12: return None, "该配置上指标无区分度"
    a, b = np.polyfit(xs, ys, 1)
    pred = float(a * np.log(metric_value) + b)
    lo, hi = float(ys.min()), float(ys.max())
    clipped = min(max(pred, lo), hi)          # 夹在训练库实测范围内, 不外推
    note = (f"锚点 {SYSIDX_FEATURE}={metric_value:,.0f}, 对齐训练配置 "
            f"s{key[0]}/t{key[1]}/a{key[2]} -> 并发指数 {clipped:.3f}"
            f"{' (已夹到训练范围 %.3f~%.3f)' % (lo, hi) if clipped != pred else ''}")
    return clipped, note


def load_repo_workloads(root="."):
    lab = [x for x in json.load(open(f"{root}/train_label_noise.json")) if x.get("median", 0) > 0]
    c0 = collections.defaultdict(list)
    for x in lab: c0[int(x["qid"])].append(x["median"])
    qcost = {q: float(np.mean(v)) for q, v in c0.items()}
    meta = json.load(open(f"{root}/pdb_meta.json"))
    out = []
    for w, m in meta.items():
        ids = [q["question_id"] for q in json.load(open(f"{root}/questions/{w}_questions.json"))]
        cost = np.array([qcost[q] for q in ids if q in qcost])
        pts = []
        for p in glob.glob(f"{root}/{w}_grid/s*/point.json"):
            mm = re.match(r"s(\d+)_l4_t(\d)_a(\d+)_b0.2$", os.path.basename(os.path.dirname(p)))
            if not mm: continue
            # : th=1 的 308 点只用于定 HT 增益表(ht_gain_fit.json), 不并入拟合。
            #   并入后反推从 88.8% 掉到 85.7%, 正向中位反而小改善 —— 退化集中在 2~10 核,
            #   因为这些点是经 effective_cores 折算后进模型的, 会把核数维度拉陡,
            #   而反推正是靠核数维度选档。两件事分开: 表留下, 点不并入。
            if int(mm[2]) != 2: continue
            d = json.load(open(p)); bk = d.get("db_time_breakdown_s") or {}
            # / 数据有效性过滤: 丢掉"每条查询平均 SQL 时间 > p99"的点。
            #   均值不可能大于 p99, 除非尾部极重 —— 实测这 7 个点 ap_tail_share_pct 中位 92.0%
            #   (其余 301 点 10.0%), 全部落在 pdb_european_football_2。
            #   判决实验已确认是可复现的负载病理(并行照旧复测 4576.2 vs 4470.8, +2.4%),
            #   不是飘忽测量; 关掉并行只降 17%, 所以也不是并行计划造成的。
            #   这是测量有效性修复, 不是调参: 出族确认 bal_test 零实测 39.9→39.3%、
            #   2 锚 6.3→5.9%、零实测 q=.9 达标率 95.2→100.0%, 1 锚 15.8→16.1%(+0.3, 噪声级)。
            if (bk.get("sql", 0) * 1000.0 / max(d.get("sql_exec_n") or 1, 1)
                    > (d.get("sql_exec_p99_ms") or float("inf"))):
                continue
            pts.append(dict(C=effective_cores(int(mm[1]), int(mm[2]), int(mm[3])), ag=int(mm[3]),
                            srv=int(mm[1]), th=int(mm[2]),
                            sql=bk.get("sql", 0),
                            ns=bk.get("vec_mem", 0) + bk.get("vec_value", 0) + bk.get("tp_txn", 0),
                            **{f"q{p_}": d.get(f"sql_exec_p{p_}_ms") for p_ in QP}))
        if len(cost) and pts:
            out.append(dict(cost=cost, steps=m["steps"], points=pts))
    return out


if __name__ == "__main__":
    W = load_repo_workloads()
    print(f"训练集: {len(W)} 个单库负载, {sum(len(w['points']) for w in W)} 点")
    M = PhoebePredictor().fit(W)
    print(f"  放大 amp = {np.exp(M.ba_[-1]):.4f} · " + " · ".join(f"{k}^{M.ba_[i]:+.3f}" for i, k in enumerate(M.FA)))
    print(f"  非SQL     = {np.exp(M.bn_[-1]):.2f} · " + " · ".join(f"{k}^{M.bn_[i]:+.3f}" for i, k in enumerate(M.FN)))
    # 在 test 上验收
    tc = np.array([x["median"] for x in json.load(open("test_label_noise.json")) if x.get("median", 0) > 0])
    T = []
    for p in glob.glob("bal_test/s*/point.json"):
        mm = re.match(r"s(\d+)_l4_t(\d)_a(\d+)_b0.2$", os.path.basename(os.path.dirname(p)))
        if not mm: continue
        d = json.load(open(p))
        T.append((int(mm[1]), int(mm[2]), int(mm[3]), d["db_time_s"], d.get("completion_seconds")))
    print(f"\n验收: test {len(T)} 点 (4 个陌生库, 未参与拟合)")
    def sc(anchor_idx):
        A = [(effective_cores(T[i][0], T[i][1], T[i][2]), T[i][2], T[i][3]) for i in anchor_idx]
        m = PhoebePredictor(); m.ba_, m.bn_ = M.ba_, M.bn_
        if A: m.calibrate(tc, STEPS_REF, A)
        e, wall = [], sum(T[i][4] for i in anchor_idx)
        for j, (s, th, ag, y, _) in enumerate(T):
            if j in anchor_idx: continue
            e.append(abs(m.predict(tc, STEPS_REF, s, th, ag) - y) / y)
        return np.array(e), wall
    # 推荐锚点: 两个高并发且并发档不同
    pick = []
    for want in ((64, 8), (128, 16)):
        for j, (s, th, ag, _, _) in enumerate(T):
            if ag == want[0] and s == want[1] and th == 2: pick.append(j); break
    for nm, ix in [("零实测", []), ("推荐 2 锚点(ag64/srv8 + ag128/srv16)", pick)]:
        e, wl = sc(ix)
        print(f"  {nm:38s} 墙钟 {wl:4.0f}s  中位 {100*np.median(e):5.1f}%  "
              f"±10% {100*np.mean(e<.1):5.1f}%  ±20% {100*np.mean(e<.2):5.1f}%")
