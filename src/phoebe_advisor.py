#!/usr/bin/env python3
"""PhoebeDB 配置顾问 —— 反推: 给定目标, 求最省的配置。

正向模型是 phoebe_predictor(配置 -> db_time)。反推有三种问法, 都在这里实现:

  A. 达标最省配置: "我要 db_time <= X 秒, 并发 M, 最少要几个核?"
  B. 预算内最优:   "我只有 N 个核, 能撑多少并发还满足 X?"
  C. 等值线:       给出 (核数, 并发) 平面上满足目标的边界, 供画图/选型

为什么不能直接解析求逆: 正向模型里 sql 部分要对整条成本分布求和,
且带锚点校正项, 没有闭式反函数。但它在核数上单调(C 指数为负),
所以对每个并发档做二分搜索即可, 且有唯一解。

精度继承正向模型: 跨最低-最高并发取 2 个锚点时中位 8.2%。
   因此建议以目标的 1.2 倍作为安全余量来选配置, 而不是卡着目标选。
"""
import numpy as np


# ── 预测值修正系数(2026-10-09, prereg_bias.json md5 530ef62f) ──────────────
# 来源: 训练侧 10 库留一的有符号残差中位 = -2.6%(10 库里 9 个为负), k = 1/0.974。
#   全程未使用 224 个评估目标定 k; 事后发现它正好落在评估集扫描峰值上, 属独立印证。
# 效果(tp0 + 剔 EF2 口径, 224 个预注册墙钟目标):
#   推荐达标率 181/220 = 82.3%  ->  187/220 = 85.0%
#   正向 424 点 |误差| 中位 6.5% -> 6.9%(预注册允许劣化 <= 1.0 个百分点)
#   "确实无解"判定 51 -> 51(不变)—— 本系数只乘在预测值上, 不碰选档判据,
#   这是它与余量规则(safety)的根本区别: 余量会把"确实无解"从 51 砸到 4。
# 局限(勿写成"修正了系统性低估"): 评估集全局有符号偏差只有 -0.1%, 低估是局部的
#   (2~6 核 -2.6%~-6.2%, 11~16 核 +1%~+4.4%)。推荐都发生在低核区, 常数 k 在那里修对了,
#   代价是高核区正向 ±15% 覆盖 82.3% -> 79.5%。按核数倾斜的修正训练/评估符号相反, 已否决。
BIAS_K = 1.0268

class PhoebeAdvisor:
    def __init__(self, predictor, cost, steps, ht_gain=1.24):
        self.m = predictor
        self.cost = np.asarray(cost, float)
        self.steps = int(steps)
        self.ht = ht_gain

    def _eff(self, srv, th, agents=None):
        """有效核数。超线程增益随并发变, agents 为 None 时退回构造时的常数。"""
        if th != 2: return float(srv)
        if agents is None: return srv * self.ht
        import importlib.util as _iu, os as _o
        global _PP
        try: _PP
        except NameError:
            _sp = _iu.spec_from_file_location("_pp_for_eff",
                     _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "phoebe_predictor.py"))
            _PP = _iu.module_from_spec(_sp); _sp.loader.exec_module(_PP)
        return srv * _PP._ht_gain(agents)

    def db_time(self, srv, th, agents):
        return BIAS_K * float(self.m.predict(self.cost, self.steps, srv, th, agents))

    # ── A. 达标最省核数 ───────────────────────────────────────────────
    def min_cores(self, target_s, agents, th=2, lo=1, hi=64):
        """满足 db_time <= target 的最少核数。返回 None 表示 hi 核也达不到。"""
        if self.db_time(hi, th, agents) > target_s:
            return None
        if self.db_time(lo, th, agents) <= target_s:
            return lo
        while hi - lo > 1:                      # db_time 对核数单调减, 二分有唯一解
            mid = (lo + hi) // 2
            if self.db_time(mid, th, agents) <= target_s:
                hi = mid
            else:
                lo = mid
        return hi

    # ── B. 给定核数能撑多少并发 ───────────────────────────────────────
    def max_agents(self, target_s, srv, th=2, lo=1, hi=512):
        """满足 db_time <= target 的最大并发。db_time 对并发单调增。

        返回 (并发, 备注)。备注非 None 时表示答案落在模型的失效区:
        ρ = 并发/有效核 < MIN_RHO 的轻载地板区里 db_time 几乎不随并发变(实测
        ag=2 与 4 只差 0.6%), 而模型是幂律、没有地板项, 会把这一段画成一路下降,
        于是二分搜索会返回一个偏小且不可信的并发。见 `useful_cores`。
        """
        if self.db_time(srv, th, lo) > target_s:
            return None, "连 1 并发都达不到"
        if self.db_time(srv, th, hi) <= target_s:
            return hi, None
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.db_time(srv, th, mid) <= target_s:
                lo = mid
            else:
                hi = mid
        rho = lo / self._eff(srv, th, lo)
        if rho < self.MIN_RHO:
            return lo, (f"ρ={rho:.2f} < {self.MIN_RHO}: 这么少的并发占不满 {srv} 核, "
                        f"db_time 已到地板、不再随并发变, 该结果不可信。"
                        f"实际能撑的并发多半更高 —— 减少核数或直接实测。")
        return lo, None

    # ── C. 等值线 ────────────────────────────────────────────────────
    def isoline(self, target_s, agent_levels, th=2, max_cores=64):
        """返回 [(并发, 核数 或 None)]。核数被轻载地板卡住时返回 None。"""
        out = []
        for a in agent_levels:
            cap = min(max_cores, self.useful_cores(a, th))
            out.append((a, self.min_cores(target_s, a, th, hi=cap)))
        return out

    # ── 模型自身误差随被预测点的负载强度变化 ──────────────────
    #   实测: 误差对 ρ = 并发/有效核数 呈 U 型, 最低点在 ρ≈2.4。
    #   在两个独立评估集(16 前瞻点 + 36 bal_test 点, 共 52 个配置)上拟合:
    #       err(ρ) = 0.0805 + 0.0289 * |log(ρ / 2.40)|      R² = 0.393
    #   两集残差中位分别 2.5 / 1.9 个百分点。Spearman(|log(ρ/3)|, 误差)
    #   = +0.572 (p=0.02, 前瞻) 和 +0.731 (p<1e-4, bal_test) —— 同号显著, 可用。
    #   取值范围: ρ=2.4 时 8.1%, ρ=0.5 时 12.6%, ρ=30 时 15.4%。
    ERR_A, ERR_B, ERR_RHO0 = 0.0805, 0.0289, 2.40
    ERR_FLOOR, ERR_CEIL = 0.08, 0.21          # 封顶用旧的全局保守值, 不让外推跑飞

    # 曾在 ρ∈[0.9,1.6] 加过一个 0.15 的误差下限, 已撤销:
    #   加的时候该箱只有 2 个配置(test:ag8/C8.0 与 test:ag16/C16.0, 两者 ρ 都恰好 1.00
    #   且都是 th=1), 实测中位 15.4%。补采 6 个中间核数档(ag=8 的 srv5/6, ag=16 的
    #   srv9~12)后, 该箱变成 5 个 ρ 值 / 8 个配置, 实测中位掉到 7.3% ——
    #   15.4% 是 2 配置的假象。留着那个下限反而让公式在该箱高估 7.7 个百分点,
    #   成为全表最不准的一箱。撤销后全表没有任何一箱低估超过 3 个百分点。

    @classmethod
    def expected_err(cls, agents, eff_cores):
        """【已弃用, 勿用于选档】拟合数据含评测集 bal_test, 仅训练集重拟不复现(R² 0.393->0.142)。
        保留只为 options 的风险标签向后兼容。生产选档走 recommend 的最近档分支。

        这个配置上模型的预期中位相对误差。"""
        rho = max(agents / max(eff_cores, 1e-9), 1e-6)
        e = cls.ERR_A + cls.ERR_B * abs(np.log(rho / cls.ERR_RHO0))
        return float(min(max(e, cls.ERR_FLOOR), cls.ERR_CEIL))

    # ── 轻载地板: 核数不能无限加 ────────────────────────────────────────
    #   实测(grid_lowconc, 同一配置 C=14.9 只变并发):
    #       ag=2 -> db_time 674s    ag=4 -> 678s    ag=8 -> 732s
    #   ag 从 2 到 4 只差 0.6%(并发指数 0.01)—— 轻载区 db_time 是个常数地板,
    #   因为没有排队, db_time 就等于纯服务时间。加核压不下去。
    #   但正向模型是幂律, 没有地板项: 它相信 ag=4 加到 128 核能到 81s。
    #   于是顾问会承诺物理上做不到的配置(目标 200s @ 4 并发 -> 推荐 50 核)。
    #   护栏: 推荐配置自身的 ρ 不得低于 MIN_RHO, 否则判为"这个并发下再加核也没用"。
    #
    #   阈值由实测定(grid_lowconc 满档后): 把所有并发档的相邻核数对
    #     汇总(128 组), 按该对的 ρ 中点分箱, 看局部弹性 d log(db_time)/d log(C):
    #         ρ∈[0,0.3)   -0.02   几乎无用      ρ∈[1.2,2)  -0.71  有用
    #         ρ∈[0.3,0.5) -0.21   部分          ρ∈[2,4)    -0.56  有用
    #         ρ∈[0.5,0.8) -0.24   部分          ρ∈[4,8)    -0.63  有用
    #         ρ∈[0.8,1.2) -0.11   几乎无用      ρ>=8       -0.75  有用
    #     (-1 = 理想线性加速, 0 = 完全无用)
    #     转折在 ρ≈1.2, 不是 0.5。 而顾问的正向模型是纯幂律 C^ce (ce≈-0.7),
    #     它在整个区间都按 -0.7 预测 —— 在实际只有 -0.11 的地方把收益高估了约 7 倍。
    #     所以护栏必须覆盖到 ρ<1, 原来的 0.5 太松。
    #     取 1.0 而不是 1.2: ρ∈[1,2) 是过渡带(四分位 [-1.15, 0.12], 方差大),
    #     在那里加核有时有用, 拒绝掉代价太大; 1.0 以下则一致地无用。
    MIN_RHO = 1.0
    TOL = 0.15             # 达标 = |实测 - 目标| <= TOL, 双向(2026-10-01 与计分口径统一)
                           #   曾是 0.10, 但用户定的达标线与 89.7% 那个数都按 ±15% 算,
                           #   工具比计分严会多报无解(224 组里 17 个误判无解)。
    OVERSHOOT_CAP = 1.10   # 兼容旧名

    def useful_cores(self, agents, th=2, min_rho=None):
        """这个并发下加核还有意义的最大核数(整数 srv)。"""
        r = self.MIN_RHO if min_rho is None else min_rho
        ht = self.ht if th == 2 else 1.0
        return max(1, int(agents / r / ht))

    # ── 零实测: 悲观并发指数 ──────────────────────────────────
    #   零实测的真问题不是"中位误差 40.9%", 而是点估计选出的配置只有 38.1% 真能达标
    #   —— 误差是系统性低估( 中位偏差 −43.3%), 不是随机散布。
    #   并发指数不可从静态特征预测(五类特征全灭), 但范围已知:
    #   11 个单库负载实测 0.401~0.953。用其 q 分位替代拟合值即可修掉这个低估。
    #
    #   实测(bal_test 42 点):
    #       q=50(点估计)  达标 38.1%  平均 7.42 核
    #       q=90          达标 95.2%  平均 10.52 核   ← 中位多用核数为 0
    #       q=100         达标 100%   平均更高
    #
    #   与 `expected_err(ρ)` 的余量互斥: 两者针对同一个误差来源,
    #      叠加只多 2.4 个百分点达标率却多 16~27% 的核。所以
    #      `zero_shot_quantile` 一旦设定, `recommend` 自动把 safety 固定为 1.0。
    #
    #   q 的具体取值尚未定稿: "名义 q ≈ 实际达标率"只在 bal_test 上验过一次;
    #      单库负载那侧核数档太粗(3 档)验不了。补档采集完成后再定。
    ZS_EXPONENTS = None          # 训练负载的并发指数样本, 由 fit 方使用方注入

    def set_zero_shot(self, exponents, q, ag_feature_index):
        """启用悲观指数。exponents: 训练负载的并发指数数组; q: 风险分位(0~1)。"""
        e = np.asarray(exponents, float)
        self._zs_shift = float(np.quantile(e, q)) - float(self.m.ba_[ag_feature_index])
        self._zs_idx = int(ag_feature_index)
        self._zs_q = float(q)
        b = np.asarray(self.m.ba_, float).copy(); b[self._zs_idx] += self._zs_shift
        self.m = _ShiftedPredictor(self.m, b)
        return self

    # ── 选型: 给三档阶梯, 让客户自己权衡 ──────────────────────────────
    def options(self, target_s, agents, th=2, n=3, max_cores=64, model_err=None):
        """返回 n 个满足目标的配置, 按核数从少到多 —— 让客户在"省钱"与"稳"之间自己选。

        为什么不给单一答案: 只报一个"卡着 1.2x 余量"的数字, 客户既没得选,
        也看不出多花的核换来多少余量。阶梯上的数值不写死 ——
        核数是整数, 可达的预测值本来就是离散的, 取"恰好达标"往上数 n 个即可,
        间距由系统本身决定(典型 8~15% 一档)。

        model_err: 模型自身的中位相对误差, 用来把"余量"翻译成风险等级
            (余量 < 1+err 的那档是有风险的)。
            默认 None = 每一档各自按它自己的负载强度 ρ 算(`expected_err`) ——
            以前是全局常数 0.21, 但实测误差在 ρ≈2.4 处是 8%、在两端升到 13~15%,
            用一个常数会同时高估中段的风险、低估两端的风险。
            传一个数则退回旧的全局口径。
        """
        max_cores = min(max_cores, self.useful_cores(agents, th))   # 轻载地板护栏, 见 useful_cores
        rows = []
        for c in range(1, max_cores + 1):
            v = self.db_time(c, th, agents)
            if v <= target_s:
                rows.append((c, v))
        if not rows:
            return []
        rows.sort(key=lambda z: -z[1])            # 预测耗时从大到小 = 核数从少到多

        # 按余量等比铺开, 不按核数连号。
        # 核数多时单核的边际收益很小(实测 64 并发下相邻核只差 3% 余量), 连号选出来
        # 三个几乎一样的方案, 等于没得选。改为在 [最紧, 最松] 的余量区间上取 n 个等比点,
        # 再各自吸附到最近的可行核数 —— 档距由系统本身的边际收益决定, 不写死。
        # 铺开区间用的那个 err: 若调用方没指定, 用"这个并发下的典型配置"的 err。
        cap_err = model_err if model_err is not None else\
            self.expected_err(agents, self._eff(rows[len(rows)//2][0], th, agents))
        lo_m = target_s / rows[0][1]              # 最省核那档的余量
        hi_m = target_s / rows[-1][1]             # 最多核那档的余量
        hi_m = min(hi_m, 1 + 4 * cap_err)         # 再多也没意义, 封顶在 4 倍模型误差
        if hi_m <= lo_m * 1.01 or n == 1:
            picks = rows[:n]
        else:
            want = np.geomspace(lo_m, hi_m, n)
            picks, seen = [], set()
            for w in want:
                c, v = min(rows, key=lambda z: abs(target_s / z[1] - w))
                if c not in seen:
                    seen.add(c); picks.append((c, v))
            # 吸附后可能撞车, 用未选中的补足
            for c, v in rows:
                if len(picks) >= n: break
                if c not in seen:
                    seen.add(c); picks.append((c, v))
            picks.sort(key=lambda z: z[0])
        out = []
        for c, v in picks[:n]:
            margin = target_s / v
            # 每一档用它自己的 ρ 算预期误差: 核数不同 -> ρ 不同 -> 风险门槛不。
            err = model_err if model_err is not None else\
                self.expected_err(agents, self._eff(c, th, agents))
            if margin >= 1 + 2 * err:   risk = "稳"
            elif margin >= 1 + err:     risk = "适中"
            else:                       risk = "偏紧"
            out.append(dict(srv=c, th=th, agents=agents, pred_db_time=v,
                            target=target_s, margin=margin, risk=risk,
                            model_err=err, rho=agents / self._eff(c, th, agents),
                            headroom_pct=100 * (margin - 1)))
        return out

    # ── 选型: 在等值线上按成本函数挑最优 ───────────────────────────────
    def recommend(self, target_s, agents, th=2, safety=None, max_cores=64):
        """safety: 按 target/safety 来选, 抵消模型自身误差。

        2026-10-07 更正文档: safety=None(默认)= 最近档, 即选 |log(预测/目标)|
        最小的那一档, 见下面 nearest 分支。不是 ρ 自适应余量 —— 那条路
        ( 的 expected_err)被查出是在含 bal_test 的 52 个点上拟的, 仅用训练集
        重拟 R² 从 0.393 掉到 0.142 且 ρ0 顶到搜索边界, U 形不复现, 已弃用;
        它现在只残留在 options 里给风险标签用, 不参与选档。
        固定 1.2x 的老做法在中段过度保守(多买核), 在两端又不够。
        因为余量本身影响选出的核数、核数又反过来决定 ρ, 这里迭代两轮即可收敛。
        """
        # 轻载地板护栏: 超过 useful_cores 再加核也压不下 db_time(实测 ag=2 与 4 相差 0.6%),
        # 而正向模型是幂律、没有地板项, 会一路承诺下去。把搜索上界卡在这里。
        cap = self.useful_cores(agents, th)
        capped = cap < max_cores
        max_cores = min(max_cores, cap)
        if getattr(self, "_zs_shift", None) is not None:
            # 悲观指数已修掉系统性低估, 再加 ρ 余量是重复计费
            safety, auto, s = 1.0, False, 1.0
        else:
            auto = False
            s = "nearest" if safety is None else safety   # None = 最近档(默认)
        need = None
        # 2026-09-29 用户决定: 默认选预测值离目标最近的一档, 允许略超。
        #   旧规则"预测必须 <= 目标/余量"是硬约束, 只能往下取, 系统性多买。
        #   训练库留一(176 个同批目标)对比:
        #       规则        达标率     实测/目标   |实测/目标-1|  超标幅度        平均核
        #       最优解      176/176   0.87x       13.0%        —             5.67
        #       最近档      121/176   0.94x       12.3%        中位+7.9%      5.32
        #       余量 1.05   166/176   0.76x       23.9%        —             6.60
        #   最近档离目标只有一半的偏差, 少买 19% 的核, 甚至优于"不许超"的最优解
        #   (因为最优解被硬约束绑着只能往下取)。代价: 达标率 94%->69%, 最大超标 +51%。
        nearest = (s == "nearest")
        if nearest:
            import numpy as _np
            best, bd = None, None
            for cc in range(1, int(max_cores) + 1):
                d = abs(_np.log(self.db_time(cc, th, agents) / target_s))
                if bd is None or d < bd: best, bd = cc, d
            # 2026-09-29: 达标 = |实测 - 目标| <= TOL(双向)。若没有任何一档的预测值
            #   落进 [1-TOL, 1+TOL]x目标, 说明目标卡在两档之间, 这台机器上无解 ——
            #   直说无解, 并给出最接近的两档各会跑到多少秒, 不硬塞一个。
            if best is not None:
                pb = self.db_time(best, th, agents)
                if abs(pb / target_s - 1.0) > self.TOL:
                    lo_c = max((cc for cc in range(1, int(max_cores)+1)
                                if self.db_time(cc, th, agents) > target_s), default=None,
                               key=lambda cc: self.db_time(cc, th, agents) if cc else 0)
                    slower = [cc for cc in range(1, int(max_cores)+1)
                              if self.db_time(cc, th, agents) > target_s * (1+self.TOL)]
                    faster = [cc for cc in range(1, int(max_cores)+1)
                              if self.db_time(cc, th, agents) < target_s * (1-self.TOL)]
                    opts = []
                    if slower: 
                        cc = min(slower, key=lambda x: self.db_time(x, th, agents))
                        opts.append((cc, self.db_time(cc, th, agents)))
                    if faster:
                        cc = max(faster, key=lambda x: self.db_time(x, th, agents))
                        opts.append((cc, self.db_time(cc, th, agents)))
                    opts.sort()
                    return dict(ok=False, infeasible_gap=True, nearest=best,
                                options=[dict(srv=cc, pred=pv, dev=pv/target_s-1) for cc, pv in opts],
                                reason=("目标卡在两档之间, 这台机器上没有任何核数能落进 "
                                        f"±{100*self.TOL:.0f}%。" +
                                        "  ".join(f"{cc} 核 -> {pv:,.0f}s ({100*(pv/target_s-1):+.1f}%)"
                                                  for cc, pv in opts)))
            need, s, auto = best, 1.0, False
        for _ in range(3):
            if need is not None and not auto:
                break
            need = self.min_cores(target_s / s, agents, th, hi=max_cores)
            if need is None or not auto:
                break
            s_new = 1.0 + self.expected_err(agents, self._eff(need, th, agents))
            if abs(s_new - s) < 1e-3:
                s = s_new; break
            s = s_new
        if need is None:
            if capped:
                return dict(ok=False, capped_by_rho=True, useful_cores=cap,
                            reason=(f"{agents} 并发下本工具最多推荐 {cap} 核(ρ={self.MIN_RHO}), "
                                    f"达不到 {target_s:.0f}s。\n"
                                    f"    ⚠ 这个上限是**负载相关**的, 不是物理定律(): "
                                    f"在 dev 负载上 ρ<1.2 时加核确实几乎无用(弹性 −0.11), "
                                    f"但在 4 个陌生库的出族负载上加核**一路有用**(弹性 −0.54~−1.42) —— "
                                    f"那里 ag=8 从 7.4 核加到 19.8 核仍把 db_time 从 1768s 降到 1067s(−40%)。\n"
                                    f"    想知道你的负载属于哪种: 在 {cap} 核和 2x{cap} 核各测一个点比一比。"
                                    f"要么提高并发, 要么放宽目标。"))
            if nearest:
                fast = self.db_time(int(max_cores), th, agents)
                return dict(ok=False, reason=(f"{max_cores} 核也达不到 {target_s:.0f}s —— 最快的一档预测 "
                            f"{fast:.0f}s, 比目标慢 {100*(fast/target_s-1):.0f}%, "
                            f"超过允许的 {100*(self.OVERSHOOT_CAP-1):.0f}%。请放宽目标或提高并发。"))
            return dict(ok=False, reason=f"{max_cores} 核也达不到 {target_s:.0f}s(含 {s:.2f}x 余量)")
        pred = self.db_time(need, th, agents)
        err = self.expected_err(agents, self._eff(need, th, agents))
        note = ((f"选**预测值离目标最近**的一档(可能略超目标; 训练库实测中位超 +7.9%, 最大 +51%)"
                 if nearest else f"按 {s:.2f}x 余量选核") +
                f"(ρ={agents/self._eff(need, th, agents):.1f} 处预期中位误差 {100*err:.0f}%)")
        # : 上限截断过搜索空间时必须说出来 —— 否则用户拿到一个"看起来正常"的建议,
        #   却不知道更大的机器被悄悄排除了。实测这个截断在出族负载上会少拿 40% 的收益。
        if capped and need >= cap:
            note += (f"\n    ⚠ 搜索上界被 ρ 护栏截断在 {cap} 核(MIN_RHO={self.MIN_RHO}), "
                     f"更大的机器**没有被考虑**。\n"
                     f"      这个护栏是负载相关的: dev 负载上那里加核几乎无用(弹性 −0.11), "
                     f"但出族负载上仍有 −0.54~−1.42。若你的目标很紧, "
                     f"在 {cap} 核与 {2*cap} 核各测一点再决定。")
        return dict(ok=True, srv=need, th=th, agents=agents, pred_db_time=pred,
                    target=target_s, margin=target_s / pred, safety=s,
                    model_err=err, rho=agents / self._eff(need, th, agents),
                    capped_by_rho=bool(capped and need >= cap), useful_cores=cap, note=note)


class _ShiftedPredictor:
    """把 ba_ 换成平移后的系数, 其余行为完全委托给原预测器。"""
    def __init__(self, base, ba):
        self._b = base; self.ba_ = ba; self.bn_ = base.bn_
    def __getattr__(self, k):
        return getattr(self._b, k)
    def predict(self, cost, steps, srv, th, agents):
        import numpy as _np
        m = type(self._b)(); m.ba_, m.bn_ = self.ba_, self.bn_
        if hasattr(self._b, "cal_"): m.cal_ = self._b.cal_
        return m.predict(cost, steps, srv, th, agents)
