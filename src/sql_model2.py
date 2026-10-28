#!/usr/bin/env python3
"""SQL 放大项的改进: 成本指数随负载 ρ=ag/C 变。

实测依据 (rc3+rc4, 900 个 (格点 x 分位) 样本, 按 (ag,C) 分格各拟一次):
    成本指数范围  -0.040 ~ +0.102      与 log(ρ) 相关 r = +0.68
    回归          成本指数 = -0.008 + 0.019·log(ρ)
而现模型拟出一个全局值 +0.097 —— 那是 ρ≈20 那一端的值, 被用在所有负载上。
后果: 低负载下模型给便宜查询的放大偏小, 实测 预测/实测 = 0.71x (0-1ms 档)。

机理 (Lazowska《Quantitative System Performance》的渐近界):
  轻载 R ≈ D (服务需求本身), 放大与 D 无关 -> 成本指数 ≈ 0；
  重载 R ≈ N·Dmax - Z, 贵查询占住资源更久、被排队的代价更大 -> 成本指数 > 0。
  所以成本指数本来就该是 ρ 的函数, 一个常数拟不住两个区制。
 在 db_time 总量层面用双渐近线(ABA)撬动过同一个洞; 这里是同一思想
落到单条查询的放大倍数上, 但用【交互项】而不是软max —— 因为单查询没有"总量"可分段。

改法: log(amp) 里加一项 log(cost)·log(ag/C), 只多一个参数, 不引入新测量。
非 SQL 项沿用 NSPredictor(核指数随并发变)。
"""
import collections, math
import numpy as np
from phoebe_predictor import PhoebePredictor, STEPS_REF, QP
from ns_model2 import NSPredictor


class SQLPredictor(NSPredictor):
    USE_RHO = True

    @staticmethod
    def _xa(oc, B, ag, C, use_rho=True):
        lo, lb = math.log(max(oc, 1e-12)), math.log(max(B, 1e-12))
        la, lc = math.log(float(ag)), math.log(float(C))
        base = [lo, lb, la, lc, 1.0]
        return np.array(base[:1] + [lo * (la - lc)] + base[1:]) if use_rho else np.array(base)

    def fit(self, workloads, anchor_rows=None):
        super().fit(workloads, anchor_rows=anchor_rows)       # 非 sql 项 + 原版 ba_ 作兜底
        rows = []
        for wi, w in enumerate(workloads):
            B = float(np.mean(w["cost"])) * w["steps"] / 1000.0
            for pc in QP:
                oc = float(np.percentile(w["cost"], pc))
                for p in w["points"]:
                    v = p.get(f"q{pc}")
                    if v and oc > 0: rows.append((oc, B, p["ag"], p["C"], v / oc, wi))
        if len(rows) < 30: self.ba2_ = None; return self
        X = np.array([self._xa(*r[:4], self.USE_RHO) for r in rows])
        y = np.log(np.array([r[4] for r in rows], float))
        gid = [r[5] for r in rows]
        cnt = collections.Counter(gid)
        wt = np.sqrt(1.0 / np.array([cnt[g] for g in gid], float))
        ex = (anchor_rows or ([], []))[0]
        if ex:
            X = np.vstack([X] + [self._xa(*e[:4], self.USE_RHO) for e in ex])
            y = np.concatenate([y, np.log(np.array([e[4] for e in ex], float))])
            wt = np.concatenate([wt, np.ones(len(ex))])       # 锚点权重 1.0, 同原版
        self.ba2_ = np.linalg.lstsq(X * wt[:, None], y * wt, rcond=None)[0]
        return self

    def _amp_vec(self, cost, steps, ag, C):
        c = np.asarray(cost, float)
        B = float(np.mean(c)) * steps / 1000.0
        if getattr(self, "ba2_", None) is None:
            amp = math.exp(self.ba_[-1]) * np.ones_like(c)
            vals = {"cost": c, "B": B, "ag": float(ag), "C": float(C)}
            for i, k in enumerate(self.FA):
                amp = amp * np.maximum(np.asarray(vals[k], float), 1e-9) ** self.ba_[i]
            return amp
        lo = np.log(np.maximum(c, 1e-12)); la, lc = math.log(float(ag)), math.log(float(C))
        b = self.ba2_
        if self.USE_RHO:
            z = b[0]*lo + b[1]*lo*(la-lc) + b[2]*math.log(max(B,1e-12)) + b[3]*la + b[4]*lc + b[5]
        else:
            z = b[0]*lo + b[1]*math.log(max(B,1e-12)) + b[2]*la + b[3]*lc + b[4]
        return np.exp(z)

    def _parts(self, cost, steps, ag, C):
        c = np.asarray(cost, float)
        per = c / 1000.0 * self._amp_vec(c, steps, ag, C)
        if getattr(self, "tau_", None) is not None:
            per = np.maximum(per, self.tau_[0] * float(ag) ** self.tau_[1] / 1000.0)
        sql = float(np.sum(per)) * steps / len(c)
        return sql, self._ns2(ag, C) * steps / STEPS_REF

    def _raw(self, cost, steps, ag, C):
        c = np.asarray(cost, float)
        sql = float(np.sum(c / 1000.0 * self._amp_vec(c, steps, ag, C))) * steps / len(c)
        return sql + self._ns2(ag, C) * steps / STEPS_REF


class SQLOnlyPredictor(SQLPredictor):
    """只改 sql 项, 非 sql 用原版 ag^a·C^b —— 用来拆开两处改动各自的贡献。"""
    def _ns2(self, ag, C):
        return self._ns_old(ag, C)


# ── 消融与扩展 ──────────────────────────────────────────────────────────────
class NoTau(SQLPredictor):
    """去掉 tau 地板。成本指数修好之后, 地板可能已经多余甚至有害。"""
    def _parts(self, cost, steps, ag, C):
        c = np.asarray(cost, float)
        sql = float(np.sum(c/1000.0 * self._amp_vec(c, steps, ag, C))) * steps / len(c)
        return sql, self._ns2(ag, C) * steps / STEPS_REF


class SepAC(SQLPredictor):
    """把 log(cost)·log(ρ) 拆成 log(cost)·log(ag) 与 log(cost)·log(C) 两项 —— 检验
    '只通过 ρ 起作用' 这个约束是不是太紧。"""
    @staticmethod
    def _xa(oc, B, ag, C, use_rho=True):
        lo, lb = math.log(max(oc,1e-12)), math.log(max(B,1e-12))
        la, lc = math.log(float(ag)), math.log(float(C))
        return np.array([lo, lo*la, lo*lc, lb, la, lc, 1.0])
    def _amp_vec(self, cost, steps, ag, C):
        c = np.asarray(cost, float)
        if getattr(self, "ba2_", None) is None: return super()._amp_vec(cost, steps, ag, C)
        B = float(np.mean(c))*steps/1000.0
        lo = np.log(np.maximum(c,1e-12)); la, lc = math.log(float(ag)), math.log(float(C))
        b = self.ba2_
        return np.exp(b[0]*lo + b[1]*lo*la + b[2]*lo*lc + b[3]*math.log(max(B,1e-12))
                      + b[4]*la + b[5]*lc + b[6])


class Quad(SQLPredictor):
    """再加 log(cost)² —— 检验放大对成本是不是纯幂律。"""
    @staticmethod
    def _xa(oc, B, ag, C, use_rho=True):
        lo, lb = math.log(max(oc,1e-12)), math.log(max(B,1e-12))
        la, lc = math.log(float(ag)), math.log(float(C))
        return np.array([lo, lo*(la-lc), lo*lo, lb, la, lc, 1.0])
    def _amp_vec(self, cost, steps, ag, C):
        c = np.asarray(cost, float)
        if getattr(self, "ba2_", None) is None: return super()._amp_vec(cost, steps, ag, C)
        B = float(np.mean(c))*steps/1000.0
        lo = np.log(np.maximum(c,1e-12)); la, lc = math.log(float(ag)), math.log(float(C))
        b = self.ba2_
        return np.exp(b[0]*lo + b[1]*lo*(la-lc) + b[2]*lo*lo + b[3]*math.log(max(B,1e-12))
                      + b[4]*la + b[5]*lc + b[6])


class NSRho(SQLPredictor):
    """非 sql 项的核指数改成随 ρ 变(而不是随 ag 变) —— 与 sql 侧同一机理。"""
    @staticmethod
    def _fn(ag, C):
        la, lc = math.log(float(ag)), math.log(float(C))
        return np.array([la, lc, lc*(la-lc), la*la, lc*lc, 1.0])
    def fit(self, workloads, anchor_rows=None):
        super().fit(workloads, anchor_rows=anchor_rows)
        rows = [(p["ag"], p["C"], p["ns"]*STEPS_REF/w["steps"], wi)
                for wi, w in enumerate(workloads) for p in w["points"] if p.get("ns",0) > 0]
        X = np.array([self._fn(r[0], r[1]) for r in rows])
        y = np.log(np.array([r[2] for r in rows], float))
        gid = [r[3] for r in rows]; cnt = collections.Counter(gid)
        wt = np.sqrt(1.0/np.array([cnt[g] for g in gid], float))
        ex = (anchor_rows or ([],[]))[1]
        if ex:
            X = np.vstack([X] + [self._fn(e[0], e[1]) for e in ex])
            y = np.concatenate([y, np.log(np.array([e[2] for e in ex], float))])
            wt = np.concatenate([wt, np.ones(len(ex))])
        self.bn3_ = np.linalg.lstsq(X*wt[:,None], y*wt, rcond=None)[0]
        return self
    def _ns2(self, ag, C):
        if getattr(self, "bn3_", None) is None: return super()._ns2(ag, C)
        return float(np.exp(self._fn(ag, C) @ self.bn3_))


class Rho2(SQLPredictor):
    """放大项再加 log(ρ)² 主效应。实测残差按 ρ 分是 U 形(ρ<1 预测/实测 0.66x,
    ρ 1-4 是 1.02x, ρ>15 是 0.74x), 对 log(ρ) 线性的项拟不出 U。"""
    @staticmethod
    def _xa(oc, B, ag, C, use_rho=True):
        lo, lb = math.log(max(oc,1e-12)), math.log(max(B,1e-12))
        la, lc = math.log(float(ag)), math.log(float(C)); lr = la - lc
        return np.array([lo, lo*lr, lr*lr, lb, la, lc, 1.0])
    def _amp_vec(self, cost, steps, ag, C):
        c = np.asarray(cost, float)
        if getattr(self, "ba2_", None) is None: return super()._amp_vec(cost, steps, ag, C)
        B = float(np.mean(c))*steps/1000.0
        lo = np.log(np.maximum(c,1e-12)); la, lc = math.log(float(ag)), math.log(float(C)); lr = la-lc
        b = self.ba2_
        return np.exp(b[0]*lo + b[1]*lo*lr + b[2]*lr*lr + b[3]*math.log(max(B,1e-12))
                      + b[4]*la + b[5]*lc + b[6])


class Rho2x(Rho2):
    """再加 log(cost)·log(ρ)² —— 让成本指数本身也沿 ρ 走 U 形。"""
    @staticmethod
    def _xa(oc, B, ag, C, use_rho=True):
        lo, lb = math.log(max(oc,1e-12)), math.log(max(B,1e-12))
        la, lc = math.log(float(ag)), math.log(float(C)); lr = la - lc
        return np.array([lo, lo*lr, lo*lr*lr, lr*lr, lb, la, lc, 1.0])
    def _amp_vec(self, cost, steps, ag, C):
        c = np.asarray(cost, float)
        if getattr(self, "ba2_", None) is None: return super()._amp_vec(cost, steps, ag, C)
        B = float(np.mean(c))*steps/1000.0
        lo = np.log(np.maximum(c,1e-12)); la, lc = math.log(float(ag)), math.log(float(C)); lr = la-lc
        b = self.ba2_
        return np.exp(b[0]*lo + b[1]*lo*lr + b[2]*lo*lr*lr + b[3]*lr*lr
                      + b[4]*math.log(max(B,1e-12)) + b[5]*la + b[6]*lc + b[7])


class ABA(NSPredictor):
    """放大倍数的双渐近线(Lazowska 渐近界,  在 db_time 总量层面验证过):

        轻载支  A = a0                        —— 服务需求本身, 与成本、并发无关
        重载支  Q = q0 · cost^qc · ag^qa · C^qc2 —— 排队, 贵查询被排队代价更大
        amp = (A^p + Q^p)^(1/p)

    与模型3(交互项)的区别: 交互项是"成本指数随 ρ 线性移动", 软max 是"两个区制各管一段,
    拐点自己定"。 的教训是软max 在并发轴上撬得动而指数形式撬不动, 值得一试。
    """
    P = 2.0
    def fit(self, workloads, anchor_rows=None):
        super().fit(workloads, anchor_rows=anchor_rows)
        rows = []
        for wi, w in enumerate(workloads):
            for pc in QP:
                oc = float(np.percentile(w["cost"], pc))
                for p in w["points"]:
                    v = p.get(f"q{pc}")
                    if v and oc > 0: rows.append((oc, p["ag"], p["C"], v/oc, wi))
        ex = [(e[0], e[2], e[3], e[4], -1) for e in (anchor_rows or ([], []))[0]]
        rows = rows + ex
        if len(rows) < 30: self.aba_ = None; return self
        cnt = collections.Counter(r[4] for r in rows)
        wt = np.array([1.0 if r[4] < 0 else math.sqrt(1.0/cnt[r[4]]) for r in rows])
        oc = np.array([r[0] for r in rows]); ag = np.array([r[1] for r in rows], float)
        C = np.array([r[2] for r in rows], float); y = np.log(np.array([r[3] for r in rows]))
        def resid(th):
            a0, q0, qc, qa, qc2 = th
            A = math.exp(a0)
            Q = np.exp(q0 + qc*np.log(oc) + qa*np.log(ag) + qc2*np.log(C))
            return wt*(np.log((A**self.P + Q**self.P)**(1.0/self.P)) - y)
        try:
            from scipy.optimize import least_squares
            best = None
            for s in ([0.3,-1.0,0.1,0.8,-0.9],[0.0,0.0,0.0,1.0,-1.0],[0.7,-2.0,0.2,0.9,-0.8]):
                r = least_squares(resid, s, max_nfev=4000)
                if best is None or r.cost < best.cost: best = r
            self.aba_ = best.x
        except Exception:
            self.aba_ = None
        return self
    def _amp_vec(self, cost, steps, ag, C):
        c = np.asarray(cost, float)
        if getattr(self, "aba_", None) is None:
            amp = math.exp(self.ba_[-1])*np.ones_like(c)
            B = float(np.mean(c))*steps/1000.0
            vals = {"cost": c, "B": B, "ag": float(ag), "C": float(C)}
            for i, k in enumerate(self.FA):
                amp = amp*np.maximum(np.asarray(vals[k], float), 1e-9)**self.ba_[i]
            return amp
        a0, q0, qc, qa, qc2 = self.aba_
        A = math.exp(a0)
        Q = np.exp(q0 + qc*np.log(np.maximum(c,1e-12)) + qa*math.log(float(ag)) + qc2*math.log(float(C)))
        return (A**self.P + Q**self.P)**(1.0/self.P)
    def _parts(self, cost, steps, ag, C):
        c = np.asarray(cost, float)
        per = c/1000.0*self._amp_vec(c, steps, ag, C)
        if getattr(self, "tau_", None) is not None:
            per = np.maximum(per, self.tau_[0]*float(ag)**self.tau_[1]/1000.0)
        return float(np.sum(per))*steps/len(c), self._ns2(ag, C)*steps/STEPS_REF
    def _raw(self, cost, steps, ag, C):
        c = np.asarray(cost, float)
        return float(np.sum(c/1000.0*self._amp_vec(c, steps, ag, C)))*steps/len(c)\
               + self._ns2(ag, C)*steps/STEPS_REF


class SQLagC(SQLPredictor):
    """模型5: 在模型3 基础上让 SQL 放大项的【核指数随并发变】—— 加 log(ag)·log(C)。

    实测 sql 分量对有效核的弹性(每库每并发档 log-log, 库间中位):
        a8 -0.485   a16 -0.817   a64 -1.103   a224 -1.176
    模型3 只有一个全局 log(C) 系数 -0.849。残差在 (并发,核) 网格上的样子:
        a8 :  2核 1.34 ... 16核 0.59     (模型随核降太快; 轻载 R≈D, 核多了也快不了)
        a224: 2核 0.70 ... 16核 1.00     (模型降太慢;   重载 R≈N·Dmax/C, 弹性趋 -1)
    即 Lazowska 渐近界的两个区。非 SQL 项早已做了同样的修正(ns_model2), SQL 项漏了。
    """
    @staticmethod
    def _xa(oc, B, ag, C, use_rho=True):
        lo, lb = math.log(max(oc,1e-12)), math.log(max(B,1e-12))
        la, lc = math.log(float(ag)), math.log(float(C))
        return np.array([lo, lo*(la-lc), la*lc, lb, la, lc, 1.0])
    def _amp_vec(self, cost, steps, ag, C):
        c = np.asarray(cost, float)
        if getattr(self, "ba2_", None) is None: return super()._amp_vec(cost, steps, ag, C)
        B = float(np.mean(c))*steps/1000.0
        lo = np.log(np.maximum(c,1e-12)); la, lc = math.log(float(ag)), math.log(float(C))
        b = self.ba2_
        return np.exp(b[0]*lo + b[1]*lo*(la-lc) + b[2]*la*lc + b[3]*math.log(max(B,1e-12))
                      + b[4]*la + b[5]*lc + b[6])


class SQLagCc(SQLPredictor):
    """模型5b: 核指数的并发依赖【按查询成本加权】—— 加 log(cost)·log(ag)·log(C)。
    机理: 重载下贵查询 CPU 受限, 时延按 1/C 走; 便宜查询被固定开销/闩锁约束, 不跟 C 走。
    模型5 的 log(ag)·log(C) 与成本无关, 把训练集查询组合的弹性(-1.18)搬到了成本分布
    完全不同的 test2(中位 0.71ms, 均值/中位 17.9x)上, 在 224并发×2核 处高估 64%。"""
    @staticmethod
    def _xa(oc, B, ag, C, use_rho=True):
        lo, lb = math.log(max(oc,1e-12)), math.log(max(B,1e-12))
        la, lc = math.log(float(ag)), math.log(float(C))
        return np.array([lo, lo*(la-lc), lo*la*lc, lb, la, lc, 1.0])
    def _amp_vec(self, cost, steps, ag, C):
        c = np.asarray(cost, float)
        if getattr(self, "ba2_", None) is None: return super()._amp_vec(cost, steps, ag, C)
        B = float(np.mean(c))*steps/1000.0
        lo = np.log(np.maximum(c,1e-12)); la, lc = math.log(float(ag)), math.log(float(C))
        b = self.ba2_
        return np.exp(b[0]*lo + b[1]*lo*(la-lc) + b[2]*lo*la*lc + b[3]*math.log(max(B,1e-12))
                      + b[4]*la + b[5]*lc + b[6])
