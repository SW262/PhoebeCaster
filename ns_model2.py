#!/usr/bin/env python3
"""非 SQL 项的改进模型 —— 核指数随并发变。

为什么改: rc3/rc4 下非 sql 占 db_time 中位 66%, 而现模型只用 `ag^a · C^b`,
强行给所有并发档一个全局核指数。308 个格点实测, 各档自己的核指数是:

    a8  -0.267    a16 -0.362    a64 -0.421    a224 -0.655

差 2.5 倍。而顾问选的就是核数档 —— 核指数错了就直接选错档。

改法: 在 log 空间加交互与二次项, 即
    log(ns) = c + a1·log(ag) + a2·log(C) + a3·log(ag)log(C) + a4·log(ag)² + a5·log(C)²
平滑、对任意并发可外推(不是档位哑变量), 只用 (ag, C) —— 预测时都已知, 不需要新测量。

格点均值上的拟合优度: R² 0.9035 -> 0.9261, 残差 sd 31.1% -> 27.2%。
SQL 项、锚点机制、calibrate 一律不动, 只换非 sql 这一项。
"""
import collections, math
import numpy as np
from phoebe_predictor import PhoebePredictor, STEPS_REF


class NSPredictor(PhoebePredictor):
    @staticmethod
    def _f(ag, C):
        la, lc = math.log(float(ag)), math.log(float(C))
        return np.array([la, lc, la * lc, la * la, lc * lc, 1.0])

    def fit(self, workloads, anchor_rows=None):
        super().fit(workloads, anchor_rows=anchor_rows)      # SQL 项与原版逐字相同
        rows = []
        for wi, w in enumerate(workloads):
            for p in w["points"]:
                if p.get("ns", 0) > 0:
                    rows.append((p["ag"], p["C"], p["ns"] * STEPS_REF / w["steps"], wi))
        X = np.array([self._f(r[0], r[1]) for r in rows])
        y = np.log(np.array([r[2] for r in rows], float))
        gid = [r[3] for r in rows]
        cnt = collections.Counter(gid)
        wt = np.sqrt(1.0 / np.array([cnt[x] for x in gid], float))   # 每个负载等权, 同原版
        ex = (anchor_rows or ([], []))[1]
        if ex:                                                        # 锚点行权重恒 1.0, 同原版
            X = np.vstack([X] + [self._f(e[0], e[1]) for e in ex])
            y = np.concatenate([y, np.log(np.array([e[2] for e in ex], float))])
            wt = np.concatenate([wt, np.ones(len(ex))])
        self.bn2_ = np.linalg.lstsq(X * wt[:, None], y * wt, rcond=None)[0]
        return self

    def _ns2(self, ag, C):
        return float(np.exp(self._f(ag, C) @ self.bn2_))

    def _ns_old(self, ag, C):
        v = float(np.exp(self.bn_[-1]))
        for i, k in enumerate(self.FN):
            v *= float({"ag": ag, "C": C}[k]) ** self.bn_[i]
        return v

    def _parts(self, cost, steps, ag, C):
        sql, _ = super()._parts(cost, steps, ag, C)
        return sql, self._ns2(ag, C) * steps / STEPS_REF

    def _raw(self, cost, steps, ag, C):
        d = (self._ns2(ag, C) - self._ns_old(ag, C)) * steps / STEPS_REF
        return super()._raw(cost, steps, ag, C) + d
