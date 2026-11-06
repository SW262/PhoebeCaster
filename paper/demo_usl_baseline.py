#!/usr/bin/env python3
"""Baseline: the USL without its coherency term (Amdahl's law), fitted to the two anchor runs alone; same 224 targets and grading as demo_failmap.py.
   python3 demo_usl_baseline.py none  -> demo_usl_baseline_none.json(论文用: 纯 USL, 只用物理核); h / 2x 为对照

USL 吞吐 X(n) = λ n / (1 + σ(n−1))(κ=0, 两点只能定两个参数), 完成时间 T = W/X。
n = min(N, C): 并发不足时受并发限制, 否则受核数限制(教科书用法, 处理器数为 n)。
C 用与 PhoebeCaster 相同的有效核数 effective_cores(含 HT 增益 h(N)) —— 偏向基线, 让它拿到同样的 HT 信息。
两个锚点(16核/HT开/224, 8核/HT开/8) 实测墙钟恰好定出 T·n = u + v(n−1) 的 u, v(σ = v/u)。
判定逐行同 demo_failmap.py(closest ±15%)和 rev1_numbers 的截止时间规则(fewest predicted ≤ T)。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))
from paths import DATA, RESULTS

import os, sys, json, statistics as st
import score_rc, eval_testv2
from phoebe_predictor import effective_cores
D = DATA
HT = sys.argv[1] if len(sys.argv) > 1 else "h"   # h: 同模型的 h(N); none: 只用物理核; 2x: HT 开算 2 倍
TOL = 0.15; RUNGS = (2, 4, 6, 8, 11, 13, 16)
REF = {r_["workload"] + r_["ht"] + str(r_["agents"]) + str(r_["k"]): r_ for r_ in
       json.load(open(f"{RESULTS}/demo_failmap.json"))["targets"]}

out, fwd, par, FW = [], [], {}, []
for wl, nm in (("test", "bal_test"), ("test2", "test2")):
    _, _, P = eval_testv2.build(wl)
    n_ = lambda s, th, ag: min(ag, effective_cores(s, th, ag) if HT == "h" else s * (2 if (HT == "2x" and th == 2) else 1))
    (n1, t1), (n2, t2) = [(n_(*k), P[k]["wall"]) for k in ((16, 2, 224), (8, 2, 8))]
    v = (t1 * n1 - t2 * n2) / ((n1 - 1) - (n2 - 1)); u = t1 * n1 - v * (n1 - 1)
    par[nm] = dict(u=u, v=v, sigma=v / u)
    pred = lambda s, th, ag: (u + v * (n_(s, th, ag) - 1)) / n_(s, th, ag)
    for k_, p in P.items():
        if k_ not in ((16, 2, 224), (8, 2, 8)) and k_[0] in RUNGS: fwd.append(abs(pred(*k_) / p["wall"] - 1)); FW.append(dict(workload=nm, srv=k_[0], ht="on" if k_[1] == 2 else "off", agents=k_[2], err=abs(pred(*k_) / p["wall"] - 1)))
    base = P[(16, 2, 224)]["wall"]
    for th in (2, 1):
        for ag in eval_testv2.AGS:
            L = {s: P[(s, th, ag)]["wall"] for s in RUNGS if (s, th, ag) in P}
            if len(L) < 4: continue
            for k in score_rc.KS:
                T = base * k
                truly = any(abs(x / T - 1) <= TOL for x in L.values())
                d, srv = min((abs(pred(s, th, ag) / T - 1), s) for s in L)
                mf = d <= TOL; hit = abs(L[srv] / T - 1) <= TOL
                ok = (truly and mf and hit) or (not truly and not mf)
                # 截止时间规则: 预测 ≤ T 的最少核数; 对 = 该核数实测 ≤ T, 或无核数预测 ≤ T 且确实无核数实测 ≤ T
                cand = [s for s in sorted(L) if pred(s, th, ag) <= T]
                any_meet = any(x <= T for x in L.values())
                dl_ok = (L[cand[0]] <= T) if cand else (not any_meet)
                out.append(dict(workload=nm, ht="on" if th == 2 else "off", agents=ag, k=k, feasible=truly,
                                ok=ok, dl_ok=dl_ok, best=min(L, key=lambda s_: abs(L[s_] / T - 1)), ok_ours=REF[nm + ("on" if th == 2 else "off") + str(ag) + str(k)]["ok"]))
pct = lambda xs: 100 * sum(xs) / len(xs)
R = dict(cmd="python3 demo_usl_baseline.py", params=par, n=len(out),
         fwd_median=100 * st.median(fwd), fwd_within15=pct([e <= TOL for e in fwd]), n_fwd=len(fwd),
         closest=pct([r["ok"] for r in out]), deadline=pct([r["dl_ok"] for r in out]),
         ours_closest=pct([r["ok_ours"] for r in out]),
         by=dict((f"{w}/{h}", pct([r["ok"] for r in out if r["workload"] == w and r["ht"] == h]))
                 for w in ("bal_test", "test2") for h in ("on", "off")),
         feasible=pct([r["ok"] for r in out if r["feasible"]]), infeasible=pct([r["ok"] for r in out if not r["feasible"]]))
R["only_ours"] = sum(r["ok_ours"] and not r["ok"] for r in out); R["only_usl"] = sum(r["ok"] and not r["ok_ours"] for r in out)
assert len(out) == 224 and len(fwd) == 192 and abs(R["ours_closest"] - 84.375) < 1e-6
R["ht_mode"] = HT
R["targets"] = out; R["fwd_points"] = FW
json.dump(R, open(f"{RESULTS}/demo_usl_baseline_{HT}.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in R.items() if k not in ("targets", "fwd_points")}, indent=1, ensure_ascii=False))
