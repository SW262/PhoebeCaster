#!/usr/bin/env python3
# 预注册 prereg_htgain.json: 只用 11 个训练库留一选 HT_GAIN 的形式。
import json,os,glob,re,collections,importlib.util
import numpy as np
D=os.path.abspath(os.path.dirname(__file__))
def imp(n):
    s=importlib.util.spec_from_file_location(n,os.path.join(D,n+".py")); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
pp=imp("phoebe_predictor")
import badpoints
meta=[w for w in json.load(open(f"{D}/pdb_meta.json")) if w not in badpoints.EXCLUDED_LIBS]
lab=[x for x in json.load(open(f"{D}/train_label_noise.json")) if x.get("median",0)>0]
_c=collections.defaultdict(list)
for x in lab: _c[int(x["qid"])].append(x["median"])
qc={q:float(np.mean(v)) for q,v in _c.items()}
def grid(w):
    P={}
    for p in glob.glob(f"{D}/{w}_grid/s*/point.json"):
        m=re.match(r"s(\d+)_l4_t(\d)_a(\d+)_b0.2$",os.path.basename(os.path.dirname(p)))
        if not m: continue
        d=json.load(open(p)); bk=d.get("db_time_breakdown_s") or {}
        if bk.get("sql",0)*1000.0/max(d.get("sql_exec_n") or 1,1)>(d.get("sql_exec_p99_ms") or float("inf")): continue
        P[(int(m[1]),int(m[2]),int(m[3]))]=d
    return P
G={w:grid(w) for w in meta}
# 训练库上的 HT 增益
HT=collections.defaultdict(list)
for w in meta:
    for (s,th,a) in G[w]:
        if th==2 and (s,1,a) in G[w]: HT[a].append(G[w][(s,1,a)]["db_time_s"]/G[w][(s,2,a)]["db_time_s"])
TAB={a:float(np.median(v)) for a,v in HT.items()}
ALLMED=float(np.median([x for v in HT.values() for x in v]))
ks=sorted(TAB)
lx=np.log(ks); ly=[TAB[k] for k in ks]
LIN=np.polyfit(lx,ly,1)
def gain(ag,mode):
    if mode=="H0": return 1.24
    if mode=="H1": return ALLMED
    if mode=="H3": return float(np.clip(np.polyval(LIN,np.log(ag)),1.0,1.5))
    if ag<=ks[0]: return TAB[ks[0]]
    if ag>=ks[-1]: return TAB[ks[-1]]
    lo=max(k for k in ks if k<=ag); hi=min(k for k in ks if k>=ag)
    if lo==hi: return TAB[lo]
    t=(np.log(ag)-np.log(lo))/(np.log(hi)-np.log(lo))
    return float((1-t)*TAB[lo]+t*TAB[hi])
def eff(s,th,ag,mode): return s*(gain(ag,mode) if th==2 else 1.0)
def build(mode):
    """按该 mode 重建 11 个库的训练行"""
    out=[]
    for w in meta:
        ids=[q["question_id"] for q in json.load(open(f"{D}/{w}_questions.json"))]
        cost=np.array([qc[q] for q in ids if q in qc])
        pts=[]
        for (s,th,a),d in G[w].items():
            bk=d.get("db_time_breakdown_s") or {}
            pts.append(dict(C=eff(s,th,a,mode),ag=a,sql=bk.get("sql",0),
                            ns=bk.get("vec_mem",0)+bk.get("vec_value",0)+bk.get("tp_txn",0),
                            **{f"q{p_}":d.get(f"sql_exec_p{p_}_ms") for p_ in pp.QP}))
        out.append(dict(cost=cost,steps=json.load(open(f"{D}/pdb_meta.json"))[w]["steps"],points=pts))
    return out
print(f"训练库 HT 增益: "+"  ".join(f"ag{a}={TAB[a]:.3f}" for a in ks)+f"   总体中位 {ALLMED:.3f}")
print(f"H3 线性拟合: gain = {LIN[0]:+.4f}·log(ag) {LIN[1]:+.4f}\n")
print(f"{'形式':<4s} {'n':>5s} {'留一中位':>8s} {'p90':>7s}")
print("-"*32)
for mode in ("H0","H1","H2","H3"):
    WLS=build(mode); e=[]
    for i,w in enumerate(meta):
        P=G[w]; A=[(16,2,224),(8,2,8)]
        if any(k not in P for k in A): continue
        cost=WLS[i]["cost"]
        def sp(k):
            bk=P[k].get("db_time_breakdown_s") or {}
            return bk.get("sql",0), bk.get("vec_mem",0)+bk.get("vec_value",0)+bk.get("tp_txn",0)
        ar=pp.PhoebePredictor.anchor_as_rows(cost,pp.STEPS_REF,[(eff(k[0],k[1],k[2],mode),k[2],*sp(k)) for k in A])
        M=pp.PhoebePredictor().fit([X for j,X in enumerate(WLS) if j!=i],anchor_rows=ar)
        m=pp.PhoebePredictor(); m.ba_,m.bn_=M.ba_,M.bn_
        m.calibrate(cost,pp.STEPS_REF,[(eff(k[0],k[1],k[2],mode),k[2],*sp(k)) for k in A])
        for (s,th,a),d in P.items():
            if (s,th,a) in A: continue
            C=eff(s,th,a,mode)
            sq,ns=m._parts(cost,pp.STEPS_REF,a,C)
            if hasattr(m,"cal_sql_"):
                v=sq*float(np.exp(m.cal_sql_[0]*np.log(a)+m.cal_sql_[1]))+ns*float(np.exp(m.cal_ns_[0]*np.log(a)+m.cal_ns_[1]))
            else:
                v=m._raw(cost,pp.STEPS_REF,a,C)
                if hasattr(m,"cal_"): v*=float(np.exp(m.cal_[0]*np.log(a)+m.cal_[1]))
            e.append(abs(v-d["db_time_s"])/d["db_time_s"])
    e=np.array(e)
    print(f"{mode:<4s} {len(e):5d} {100*np.median(e):7.1f}% {100*np.percentile(e,90):6.1f}%")
json.dump({"table":TAB,"all_median":ALLMED,"lin":list(LIN)},open("ht_gain_fit.json","w"))
