#!/usr/bin/env python3
"""把 dev 的 11 个库拆成 11 个单库负载 —— 让 schema 变量在拟合集里有范围。

为什么必须做这个( 的根因):
  两次"解释 schema 常数"的尝试(页/题 r=-0.95, BAL r=-0.98)都因拟合集范围太窄而外推爆炸。
  wb1/wb2/wb3/W3 只是同一个库(bird_bench)上不同的题子集, 所以 schema 层面的变量
  在拟合集里根本不变(页/题只跨 1.35 倍)。而 test 在另一个库, 一步就跨出去了。

单库负载的范围: B 跨 64 倍, 页/题跨 389 倍 —— 这才够学出 schema 效应。
且全部复用现有轨迹, 不需要任何 LLM 调用。

每个负载的步数统一到 ~24500(passes = 24500/题数), 与其他负载可比。
"""
import json, os, collections, numpy as np
BIRD = os.environ.get("BIRD_DIR", "bird/dev_20240627")
tr = json.load(open("grid_full/trace_snapshot.json"))
z = np.load("question_vecs.npz"); vmap = {int(q): e for q, e in zip(z["qids"], z["embs"])}
allq = {q["question_id"]: q for q in json.load(open(f"{BIRD}/dev.json"))}
bydb = collections.defaultdict(list)
for t in tr:
    q = int(t["qid"])
    if q in vmap and q in allq:
        bydb[t["db"]].append(t)
meta = {}
print(f"{'库':24s} {'题数':>5s} {'遍数':>6s} {'步数':>7s}")
for db, ts in sorted(bydb.items()):
    ids = [int(t["qid"]) for t in ts]
    name = f"pdb_{db}"
    json.dump([allq[q] for q in ids], open(f"{name}_questions.json", "w"))
    np.savez(f"{name}_vecs.npz", qids=np.array(ids), embs=np.array([vmap[q] for q in ids]))
    json.dump(ts, open(f"{name}_trace.json", "w"))
    passes = max(1, round(24500 / len(ids)))
    meta[name] = dict(db=db, nq=len(ids), passes=passes, steps=passes * len(ids))
    print(f"{db:24s} {len(ids):5d} {passes:6d} {passes*len(ids):7d}")
json.dump(meta, open("pdb_meta.json", "w"), indent=1)
print(f"\n建好 {len(meta)} 个单库负载")
