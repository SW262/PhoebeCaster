#!/usr/bin/env python3
"""把 v2 训练集 (rc3+rc4, 11 个 pdb 库) 固化成 train_set_v2.json, 供 phoebe_cli._load 读。

每格点取 n 次重复的均值(窗口均值估计量); 坏点判据走 badpoints.py; 有效性过滤同老流程 。
重跑这一条命令即可重建, 内容由 rc3/ rc4/ 下的 point.json 唯一决定。
"""
import json, os, datetime
import numpy as np
import eval_parts
from paths import DATA as D
W = eval_parts.build()
out = {"generated": datetime.datetime.now().strftime("%F %T"),
       "source": "rc3+rc4 (b0, x5), 2026-10-10~14", "cmd": "python3 build_train_v2.py",
       "workloads": [dict(lib=l, cost=list(map(float, w["cost"])), steps=int(w["steps"]),
                          points=w["points"]) for l, w in sorted(W.items())]}
json.dump(out, open(os.path.join(D, "train_set_v2.json"), "w"))
print(f"train_set_v2.json: {len(out['workloads'])} 库  "
      f"{sum(len(w['points']) for w in out['workloads'])} 格点")
