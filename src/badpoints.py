#!/usr/bin/env python3
"""
坏点判据 —— 一处定义, 所有分析脚本共用。

背景: 2026-10-12 的 CLOG 截断事故烧掉 50 个点。当时装的闸是 steps==0,
只抓得住"一步没跑"的点。但事故发生的那一瞬间正在跑的那个点不是 0,
是 95355 / 108300 = 88% —— 跑到一半被打断, rc=0, 一路过关进了好点集合。

真正的判据: steps 对同一个库是常数 (passes × nq 定死, 与资源配置无关)。
全部 967 个点里只有 1 个偏离, 就是事故那一刻的点。所以:

  坏点 = steps 不等于该库的众数

sql_errors 不能当判据: 它是 BIRD gold SQL 本身跑不过的条数, 每库一个基线
(card_games 342, thrombosis 7125 ...), 在高并发库上还会随锁竞争小幅浮动
(european_football_2 1300~1490)。拿它当闸会误杀 22 个好点。
"""
import json, glob, collections, os, re

from paths import DATA
RC3 = os.path.join(DATA, "rc3")

# 整个剔除的训练负载(2026-10-28)。pdb_european_football_2: qid 1127 的第 1 轮 LLM SQL 已被
# sql_screen.json 判为慢查询排除(结果也是错的), 但该题其余轮次都是截断, 轨迹为空, workload.py
# 回放时退回缓存里的第 1 轮, 绕过了筛查 —— 全部运行都混进这条约 21 s 的查询(TODO #20)。
# nominal_steps 不给它投票, 所有按 NOM 过滤的加载(eval_parts / paper_numbers_v2 / build_train_v2)
# 随之跳过; 老网格(fit_rmodel / ht_select)与离线标签(rev1_numbers 探测)各自查这张表。
EXCLUDED_LIBS = {"pdb_european_football_2": "qid 1127 screened-out SQL leaked in via cache fallback"}


def nominal_steps(root=RC3):
    """{lib: 众数 steps}。只用 steps>0 的点投票。"""
    v = collections.defaultdict(collections.Counter)
    for p in glob.glob(os.path.join(root, "pdb_*", "*", "point.json")):
        try:
            s = json.load(open(p)).get("steps", 0)
        except Exception:
            continue
        if s > 0 and p.split(os.sep)[-3] not in EXCLUDED_LIBS:
            v[p.split(os.sep)[-3]][s] += 1
    return {k: c.most_common(1)[0][0] for k, c in v.items()}


def bad_point_files(root=RC3, nom=None):
    """返回 [(path, lib, tag, steps, nominal), ...] —— 需要作废的 point.json。"""
    nom = nom or nominal_steps(root)
    out = []
    for p in sorted(glob.glob(os.path.join(root, "pdb_*", "*", "point.json"))):
        lib, tag = p.split(os.sep)[-3], p.split(os.sep)[-2]
        try:
            s = json.load(open(p)).get("steps", 0)
        except Exception:
            out.append((p, lib, tag, None, nom.get(lib)))
            continue
        n = nom.get(lib)
        if n is None or s != n:
            out.append((p, lib, tag, s, n))
    return out


def bad_idx(prog=os.path.join(RC3, "progress.tsv"), nom=None):
    """progress.tsv 里 steps 偏离众数的 idx 集合 —— 给按 run.log 解析的脚本用。

    run.log 的点行只打了 db_time/wall/sql/tp_txn, 没有 steps, 所以那些脚本
    没法自己判。用 idx 把 progress.tsv 的判定接过去。
    """
    nom = nom or nominal_steps()
    bad, seen = set(), {}
    if not os.path.exists(prog):
        return bad
    with open(prog) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ii, li, si = hdr.index("idx"), hdr.index("lib"), hdr.index("steps")
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) <= si:
                continue
            try:
                idx, steps = int(p[ii]), int(p[si])
            except ValueError:
                continue
            n = nom.get(p[li])
            if n is None or steps != n:
                bad.add(idx)
    return bad


if __name__ == "__main__":
    nom = nominal_steps()
    print(f"{'库':<30}{'nominal steps':>14}")
    for k in sorted(nom):
        print(f"{k:<30}{nom[k]:>14}")
    b = bad_point_files(nom=nom)
    print(f"\n坏 point.json: {len(b)}")
    for p, lib, tag, s, n in b:
        print(f"  {p}\n     steps={s} 应为 {n}  ({100*s/n:.1f}%)" if s and n else f"  {p}")
    print(f"\nprogress.tsv 坏 idx: {len(bad_idx(nom=nom))} 个")
