#!/usr/bin/env python3
"""采集与数据集状态采集器 —— 只读, 不碰正在跑的采集。

所有数字都从磁盘上已有的产物读:
  rc*/progress.tsv  rc*/alerts.tsv  rc3/clog_watchdog.log  rc3/envtrace.tsv
坏点判据统一走 badpoints.py (steps 必须等于该库众数), 与所有分析脚本同一处定义。
CI 覆盖走 ci_status.py --json, 同一条公式, 不另起一套。
"""
import os, re, csv, sys, json, time, subprocess, datetime, collections

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # bird-agent/
sys.path.insert(0, BASE)
_CACHE = {}


def _cached(key, ttl, fn):
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    val = fn()
    _CACHE[key] = (time.time(), val)
    return val


def _alive(pidfile):
    try:
        pid = int(open(os.path.join(BASE, pidfile)).read().strip())
        os.kill(pid, 0)
        return pid
    except Exception:
        return None


def _nominal():
    import badpoints
    return _cached("nom", 300, badpoints.nominal_steps)


def run_progress(run, total):
    """一轮采集的进度。好点判据 = steps 等于该库标称值。"""
    p = os.path.join(BASE, run, "progress.tsv")
    if not os.path.exists(p):
        return None
    nom = _nominal()
    rows = list(csv.DictReader(open(p), delimiter="\t"))
    if not rows:
        return None
    good = [r for r in rows if int(r["steps"] or 0) == nom.get(r["lib"], -1)]
    T = lambda s: datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    ts = [T(r["ts"]) for r in rows]
    idx = max(int(r["idx"]) for r in rows)
    hours = (max(ts) - min(ts)).total_seconds() / 3600 or 1e-9
    rate = idx / hours
    rem = max(0, total - idx)
    return dict(run=run, idx=idx, total=total, good=len(good), bad=len(rows) - len(good),
                rate=round(rate, 1), hours=round(hours, 1), remaining=rem,
                eta_h=round(rem / rate, 1) if rate else None,
                eta=(datetime.datetime.now() + datetime.timedelta(hours=rem / rate)
                     ).strftime("%m-%d %H:%M") if rate and rem else None,
                last=max(ts).strftime("%F %T"))


def testv2_progress():
    """新定义测试集。好点判据: steps == passes x questions (expect 列)。"""
    p = os.path.join(BASE, "testv2", "progress.tsv")
    if not os.path.exists(p): return None
    rows = list(csv.DictReader(open(p), delimiter="\t"))
    if not rows: return None
    T = lambda s: datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    total = order_total("testv2_order.tsv")
    good = [r for r in rows if r["steps"] == r["expect"] and r["steps"] != "0"]
    ts = [T(r["ts"]) for r in rows]
    idx = len({r["idx"] for r in rows})
    first = min(ts)
    hours = max((datetime.datetime.now() - first).total_seconds() / 3600, 1e-9)
    rate = idx / hours; rem = max(0, total - idx)
    return dict(run="testv2", idx=idx, total=total, good=len(good), bad=len(rows) - len(good),
                rate=round(rate, 1), hours=round(hours, 1), remaining=rem,
                eta_h=round(rem / rate, 1) if rate else None,
                eta=(datetime.datetime.now() + datetime.timedelta(hours=rem / rate)).strftime("%m-%d %H:%M")
                    if rate and rem else None,
                last=max(ts).strftime("%F %T"))


def order_total(f):
    p = os.path.join(BASE, f)
    if not os.path.exists(p):
        return 0
    return sum(1 for ln in open(p)) - 1


def _watchdogs():
    """root 常驻的看门狗按 CLOG 段各一个实例, 没有统一 pidfile —— 数进程。"""
    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
        return sum(1 for l in out.splitlines() if l.startswith("python3 clog_watchdog2.py"))
    except Exception:
        return None


def guards():
    log = os.path.join(BASE, "rc3", "clog_watchdog.log")
    trunc, last_trunc = 0, None
    if os.path.exists(log):
        for ln in open(log):
            if "又被截断" in ln:
                trunc += 1; last_trunc = ln[:19]
    partial = 0
    for run in ("rc3", "rc4", "rc5"):
        a = os.path.join(BASE, run, "alerts.tsv")
        if os.path.exists(a):
            partial += sum(1 for ln in open(a) if "\tPARTIAL\t" in ln)
    # 数隔离目录里的 point.json, 不数 LIST.tsv 的行 —— 后者带表头, 会多算一个
    import glob as _g
    quarantined = len(_g.glob(os.path.join(BASE, "rc3_bad", "", "point.json"), recursive=True))
    return dict(
        clog_truncations=trunc, clog_last=last_trunc, partial_caught=partial,
        quarantined=quarantined,
        watchdogs=_watchdogs(),
        procs=[dict(name=n, pidfile=f, pid=_alive(f))
               for n, f in (("Test-set collection (testv2)", "testv2.pid"),
                            ("Training collection rc4", "rc4.pid"),
                            ("Training collection rc5", "rc5.pid"),
                            # v2 是 root 常驻、0.2s 轮询的版本; 现在按段各一个实例
                            ("CLOG watchdog", "clog_watchdog2.pid"),
                            ("rc5 chain", "rc5_chain.pid"))])


def env():
    p = os.path.join(BASE, "rc3", "envtrace.tsv")
    if not os.path.exists(p):
        return {}
    last = None
    with open(p) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for ln in f:
            q = ln.rstrip("\n").split("\t")
            if len(q) == len(hdr): last = dict(zip(hdr, q))
    if not last:
        return {}
    num = lambda k: (float(last[k]) if last.get(k) not in (None, "") else None)
    return dict(ts=last.get("ts"), comp_c=num("comp_c"), ts2_c=num("ts2_c"),
                warn_min=num("warn_min"), avail_g=num("avail_g"),
                cpu_mine=num("cpu_mine"), cpu_other=num("cpu_other"))


_EN = [("本次=", "this run="), ("偏离基准", "vs baseline"), ("累计(最近4/最初4)", "cumulative(last4/first4)"),
       ("!! 硬停A(单次偏离>40%)", "!! hard-stop A (single deviation >40%)"),
       ("!! 硬停B(累计漂移>20%)", "!! hard-stop B (cumulative drift >20%)"),
       ("滚动4/4", "rolling4/4"), ("滚动8/8", "rolling8/8"), ("滚动CV", "rolling CV"), ("预测", "predicted "),
       (" 警告: 实测CV超预测2倍, 重复次数可能不够", " warning: measured CV >2x predicted, repeats may be insufficient"),
       ("gate: 无 canary 记录, 放行", "gate: no canary records, pass")]


def _en(line):
    for a, b in _EN: line = line.replace(a, b)
    return line


def canary():
    """复用 rc3_gate.py —— 网页上的漂移数和采集里闸用的是同一行代码。"""
    out = {}
    for run in ("rc3", "rc4", "rc5"):
        p = os.path.join(BASE, run, "progress.tsv")
        if not os.path.exists(p): continue
        e = dict(os.environ, GATE_PROG=f"{run}/progress.tsv")
        try:
            r = subprocess.run([sys.executable, "rc3_gate.py"], cwd=BASE, env=e,
                               capture_output=True, text=True, timeout=60)
            out[run] = dict(line=_en(r.stdout.strip()), stop=(r.returncode != 0))
        except Exception as ex:
            out[run] = dict(line=f"(取不到: {ex})", stop=False)
    return out


def ci():
    def go():
        r = subprocess.run([sys.executable, "ci_status.py", "--json"], cwd=BASE,
                           capture_output=True, text=True, timeout=180)
        return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else {}
    try:
        return _cached("ci", 120, go)
    except Exception:
        return {}


def snapshot():
    runs = [r for r in (run_progress("rc3", order_total("rc3_order.tsv")),
                        run_progress("rc4", order_total("rc4_order.tsv")),
                        run_progress("rc5", order_total("rc5_order.tsv")),
                        testv2_progress()) if r]
    return dict(now=time.strftime("%F %T"), runs=runs, guards=guards(),
                env=env(), canary=canary(), ci=ci())


if __name__ == "__main__":
    print(json.dumps(snapshot(), ensure_ascii=False, indent=2))
