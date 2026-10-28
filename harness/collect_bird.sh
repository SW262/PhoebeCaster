#!/usr/bin/env bash
# BIRD 复合负载采集: 一个配置点 = 一次测量。
#
# 每点从纯净模板整库复制(实测 20.66 秒/11GB), 而不是 DELETE 重置 ——
# 因为 DELETE 只删堆行, cognitivehnsw 索引只涨不缩: 实测同配置重复 10 次
# 因索引 66->90GB 漂移导致吞吐 1926->698 (2.76x, CV 33.4%, 与索引大小 r=-0.836)。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SERVER_CPUS=""; LOAD_CPUS=""; AGENTS=8; DURATION=180; QUESTIONS=100; STEPS=0
PASSTHRU=()
DISCARD=0
DBS=""; EF=64; WRITES=1; TOPK=10; MEM_GB=""; TAG=""; OUT="bird_points"
REPLAY="--replay-only"
PGDATA="${PGDATA:-$HOME/pg17}"                 # PostgreSQL 17 + cognitivehnsw 的数据目录
PW="${SUDO_PW:?export SUDO_PW (taskset 绑核需要 sudo)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server-cpus) SERVER_CPUS=$2; shift 2;;
    --load-cpus)   LOAD_CPUS=$2;   shift 2;;
    --agents)      AGENTS=$2;      shift 2;;
    --duration)    DURATION=$2;    shift 2;;
    --steps)       STEPS=$2;       shift 2;;   # 固定功量模式
    --questions)   QUESTIONS=$2;   shift 2;;
    --dbs)         DBS=$2;         shift 2;;
    --ef-search)   EF=$2;          shift 2;;
    --writes-per-step) WRITES=$2;  shift 2;;
    --topk)        TOPK=$2;        shift 2;;
    --mem-gb)      MEM_GB=$2;      shift 2;;
    --run-tag)     TAG=$2;         shift 2;;
    --results-dir) OUT=$2;         shift 2;;
    --live-llm)    REPLAY="";      shift  ;;
    --discard)     DISCARD=1;      shift  ;;   # 丢弃点: 跑完不写 point.json
    # 未识别的参数原样透传给 workload.py。这是第三次因为"给 workload.py 加了旋钮
    # 却忘了接采集脚本"导致整批点白跑(--steps 一次, --tp-updates/--business-write-pct
    # 一次), 改成透传从根上解决。
    *) PASSTHRU+=("$1")
       if [[ -n "${2:-}" && "${2:-}" != -* ]]; then PASSTHRU+=("$2"); shift 2; else shift; fi;;
  esac
done
[[ -z "$TAG" ]] && TAG="srv${SERVER_CPUS//,/-}_ag${AGENTS}_ef${EF}_w${WRITES}"
mkdir -p "$OUT/$TAG"

free_g=$(df -BG --output=avail "${DATA_DISK:-$PGDATA}" | tail -1 | tr -dc '0-9')
if (( free_g < 300 )); then echo "!! 磁盘仅剩 ${free_g}G (<300G), 中止"; exit 1; fi
pg_isready -p 5433 -U postgres >/dev/null || { echo "!! PG 不可用"; exit 1; }

echo "════ $TAG  $(date '+%F %T') ════"

# ── 1. 从模板重建工作库(消除索引漂移)──
# 模板库与运行库可覆盖: 泛化测试集用 bird_test 作模板。
# 每点从模板整库复制这一机制不变 —— 那是重复性 CV 0.81% 的来源。
TEMPLATE_DB="${BIRD_TEMPLATE_DB:-bird_bench}"
RUN_DB="${BIRD_RUN_DB:-bird_bench_run}"
export RUN_DB TEMPLATE_DB          # 下面补 meta 的 python 段要读 RUN_DB
P="postgresql://postgres@localhost:5433/postgres"
psql "$P" -q -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity
                 WHERE datname='$RUN_DB' AND pid<>pg_backend_pid();" >/dev/null 2>&1
psql "$P" -q -c "DROP DATABASE IF EXISTS $RUN_DB;" 2>&1 | grep -v NOTICE || true
T0=$(date +%s)
psql "$P" -v ON_ERROR_STOP=1 -q -c \
  "CREATE DATABASE $RUN_DB TEMPLATE $TEMPLATE_DB STRATEGY = FILE_COPY;" \
  || { echo "!! 建库失败"; exit 1; }
RESET_S=$(( $(date +%s) - T0 ))
echo "  库复制完成 ${RESET_S}s"

# ── 1b. 静默期: 库复制会写 7.5GB, 下一个点若在 IO 未沉降时开测, TP 的尾延迟会被污染。
# 实测证据: srv12 两点重置耗时 34s vs 47s, 而 TP p99 差 24.83%(119.2 vs 148.8ms);
# 同时 pg_stat_checkpointer 显示 num_requested=120 —— checkpoint 是被 WAL 量触发的,
# 不是定时(checkpoint_timeout=1800s), 说明写入速率撑爆了 max_wal_size。
# 做三件事: 强制 checkpoint 把脏页刷干净 -> sync 落盘 -> 等 IO 队列排空。
QUIESCE=${QUIESCE:-15}
psql "postgresql://postgres@localhost:5433/$RUN_DB" -q -c "CHECKPOINT;" 2>/dev/null
sync
Q0=$(date +%s)
while (( $(date +%s) - Q0 < QUIESCE )); do
    # /proc/diskstats 第 9 列 = 正在进行的 IO 数; 连续两次为 0 就认为沉降
    INFL=$(awk '$3=="nvme1n1"{print $12}' /proc/diskstats 2>/dev/null | head -1)
    [[ "${INFL:-1}" == "0" ]] && sleep 1 &&       INFL2=$(awk '$3=="nvme1n1"{print $12}' /proc/diskstats 2>/dev/null | head -1) &&       [[ "${INFL2:-1}" == "0" ]] && break
    sleep 1
done
QUIESCE_S=$(( $(date +%s) - Q0 ))
export QUIESCE_S
echo "  静默 ${QUIESCE_S}s (checkpoint + sync + IO 排空)"

# ── 2. 资源绑定 ──
if [[ -n "$SERVER_CPUS" ]]; then
  # postmaster.pid 属 postgres 用户不可读 -> 用 pgrep 按 datadir 找(同 collect_cognitive_v3.sh)
  N=0
  for PID in $(pgrep -f "$PGDATA" 2>/dev/null || true); do
    echo "$PW" | sudo -S taskset -acp "$SERVER_CPUS" "$PID" >/dev/null 2>&1 && N=$((N+1))
    for ch in $(pgrep -P "$PID" 2>/dev/null || true); do
      echo "$PW" | sudo -S taskset -acp "$SERVER_CPUS" "$ch" >/dev/null 2>&1 && N=$((N+1))
    done
  done
  echo "  postgres 绑核 -> $SERVER_CPUS  (${N} 个进程)"
  [[ "$N" -eq 0 ]] && { echo "  !! 绑核失败, 中止 —— 否则会采到没绑核的假数据"; exit 1; }
fi
if [[ -n "$MEM_GB" ]]; then
  MPID=$(pgrep -f "$PGDATA" 2>/dev/null | head -1)
  echo "$PW" | sudo -S bash -c "
    mkdir -p /sys/fs/cgroup/pgmem 2>/dev/null
    echo '${MEM_GB}G' > /sys/fs/cgroup/pgmem/memory.high 2>/dev/null
    echo ${MPID} > /sys/fs/cgroup/pgmem/cgroup.procs 2>/dev/null" >/dev/null 2>&1 || true
  echo "  RAM 上限 -> ${MEM_GB}G"
fi

# ── 3. 跑负载 ──
RUNNER=(python3 -u workload.py --agents "$AGENTS" --duration "$DURATION"
        --questions "$QUESTIONS" --ef-search "$EF" --writes-per-step "$WRITES"
        --topk "$TOPK" --out "$OUT/$TAG/point.json")
[[ "$STEPS" -gt 0 ]] && RUNNER+=(--steps "$STEPS")
[[ ${#PASSTHRU[@]} -gt 0 ]] && RUNNER+=("${PASSTHRU[@]}")
[[ -n "$DBS"    ]] && RUNNER+=(--dbs "$DBS")
[[ -n "$REPLAY" ]] && RUNNER+=("$REPLAY")
if [[ -n "$LOAD_CPUS" ]]; then RUNNER=(taskset -c "$LOAD_CPUS" "${RUNNER[@]}"); fi

set -a; . ./.env; set +a
export BIRD_DSN="postgresql://postgres@localhost:5433/$RUN_DB"
# ── 2b. run 前后快照 checkpointer/bgwriter/WAL ──
#   起因: tpvar 28+10 点中 blk_write_time 漂 14x(2.4s~34.1s)而写入工作量恒定
#   (steps 恒 20825)。point.json 里没有任何检查点计数, 无法区分"run 内撞上检查点"
#   与"backend 自刷"。装上仪表, 让写阻塞可归因而不是玄学。
SNAP="$(dirname "$0")/snap_stats.sql"
SNAPIO="$(dirname "$0")/snap_io.sql"
psql "postgresql://postgres@localhost:5433/postgres" -Atf "$SNAP" \
     > "$OUT/$TAG/.snap_before.json" 2>/dev/null || true
# prereg_tpvar_L.json (md5 79351fbc): 仅当 SNAP_IO=1 时额外快照 pg_stat_io + chnsw 索引大小,
# 并以 1s 采样 postgres 进程的 /proc/<pid>/io write_bytes。默认关闭, 不影响既有口径。
if [[ "${SNAP_IO:-0}" == 1 ]]; then
  psql "$BIRD_DSN" -Atf "$SNAPIO" > "$OUT/$TAG/.snapio_before.json" 2>/dev/null || true
  "$(dirname "$0")/pgio_sampler.sh" "$OUT/$TAG/.pgio.txt" & PGIO_PID=$!
fi
"${RUNNER[@]}" > "$OUT/$TAG/run.log" 2>&1
RC=$?
if [[ "${SNAP_IO:-0}" == 1 ]]; then
  [[ -n "${PGIO_PID:-}" ]] && kill "$PGIO_PID" 2>/dev/null
  psql "$BIRD_DSN" -Atf "$SNAPIO" > "$OUT/$TAG/.snapio_after.json" 2>/dev/null || true
fi
psql "postgresql://postgres@localhost:5433/postgres" -Atf "$SNAP" \
     > "$OUT/$TAG/.snap_after.json" 2>/dev/null || true

# ── 3b. 丢弃点: 只为让系统沉降, 不产出数据 ──
# 依据: 消融那批的 full 组紧跟在 value_vectors_full 建索引(7.5GB 重 IO)之后,
# 结果 70.9s / TP p99 2048ms / 80 个事务错误; 同配置隔开重跑 3 次全部正常
# (27.2/27.6/29.1s, TP p99 89.9/92.3/138.8, 0 错误)。15 秒的静默期不足以
# 消化重 DDL 的余波, 所以每批扫描先跑一个丢弃点。
if [[ "$DISCARD" == "1" ]]; then
  rm -f "$OUT/$TAG/point.json"
  echo "  (丢弃点, 不产出数据)"
  exit 0
fi

# ── 4. 补充 meta ──
python3 - "$OUT/$TAG/point.json" "$SERVER_CPUS" "$LOAD_CPUS" "$MEM_GB" "$RESET_S" "$RC" <<'PY'
import json, subprocess, sys, os
f, srv, load, mem, reset_s, rc = sys.argv[1:7]
d = json.load(open(f)) if os.path.exists(f) else {}
def cnt(s): return len([x for p in s.split(',') if p for x in (
    range(int(p.split('-')[0]), int(p.split('-')[1])+1) if '-' in p else [int(p)])]) if s else None
d.update({"server_cpus": srv or None, "server_cpu_count": cnt(srv),
          "load_cpus": load or None, "load_cpu_count": cnt(load),
          "mem_gb": int(mem) if mem else None,
          "reset_seconds": int(reset_s), "exit_code": int(rc),
          "quiesce_seconds": int(os.environ.get("QUIESCE_S") or -1),
          "reset_mode": "template_file_copy"})
D = "postgresql://postgres@localhost:5433/" + __import__("os").environ.get("RUN_DB","bird_bench_run")
# 缓存命中/磁盘读必须记: RAM 轴测出"无效应"时无法区分两种情形 ——
# ① RAM 确实不是约束, ② 热集全在 shared_buffers 命中, 压根没碰页缓存。
# 实测 60 次 chnsw 检索在 7.5GB 索引上 idx_blks_read = 0, 就是情形 ②。
q = ("SELECT (SELECT count(*) FROM agent_memories), "
     # value_vectors 已按 db_id 分区(11 个分区各自的 chnsw 索引), 原来的单一索引名
     # 'value_vectors_chnsw_idx' 已不存在 -> 那条 pg_relation_size 会报错, 而报错会让
     # 整个 meta 补充块失败, 连带丢掉 memories_after / idx_cache_hit_pct / cgroup_* 等
     # 全部字段(实测 memories_after=None 就是这么来的)。改成汇总所有分区索引。
     # ::bigint 不能省: sum 之后是 numeric, /1024/1024 得到 "5319.906250..." ,
     # 下面 int 会抛 ValueError, 被 except 吞掉 -> 整个 meta 块丢失(实测复现)。
     "(SELECT (coalesce(sum(pg_relation_size(indexrelid)),0)/1024/1024)::bigint "
     " FROM pg_stat_user_indexes WHERE indexrelname LIKE 'value_vectors%%chnsw%%'), "
     "pg_relation_size('agent_memories_chnsw_idx')/1024/1024, "
     "(SELECT n_dead_tup FROM pg_stat_user_tables WHERE relname='agent_memories'), "
     "(SELECT coalesce(sum(idx_blks_read),0) FROM pg_statio_user_indexes), "
     "(SELECT coalesce(sum(idx_blks_hit),0)  FROM pg_statio_user_indexes), "
     "(SELECT coalesce(sum(heap_blks_read),0) FROM pg_statio_user_tables), "
     "(SELECT coalesce(sum(heap_blks_hit),0)  FROM pg_statio_user_tables)")
r = subprocess.run(["psql", D, "-t", "-A", "-F", "|", "-c", q], capture_output=True, text=True)
try:
    a, b, c, e, ir, ih, hr, hh = r.stdout.strip().split("|")
    ir, ih, hr, hh = int(ir or 0), int(ih or 0), int(hr or 0), int(hh or 0)
    _i = lambda x: int(float(x or 0))     # 对 numeric 字符串也免疫
    d.update({"memories_after": _i(a), "value_index_mb_after": _i(b),
              "memory_index_mb_after": _i(c), "memories_dead_tup": _i(e or 0),
              "idx_blks_read": ir, "idx_blks_hit": ih,
              "heap_blks_read": hr, "heap_blks_hit": hh,
              # 命中率 <100% 才说明真的在读磁盘, RAM 轴才有意义
              "idx_cache_hit_pct":  round(100.0*ih/max(ih+ir,1), 3),
              "heap_cache_hit_pct": round(100.0*hh/max(hh+hr,1), 3)})
except Exception as _ex:
    # 绝不静默: 这个块丢过两次(索引名变更、numeric 除法), 每次都因为 except: pass,
    # 结果是 memories_after / idx_cache_hit_pct / heap_cache_hit_pct 全缺,
    # RAM 轴的分析直接失去。出错要吼出来并记进 point.json。
    d["meta_error"] = f"{type(_ex).__name__}: {_ex}"[:200]
    d["meta_stdout"] = (r.stdout or "")[:200]
    d["meta_stderr"] = (r.stderr or "")[:200]
    print(f"  !! meta 查询失败: {d['meta_error']}  stdout={d['meta_stdout']!r} "
          f"stderr={d['meta_stderr']!r}", file=sys.stderr)
# ── BAL (Buffer Access Latency) ───────────────────────────────────────────────
# Duggan et al. SIGMOD'11 的核心指标: 从缓冲池请求发出到块返回的平均时间,
# 把磁盘寻道/顺序读/OS 缓存命中/缓冲池命中归一成一个数。原文改了 PG 8.4.3 的
# 缓冲池管理器才测到; 现在 track_io_timing 打开后 blk_read_time/blks_read 即是。
#
# 必须是独立的一条查询 + 独立的 try: 上面那个 meta 块的注释记录了两次事故 ——
# 单个子查询报错会让整块字段全丢。新指标绝不能拖累已有字段。
try:
    qbal = ("SELECT blks_read, blks_hit, blk_read_time, blk_write_time "
            "FROM pg_stat_database WHERE datname = current_database()")
    rb = subprocess.run(["psql", D, "-t", "-A", "-F", "|", "-c", qbal],
                        capture_output=True, text=True)
    br, bh, brt, bwt = rb.stdout.strip().split("|")
    br, bh = int(br or 0), int(bh or 0)
    brt, bwt = float(brt or 0), float(bwt or 0)
    d.update({"db_blks_read": br, "db_blks_hit": bh,
              "blk_read_time_ms": brt, "blk_write_time_ms": bwt,
              # BAL: 每次物理读块的平均等待(毫秒)。br=0 时无意义, 置 None 而非 0
              "bal_ms": round(brt / br, 5) if br > 0 else None,
              "blk_write_time_per_op_ms": round(bwt / max(br, 1), 5)})
except Exception as _bx:
    d["bal_error"] = f"{type(_bx).__name__}: {_bx}"[:200]
    print(f"  !! BAL 查询失败(不影响其他字段): {d['bal_error']}", file=sys.stderr)

# run 期间的 checkpointer / bgwriter / WAL 增量 —— 写阻塞的归因
try:
    _dir = os.path.dirname(f)
    _b = json.load(open(os.path.join(_dir, ".snap_before.json")))
    _a = json.load(open(os.path.join(_dir, ".snap_after.json")))
    _d = {f"d_{k}": (_a[k] - _b[k]) for k in _b if k != "lsn" and isinstance(_b[k], (int, float))}
    # WAL 推进量(字节): 直接用 lsn 差更可靠
    def _lsn(x):
        h, l = x.split("/"); return (int(h, 16) << 32) + int(l, 16)
    _d["d_wal_lsn_mb"] = round((_lsn(_a["lsn"]) - _lsn(_b["lsn"])) / 2**20, 1)
    # run 内是否发生检查点, 以及是定时还是 WAL 触发
    _d["ckpt_in_run"] = int(_d.get("d_ck_timed", 0) + _d.get("d_ck_req", 0))
    d.update(_d)
except Exception as _cx:
    d["ckpt_delta_error"] = f"{type(_cx).__name__}: {_cx}"[:200]

# cgroup 的实际用量与限制事件 —— 判定 RAM 上限是否真的触发了回收
try:
    cg = "/sys/fs/cgroup/pgmem"
    d["cgroup_mem_current_mb"] = int(open(cg+"/memory.current").read()) // 2**20
    for line in open(cg+"/memory.events"):
        k, v = line.split()
        if k == "high": d["cgroup_high_events"] = int(v)
    for line in open(cg+"/memory.stat"):
        k, v = line.split()
        if k == "file": d["cgroup_pagecache_mb"] = int(v) // 2**20
except Exception: pass
json.dump(d, open(f, "w"), indent=1)
cs = d.get("completion_seconds")
head = ("完成时间=%.1fs" % cs) if cs else ("时间窗=%ss" % d.get("duration_s"))
print(f"  {head}  steps/s={d.get('steps_per_second')}  sql_p99={d.get('sql_exec_p99_ms')}ms  "
      f"vec_p99={d.get('vector_search_p99_ms')}ms  tp_p99={d.get('tp_txn_p99_ms')}ms  "
      f"记忆+{d.get('memories_after')}行  rc={rc}")
PY
exit $RC
