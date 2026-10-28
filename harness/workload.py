"""复合负载生成器: agent 向量检索(AP前置) + LLM 生成的 SQL(AP) + 记忆写回(TP)。

一个 agent = 一个进程 = 一个连接, 沿用 cognitivebench 的模型。
每步:
  1. 问题向量 -> chnsw 检索 value linking (top-k)          [向量查询]
  2. chnsw 检索历史记忆                                     [向量查询]
  3. LLM 生成 SQL (真调 or 回放缓存) —— 不计入指标           [不测]
  4. 执行那条 SQL                                           [AP]
  5. 写回一条记忆(带向量)                                  [TP]

指标全部是服务端侧的: LLM 的 1167ms 不进任何数字。
TP:AP 配比通过 --writes-per-step 调(每步写几条记忆)。
"""
from __future__ import annotations
import argparse, collections, json, os, random, signal, sys, time
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BIRD = os.environ.get("BIRD_DIR", "bird/dev_20240627")   # BIRD dev 集解压目录
DSN  = os.environ.get("BIRD_DSN", "postgresql://postgres@localhost:5433/bird_bench_run")

MEM_DDL = """
CREATE TABLE IF NOT EXISTS agent_memories (
    memory_id  bigserial PRIMARY KEY,
    agent_id   int  NOT NULL,
    db_id      text NOT NULL,
    qid        int,
    kind       text,                    -- attempt / success / error
    content    text,
    embedding  vector(768),
    created_at timestamptz DEFAULT now()
);
-- 必须有 chnsw 索引: 没有索引时 memory_search 走顺序扫描, 首测量到 55.84ms p50,
-- 占单步时间 82%, 把整个负载形状带偏, 而且随写入单调变慢。
CREATE INDEX IF NOT EXISTS agent_memories_chnsw_idx ON agent_memories
    USING cognitivehnsw (embedding cognitivehnsw_cosine_ops) WITH (m=16, ef_construction=64);

-- ── TP 侧 ──────────────────────────────────────────────────────────────────
-- 之前的"TP"只有 INSERT: 无 UPDATE、无行级争用、无多语句事务, 名不副实。
-- 下面三样让 TP 长在 agent 语义里而不是外挂 pgbench:
--   agent_state  每 agent 一行, 每步 UPDATE —— 无争用基线
--   memory_stats 记忆的访问计数, UPDATE 刚检索到的那几行 —— 热行, 真实记忆系统
--                检索后都要 bump 访问次数/时效性, 这是 recency 排序的基础
-- 业务写入   规格 §5.1.1: 一定比例的认知步骤在**同一事务内**写业务 schema,
--            "modeling an agent that acts on business data (update an order, post a
--            record) and memorizes the action atomically"。
--            用 financial 库(PKDD'99 真实银行数据), 结构与 pgbench(TPC-B)一一对应:
--              account   4,500 行  <- pgbench_accounts, 低争用
--              district     77 行  <- pgbench_branches, 争用点
--              trans   1,056,320 行 <- pgbench_history,  只增
--            77 行的选择有实测依据: P5 争用梯度显示 48 并发下 77 行只掉 1.28x
--            (5000行 6586/s -> 77行 5127/s), 而 8 行掉到 0.21x。悬崖在 8 行以下,
--            所以 77 行是温和的真实争用点。原先自造的 db_stats 只有 11 行, 落在悬崖上,
--            是个极端值而不是可调梯度 —— 已弃用。
ALTER TABLE agent_memories ADD COLUMN IF NOT EXISTS access_count bigint DEFAULT 0;
ALTER TABLE agent_memories ADD COLUMN IF NOT EXISTS last_access timestamptz;

CREATE TABLE IF NOT EXISTS agent_state (
    agent_id   int PRIMARY KEY,
    steps      bigint      DEFAULT 0,
    last_db    text,
    last_qid   int,
    updated_at timestamptz DEFAULT now()
);

-- TPC-B 的计数放**独立表**, BIRD 的表保持原版(不加任何列)——
-- 给 BIRD 的表加列就不能再声称"在标准 BIRD 上测的"。
-- 争用金字塔靠行数差异实现, 与 pgbench 对应:
--   biz_account  4,500 行 <- pgbench_accounts (低争用)
--   biz_district    77 行 <- pgbench_branches (争用点; P5 实测 48 并发只掉 1.28x)
CREATE TABLE IF NOT EXISTS biz_account  (account_id  bigint PRIMARY KEY,
                                         balance   bigint NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS biz_district (district_id bigint PRIMARY KEY,
                                         txn_total bigint NOT NULL DEFAULT 0);
-- 测试库没有 financial.trans 可写, 业务写入落这里。形态对齐: 每笔业务写产生
-- 一条新行, 带自增主键和时间戳, 行宽与 trans 同量级。
CREATE TABLE IF NOT EXISTS biz_write_log (id bigserial PRIMARY KEY,
                                          account_id bigint, amount bigint,
                                          wrote_at timestamptz DEFAULT now());
-- 行数(4500 / 77)必须与 dev 一致 —— TP 的争用结构靠行数差异实现,
-- 换题集时若行数变了, tp_txn 就不再是"固定动作", 对比也就不成立。
-- dev 从 financial 表取真实 id; 测试库没有那个 schema, 直接生成同样多的行。
DO $seed$
BEGIN
  IF to_regclass('financial.account') IS NOT NULL THEN
    -- dev: biz_account 上有外键指向 financial.account, 必须用真实 id
    INSERT INTO biz_account  (account_id)  SELECT account_id  FROM financial.account  ON CONFLICT DO NOTHING;
    INSERT INTO biz_district (district_id) SELECT district_id FROM financial.district ON CONFLICT DO NOTHING;
    -- 争用旋钮(2026-10-09, 预注册 prereg_district.json): 默认 __DISTRICT_ROWS__=77 时
    -- generate_series(78,77) 为空集, 行为与改动前**完全一致**。
    INSERT INTO biz_district (district_id) SELECT generate_series(78,__DISTRICT_ROWS__) ON CONFLICT DO NOTHING;
  ELSE
    -- 测试库没有 financial schema; 行数仍锁死 4500/77, 保证争用结构一致
    INSERT INTO biz_account  (account_id)  SELECT generate_series(1,4500) ON CONFLICT DO NOTHING;
    INSERT INTO biz_district (district_id) SELECT generate_series(1,__DISTRICT_ROWS__) ON CONFLICT DO NOTHING;
  END IF;
END $seed$;
"""

def worker(agent_id, args, questions, barrier, stop, resdir, ticket=None):
  try:
    import psycopg2, numpy as np
    from llm import LLM
    random.seed(1000 + agent_id)
    # 2026-10-09 修坑1: 运行时写回的记忆 embedding 原先直接用原始问题向量
    #   (零抖动), 同一道题写出的所有记忆逐位相; 而 prefill_memories.py:47 给预填的
    #   50 万条每条加了独立抖动再归一化(注释原话: "每条加独立抖动 -> 向量唯一")。
    #   两条路不同分布, 且运行时那条是退化的: 往 HNSW 里插与已有点距离为 0 的点,
    #   图构建与检索行为都不代表真实记忆负载 —— 而这正是本项目要建模的量。
    #   修法: 用与预填完全相同的抖动形式(sigma=0.05/sqrt(D), 再归一化)。
    #   种子 20260906+agent_id, 每 agent 固定 -> 整份采集仍逐位可复现。
    _JIT = float(os.environ.get("MEM_JITTER", "0.05"))
    _mrng = np.random.default_rng(20260906 + agent_id)
    _zr = np.load(VECS_ENV or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "question_vecs.npz"))
    _qraw = {int(q): e for q, e in zip(_zr["qids"], _zr["embs"])}
    _D = int(_zr["embs"].shape[1])
    def _jit_vec(qid_):
        """与 prefill_memories.py:47 同形的抖动 + 归一化, 返回 pgvector 字面量。"""
        if _JIT <= 0:
            return None
        v = _qraw[qid_] + _mrng.standard_normal(_D).astype(np.float32) * (_JIT / np.sqrt(_D))
        v = v / np.linalg.norm(v)
        return "[" + ",".join(f"{x:.6f}" for x in v) + "]"
    conn = psycopg2.connect(DSN); conn.autocommit = True
    cur = conn.cursor()
    # 事务内分桶计时器, 按来源分, 不按"是不是向量"分:
    #   biz   = 业务事务(TPC-B 形态) —— BIRD 的 SQL 全是 AP, 这是为此补上的 TP
    #   mem   = 记忆循环(检索后强化/修剪/信念修正/把 agent<->LLM 这一轮的结果写回成记忆)
    #   vlink = 业务写产生的新文本值进 value-linking 向量索引(跨域原子性, 系统卖点)
    #   state = agent 步数簿记
    # _bk[0] 是当前分桶, 在组边界处设置; _tpa 每笔事务前清零。
    _tpa = {"biz": 0.0, "mem": 0.0, "vlink": 0.0, "state": 0.0}
    _bk = ["state"]
    def _x(q, pr=None):
        _t = time.perf_counter()
        cur.execute(q) if pr is None else cur.execute(q, pr)
        _tpa[_bk[0]] += time.perf_counter() - _t
    cur.execute(f"SET cognitivehnsw.ef_search = {args.ef_search}")
    cur.execute(f"SET statement_timeout = {args.stmt_timeout_ms}")
    cur.execute("INSERT INTO public.agent_state (agent_id) VALUES (%s) "
                "ON CONFLICT (agent_id) DO NOTHING", (agent_id,))

    # 预生成问题向量, 避免把 embedding 的 CPU 开销算进负载
    qvecs = {q["question_id"]: q["_vec"] for q in questions}
    # 新业务值池(离线算好的 embedding)。真实 agent 在等 LLM 的空闲窗口里算,
    # 不占额外时间; 回放模式没有真实窗口, 用预算的池子等价替代。
    # 新业务值池: 用真实的、未入库的 BIRD 值, 不用我合成的银行代码。
    # 手上有 523,178 个真实值未入库(被每列上限砍掉 299,030 + ID 列 199,729 + 自由文本 24,419),
    # 其中 card_games.foreign_data.name 的 110,557 个卡牌译名正是有语义、值得检索的值。
    _bv = None
    if args.value_sync != "off":
        import numpy as _np, os as _os
        _f = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "unindexed_vals.npz")
        if _os.path.exists(_f):
            _z = _np.load(_f, allow_pickle=False)
            _bv = {"t": _z["tables"], "c": _z["columns"], "v": _z["values"],
                   "d": _z["docs"], "e": _z["embs"]}
    llm = LLM("groq", args.model, cache_path="llm_cache.sqlite",
              max_tokens=1500,          # 与 retry_trace.py 一致(缓存键不含 max_tokens)
              replay_only=args.replay_only)

    # 2026-10-09: tp_txn 拆分。原先一个计时器套住整个显式事务, 里面同时装着
    #   (a) 用户当初要求补上的 TP SQL(5a/5b/5c/5d, 行级读-改-写, TPC-B 形态)与
    #   (b) HNSW 向量索引维护(5c 尾的 value_vectors、5e、以及每步必做的 5f),
    #   于是"tp_txn 漂移"一直在替向量索引维护和设备饱和背锅。
    #   纯加计时器: 语句/顺序/事务边界一个字不动, tp_txn 外层总量不变, 与历史可比。
    lat = {"vec_value": [], "vec_mem": [], "sql": [], "tp_txn": [],
           "tp_biz": [], "tp_mem": [], "tp_vlink": [], "tp_state": [], "tp_commit": []}
    # 2026-10-07: 逐条查询延迟, 按 question_id 分组。用途: 算与文献同口径的
    #   单查询 Q-error(Hilprecht PVLDB'22 的 max(c/c_hat, c_hat/c) 是每条查询
    #   的量)。此前只落盘 p50/p95/p99 三个分位, 拿分位去和文献的单查询数并排是
    #   不成立的 —— 预测分位比预测单条容易得多。每条查询在一次运行里被执行约
    #   steps/|Q| 次, 取中位正好消掉并发抖动。
    lat_by_q = {}
    n_step = n_sqlerr = n_llmerr = n_tperr = 0
    n_biz = n_del = n_embupd = 0
    n_vsync = n_vstale = 0
    n_rounds_done = 0
    barrier.wait(120)          # 超时而非无限等 —— 有 worker 提前挂掉时要报错不要挂死
    # 固定功量模式: 每个 agent 跑 steps/agents 步。
    # 为什么不用固定时间窗: 每步都写入并撑大索引, 180 秒里 32 核写了 168,104 行
    # (索引 1041MB) 而 4 核只写了 33,243 行 (210MB) —— 快的配置到窗口末尾要对着
    # 5.0x 大的索引干活, 被自己产生的增长惩罚, 伸缩比被低估。固定功量下每个点
    # 做完全相同的功、结束在完全相同的状态。
    # 确定性分片: agent i 固定拿 questions[i::agents], 顺序遍历, 重复 K 轮。
    # 不用随机抽样的原因: 24,000 步从 280 题里随机有放回抽, 每题实际被执行 61-112 次
    # (期望 85.7, 标准差 9.2, 极差 1.84x)。而金标准 SQL 里 3.2% 的重查询占 50% 的
    # 执行时间 —— 各配置抽到的重查询次数不, 是纯粹的方差来源。
    # 整轮遍历下每题恰好 K 次, 各点执行的 SQL 序列逐条相。
    # ── 静态分片 vs 动态领题 ────────────────────────────────────────────────
    # 静态分片(questions[agent_id::agents])的问题: 题目开销跨 4 个数量级
    # (p50 1.6ms, max 20.8s), 分片必然轻重悬殊 -> 墙钟由抽到最重分片的 agent 决定。
    # 实测 18 核那点 agent 只忙了 1920s, 却要等到相当于 15779 agent-秒才收工 ——
    # 88% 的墙钟是跑完的 agent 在等最慢那个。
    # 动态领题: 所有 agent 从一个共享票号里领下一步, 工作量一步不差(总步数仍是
    # passes x 题数), 但没人空等 -> 墙钟 ~= 总功量/聚合吞吐, 完成时间重新有意义。
    # 代价: agent 的记忆不再聚焦于自己那几十道题(话题一致性变差), 论文里如实写。
    _dynamic = ticket is not None
    if _dynamic:
        my_qs = questions                      # 全体共享, 用票号定位
        my_steps = 0                           # 由票号耗尽决定何时停
        _total_steps = (args.passes * len(questions)) if args.passes else\
                       (args.steps if args.steps else 0)
    else:
        my_qs = questions[agent_id::args.agents] or questions[:1]
        my_steps = (args.passes * len(my_qs)) if args.passes else\
                   ((args.steps // args.agents) if args.steps else 0)
    t_end = time.time() + (args.duration if not (args.steps or args.passes) else 86400)
    t_begin = time.time()

    while not stop.is_set():
        if _dynamic:
            with ticket.get_lock():
                _gs = ticket.value
                if _total_steps and _gs >= _total_steps: break
                ticket.value = _gs + 1
            _idx = _gs
        else:
            if my_steps and n_step >= my_steps: break
            if not my_steps and time.time() >= t_end: break
            _idx = n_step
        # 一步 = 一道题的一轮。每一遍(pass)里每道题恰好 1 步: qi = 步号 % 题数;
        # 多轮的题把各轮分摊到不同的遍上(第 k 遍用第 k % 轮数 轮), 不是同一遍里连跑几步。
        # 所以 steps = passes × 题数(如 245 × 100 = 24,500)。(2026-10-16 改正注释, 代码未动)
        qi = _idx % len(my_qs)
        q = my_qs[qi]
        _rounds = q.get("_sql_rounds") or [q.get("_llm_sql")]
        _round_i = (_idx // len(my_qs)) % len(_rounds)
        db, qid = q["db_id"], q["question_id"]
        vec = qvecs[qid]
        # 本步要写回的记忆向量, 在事务外算好: 否则 numpy 开销会被计进 tp_txn/tp_mem。
        # 每条写入各自独立抖动(预填一致); MEM_JITTER=0 可退回旧行为做对照。
        _wvecs = [(_jit_vec(qid) or vec) for _ in range(max(args.writes_per_step, 1))]
        try:
            cur.execute(f'SET search_path TO "{db}", public')
            # 1. value linking 检索
            t0 = time.perf_counter()
            cur.execute("SELECT cell_value, table_name, column_name FROM public.value_vectors "
                        "WHERE db_id = %s ORDER BY embedding <=> %s::vector LIMIT %s",
                        (db, vec, args.topk))
            cur.fetchall(); lat["vec_value"].append((time.perf_counter()-t0)*1000)
            # 2. 历史记忆检索
            t0 = time.perf_counter()
            cur.execute("SELECT memory_id FROM public.agent_memories "
                        "ORDER BY embedding <=> %s::vector LIMIT %s", (vec, args.topk_mem))
            hit_ids = [r[0] for r in cur.fetchall()]
            lat["vec_mem"].append((time.perf_counter()-t0)*1000)
        except Exception:
            conn.rollback(); continue

        # 3. LLM —— 不计时
        sql = _rounds[_round_i] if _rounds and _rounds[_round_i] else q.get("_llm_sql")
        n_rounds_done += 1
        if sql is None:
            try:
                r = llm.complete(q["_system"], q["_prompt"])
                sql = q["_llm_sql"] = _extract(r.text)
            except Exception:
                n_llmerr += 1; sql = None
        # 4. 执行 SQL  [AP]
        outcome = None
        if sql:
            t0 = time.perf_counter()
            try:
                cur.execute(sql); rows = cur.fetchall()
                _ms=(time.perf_counter()-t0)*1000
                lat["sql"].append(_ms); lat_by_q.setdefault(qid, []).append(_ms)
                outcome = ("success", "SQL: %s -> %d rows" % (sql, len(rows)))
            except Exception as e:
                conn.rollback(); n_sqlerr += 1
                outcome = ("error", "SQL: %s -> ERROR: %s" % (sql, str(e).split(chr(10))[0]))
        # 5. TP 事务: 一个显式事务里做 读-改-写, 含热行 UPDATE
        t0 = time.perf_counter()
        for _k in _tpa: _tpa[_k] = 0.0
        _bk[0] = "state"
        try:
            conn.autocommit = False
            _bk[0] = "state"
            # 5a. 自己的状态行 —— 无争用基线
            _pending_dist = None
            _x("UPDATE public.agent_state SET steps = steps + 1, last_db = %s, "
                        "last_qid = %s, updated_at = now() WHERE agent_id = %s",
                        (db, qid, agent_id))
            _bk[0] = "mem"
            # 5b. 刚检索到的记忆 bump 访问次数 + last_access —— 热行, 多 agent 争用
            #     依据: Generative Agents (arXiv:2304.03442) 的检索评分
            #     α_rec·recency + α_rel·relevance + α_imp·importance, 其中 recency 是
            #     按"距上次访问"的指数衰减 γ=0.995/小时 -> 命中即更新 last_access 是该
            #     评分函数的必要写操作。MemoryBank (arXiv:2305.10250) 用 Ebbinghaus
            #     遗忘曲线, 同样依赖访问时间与强化次数。Mem0 issue #5330。
            #     实测: 这一条使 TP p99 +206%, 死元组 60x —— TP 侧最大贡献项。
            if hit_ids and args.tp_updates:
                _x("UPDATE public.agent_memories SET access_count = access_count + 1, "
                            "last_access = now() WHERE memory_id = ANY(%s)", (hit_ids,))
            _bk[0] = "biz"
            # 5c. 业务写入(规格 §5.1.1): TPC-B 式的一笔账, 在同一事务内
            #     形态依据: TPC Benchmark B v2.0 标准事务, 以及 PostgreSQL 文档里
            #     pgbench 内置脚本 "loosely based on TPC-B"。三级争用结构
            #     (branch 少而热 / account 多而冷) 照搬 TPC-B。
            #     ⚠ business_write_pct=0.2 无外部依据(规格自标 calibration open),
            #       论文里必须作为敏感性维度扫 {0, 0.2, 0.5}, 不能当固定配置。
            #     -> 跨域原子性: 业务写入和记忆一起提交或一起回滚
            if random.random() < args.business_write_pct:
                acct = random.randrange(1, 4501)          # account 4,500 行, 低争用
                # 2026-10-09 修 BUG: 原为写死的 randrange(1, 78), 使 BIZ_DISTRICT_ROWS
                #   旋钮完全失效 —— 加出来的行一次都不会被碰到。默认 77 时行为不变。
                dist = random.randrange(1, _DISTRICT_ROWS + 1)   # 争用点
                amt  = random.randrange(1, 1000)
                # 计数在独立表(public.biz_account / biz_district), BIRD 的表保持原版 ——
                # 给 BIRD 的表加列就不能再声称"在标准 BIRD 上测的"。
                _x("UPDATE public.biz_account SET balance = balance + %s "
                            "WHERE account_id = %s", (amt, acct))
                _x("SELECT balance FROM public.biz_account WHERE account_id = %s", (acct,))
                cur.fetchone()
                if _TP_ORDER == "late":
                    _pending_dist = (amt, dist)      # 推到 COMMIT 前
                else:
                    _x("UPDATE public.biz_district SET txn_total = txn_total + %s "
                                "WHERE district_id = %s", (amt, dist))
                # 业务写入引入一个新的文本值 —— 真实业务写入会产生新客户名/新代码,
                # 这些值必须进 value linking 索引, 否则 agent 检索不到刚写入的业务真相。
                if _bv is not None:
                    j = random.randrange(len(_bv["v"]))
                    newval = str(_bv["v"][j])
                    # trans_id 必须给: 原版 BIRD 里它是主键, 不给会插入 NULL 主键行
                    # (实测漏掉时 run 库积累了 3,910 行 NULL 主键, 导致主键建不上)
                    if _trans_ok(cur):
                        _x("INSERT INTO financial.trans "
                                    "(trans_id, account_id, date, type, operation, amount, balance, bank) "
                                    "VALUES (nextval('trans_id_seq'), %s, current_date, "
                                    "'PRIJEM', 'VKLAD', %s, %s, %s)",
                                    (acct, amt, amt, newval[:2]))
                    else:
                        # 没有 financial.trans 的库写进 biz_write_log, 形态一致:
                        # 每笔业务写产生一条新行。不能让整个事务失败。
                        _x("INSERT INTO public.biz_write_log "
                                    "(account_id, amount, wrote_at) VALUES (%s,%s,now())",
                                    (acct, amt))
                    if args.value_sync == "atomic":
                        # 关键: 向量插入和业务写入在同一个事务里 -> staleness = 0
                        #   独立向量库做不到这一步(跨系统无法原子提交)。
                        #   embedding 是事务外算好的, 事务内只做一次插入。
                        _bk[0] = "vlink"
                        vec_lit = "[" + ",".join(f"{x:.6f}" for x in _bv["e"][j]) + "]"
                        _x(
                            "INSERT INTO public.value_vectors "
                            "(db_id, table_name, column_name, cell_value, doc, embedding) "
                            "VALUES (%s,%s,%s,%s,%s,%s::vector)",
                            # 2026-10-09 修 BUG: 原为硬编码的 "financial"。value_vectors
                            #   是按 db_id 分区的表, 测试库上根本没有 financial 分区
                            #   (bird_test 只有 vv_airline / vv_chicago_crime / vv_movielens /
                            #   vv_public_review_platform), 于是每笔业务写事务都以
                            #   "no partition of relation value_vectors found for row" 中止 ——
                            #   业务写入与同事务的向量插入一起回滚。实测 bird_test:
                            #   business_writes=0, tp_errors≈4173。
                            #   db_id 取当前 agent 正在查的库, 它必然有分区; 这也让训练库
                            #   不再把全部向量插入堆进 financial 这一个分区。
                            (db, str(_bv["t"][j]), str(_bv["c"][j]),
                             newval, str(_bv["d"][j]), vec_lit))
                        n_vsync += 1
                        _bk[0] = "biz"
                    else:
                        n_vstale += 1     # deferred: 业务值已提交但索引里还没有
                else:
                    if _trans_ok(cur):
                        _x("INSERT INTO financial.trans "
                                    "(trans_id, account_id, date, type, operation, amount, balance) "
                                    "VALUES (nextval('trans_id_seq'), %s, current_date, "
                                    "'PRIJEM', 'VKLAD', %s, %s)",
                                    (acct, amt, amt))
                    else:
                        # 测试库没有 financial schema。写进 biz_write_log, 保持
                        # "每笔业务写产生一条新行"的形态, 行宽和索引开销可比。
                        _x("INSERT INTO public.biz_write_log "
                                    "(account_id, amount, wrote_at) VALUES (%s,%s,now())",
                                    (acct, amt))
                n_biz += 1
            _bk[0] = "mem"
            # 5d. 记忆修剪(规格 §6.3): DELETE 不回收索引空间(实测 +0.00MB),
            #     比例 0.01 的依据: MemTrace (arXiv:2605.28732) 脚注 5 —— delete 占
            #     Mem0 全部 add/update/delete 操作的 1.02%。不是我拍的数。
            #     只留墓碑并产生死元组 -> autovacuum 压力。这是 C 类写操作。
            if args.delete_rate > 0 and random.random() < args.delete_rate:
                _x("DELETE FROM public.agent_memories WHERE memory_id IN "
                            "(SELECT memory_id FROM public.agent_memories WHERE agent_id = %s "
                            " ORDER BY memory_id LIMIT 1)", (agent_id,))
                n_del += cur.rowcount

            # 5e. 信念修正(规格 §6.x): UPDATE embedding。实测代价与 INSERT 相同
            #     (3.186 vs 3.073 ms, 索引均 +1.17MB/300次) —— PG 的 MVCC 下 UPDATE 就是
            #     插新版本, 索引侧只插不删, 不存在"拆边重连"。但会产生死元组。
            if args.embedding_update_rate > 0 and hit_ids and\
               random.random() < args.embedding_update_rate:
                _x("UPDATE public.agent_memories SET embedding = %s::vector, "
                            "content = %s WHERE memory_id = %s",
                            (_wvecs[0], f"revised by agent{agent_id}", hit_ids[0]))
                n_embupd += cur.rowcount

            _bk[0] = "mem"
            # 5f. 写回新记忆。内容用真实的 SQL 和报错文本, 不用占位串 ——
            #     原先写 "agent12 q1511 step1119"(21 字节), 而真实记忆是几百字节,
            #     占位串会让 TOAST / WAL 量 / 缓冲池占用全部偏小, 而这些都是实测量。
            kind, content = outcome if outcome else ("attempt", "no SQL generated")
            for _wi in range(args.writes_per_step):
                _x("INSERT INTO public.agent_memories "
                            "(agent_id,db_id,qid,kind,content,embedding) "
                            "VALUES (%s,%s,%s,%s,%s,%s::vector)",
                            (agent_id, db, qid, kind, content, _wvecs[_wi]))
            _bk[0] = "biz"
            if _pending_dist is not None:        # 实验臂 late: 临界区只剩这一条
                _x("UPDATE public.biz_district SET txn_total = txn_total + %s "
                            "WHERE district_id = %s", _pending_dist)
            _tc = time.perf_counter()
            conn.commit()
            _tc = (time.perf_counter()-_tc)*1000
            lat["tp_txn"].append((time.perf_counter()-t0)*1000)
            lat["tp_biz"].append(_tpa["biz"]*1000)
            lat["tp_mem"].append(_tpa["mem"]*1000)
            lat["tp_vlink"].append(_tpa["vlink"]*1000)
            lat["tp_state"].append(_tpa["state"]*1000)
            lat["tp_commit"].append(_tc)
        except Exception:
            conn.rollback(); n_tperr += 1
        finally:
            conn.autocommit = True
        n_step += 1

    # 不用 mp.Queue: 原始延迟列表在 8 agents x 60s 下就有数万个 float, 超过管道缓冲区
    # -> worker 阻塞在 flush 而父进程卡在 join, 双向死锁(实测挂 12 分钟, 0 连接)。
    # 写文件, 扩到几百 agent 也安全。
    my_elapsed = time.time() - t_begin
    with open(os.path.join(resdir, "agent%03d.json" % agent_id), "w") as fh:
        json.dump({"agent": agent_id, "steps": n_step, "sql_err": n_sqlerr,
                   "llm_err": n_llmerr, "tp_err": n_tperr,
                   "biz": n_biz, "del": n_del, "embupd": n_embupd,
                   "vsync": n_vsync, "vstale": n_vstale,
                   "elapsed": my_elapsed, "lat": lat,
                   "lat_by_q": lat_by_q}, fh)
    cur.close(); conn.close()
  except Exception as e:
    import traceback
    print(f"[agent {agent_id}] 异常: {e}", flush=True); traceback.print_exc()
    try: barrier.abort()
    except Exception: pass

def _extract(text):
    from prompts import extract_sql
    return extract_sql(text)

# prepare 里解析出的轨迹来源, 供 main 写进 point.json ——
# 每个点必须能事后核对"我用的是哪份轨迹快照、多少题", 否则采集与网格并行时
# 无法证明各点工作量一致。
# dev 的 TP 业务写入落在 financial.trans; 测试库(train 的 4 个库)没有这个
# schema, 置空则落到 public.biz_write_log。行数与争用结构保持一致。
BIZ_SCHEMA = os.environ.get("BIRD_BIZ_SCHEMA", "financial")

# 争用点的行数。pgbench 官方文档: scale factor(= branches 行数)应 >= 最大客户端数,
# 否则该行就是瓶颈。77 行配 224 并发远低于这条规则。默认仍为 77, 不改变既有行为。
_DISTRICT_ROWS = int(os.environ.get("BIZ_DISTRICT_ROWS", "77"))
# 实验臂(2026-10-09, prereg_tpvar.json): late = 把争用行的 UPDATE 推到 COMMIT 前最后一条,
# 把它的行锁临界区从"跨 2 次 HNSW 插入 + DELETE"缩成一条语句。同一事务, 原子性不变。
_TP_ORDER = os.environ.get("TP_ORDER", "orig")

# 2026-10-09 修 BUG: 业务写入的目标表必须按库实际存在什么来定, 不能靠环境变量,
#   更不能硬写。原代码在 `_bv is not None` 分支里硬写 financial.trans, 完全绕过 BIZ_SCHEMA ——
#   bird_test / bird_test2 根本没有 trans 表(任何 schema 下都没有), 于是每笔业务写事务
#   直接 abort。后果不只是业务写丢失: 同一事务里的 value_vectors 插入(原子性卖点)
#   一并回滚, 所以测试库的 tp_txn 与 vec_value 两个分量都是坏数据。
#   实测: bird_test business_writes=0, tp_errors≈4913, 20% 的事务回滚。
_TRANS_OK = None
def _trans_ok(cur):
    """这个库里有没有 financial.trans。每进程查一次并缓存。"""
    global _TRANS_OK
    if _TRANS_OK is None:
        cur.execute("SELECT to_regclass('financial.trans') IS NOT NULL")
        _TRANS_OK = bool(cur.fetchone()[0])
    return _TRANS_OK
# 泛化测试集: 题集与题向量文件。空则用 dev 的。
QFILE_ENV = os.environ.get("BIRD_QUESTIONS", "")
VECS_ENV  = os.environ.get("BIRD_VECS", "")
TRACE_INFO = {"file": "", "path": "", "n_questions": 0}


def prepare(args):
    """选题、加载预算好的问题向量、拼 prompt。

    绝不能在这里 import torch / sentence_transformers ——
    父进程加载 torch 后 fork, 子进程会继承加锁的 OpenMP 互斥量, 8 个 worker
    全部卡死在 barrier(实测 990 秒零连接)。向量由 precompute_vecs.py 预先算好。
    """
    import psycopg2, numpy as np
    from pg_schema import schema_text
    vf = VECS_ENV or os.path.join(os.path.dirname(os.path.abspath(__file__)), "question_vecs.npz")
    if not os.path.exists(vf):
        sys.exit("缺 question_vecs.npz —— 先跑 python3 precompute_vecs.py")
    z = np.load(vf)
    vecmap = {int(q): e for q, e in zip(z["qids"], z["embs"])}
    d = json.load(open(QFILE_ENV)) if QFILE_ENV else json.load(open(f"{BIRD}/dev.json"))
    if args.dbs: d = [q for q in d if q["db_id"] in args.dbs.split(",")]
    random.seed(42); random.shuffle(d)
    # 不能在这里就截断: 缓存只覆盖一部分题, 先截断再筛缓存会把命中的丢掉。
    # 实测 --questions 247 先截断后只剩 34 题可回放 -> 11.7 万步在 34 条 SQL 上循环。
    # 正确顺序: 拼好 prompt -> 筛出有缓存的 -> 再取前 N 道。
    from prompts import SYSTEM, build_prompt, looks_truncated, looks_like_sql
    schemas = {}
    for q in d:
        e = vecmap[q["question_id"]]
        q["_vec"] = "[" + ",".join(f"{x:.6f}" for x in e) + "]"
        db = q["db_id"]
        if db not in schemas: schemas[db] = schema_text(DSN, db)
        q["_system"] = SYSTEM
        q["_prompt"] = build_prompt(schemas[db], q["question"], q.get("evidence"))
    # 多轮返工轨迹: 一道题不是一步就完, 实测单轮只有 52.1% 正确, 48% 要带报错返工,
    # 平均约 1.75 轮。每轮都是一整轮库内往返(2 次向量检索 + 1 条 SQL + 1 个 TP 事务),
    # 所以"一步只算一轮"低估了约 1.75 倍的数据库工作量。
    # 轨迹由 retry_trace.py 用真 LLM 采集(受 RPD 1000 限, 增量落盘)。
    # 没有轨迹的题退回单轮 —— 保真度随采集自动提升, 不用等采满。
    trace = {}; trace_prov = {}   # qid -> 哪个(些) provider 答的, 论文里要报异质性
    # 慢查询排除名单(qid, round), 来自 screen_slow_sql.py 的离线筛查。
    # 为什么筛题池而不是调 statement_timeout: 问题出在 LLM 生成的 SQL(偶尔写出
    # 没有连接条件的 join), 不在数据库配置。筛掉之后 statement_timeout 能保持
    # BIRD 官方评测的 30 s, 判对错与回放负载同一口径。
    # 实测不筛的后果: 5 条离群查询(最慢 20.8s, 而 p50 只有 1.6ms)占住 5 个 agent,
    # 整点墙钟由它们决定 -> 端到端时间 2->18 核只快 1.06 倍, 而分项各快 1.63~8.20 倍。
    _slow = set()
    _sf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sql_screen.json")
    if os.path.exists(_sf):
        try:
            _sc = json.load(open(_sf))
            _slow = {(r["qid"], r.get("round", 1)) for r in _sc["rows"] if r.get("excluded")}
            print(f"  慢查询筛除: {len(_slow)} 轮(阈值 "
                  f"{_sc['meta']['threshold_ms']:.0f}ms)", flush=True)
        except Exception as e:
            print(f"  !! sql_screen.json 读取失败, 未排除慢查询: {e}", flush=True)
    else:
        print("  !! 没有 sql_screen.json —— 未排除慢查询, 资源轴可能被离群查询压平",
              flush=True)
    # 轨迹文件可被 --trace-file 固定。为什么必须能固定: LLM 采集要跑约 16 天,
    # 网格只跑约 1 天, 两者必然并行 —— 若每个点自己去读 retry_trace.json,
    # 采集每走完一轮就换一次题集, 不同点悄悄用了不同工作量, 主 target 不可比。
    tf = args.trace_file or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "retry_trace.json")
    TRACE_INFO["file"] = os.path.basename(tf) if os.path.exists(tf) else ""
    TRACE_INFO["path"] = tf
    if args.use_trace and os.path.exists(tf):
        for x in json.load(open(tf)):
            # 排除 truncated 轮: 那是 max_tokens 截断的产物(实测 800 时 6.4% 调用撞上限),
            # 属于测量假象, 不是 agent 行为 —— 放进轨迹会让负载里出现空/半截 SQL。
            # 除了 truncated 标记, 再用结构判据兜一层 —— 旧轨迹里有一批
            # Gemini 的半截 SQL 当时没被标上(token 判据对它失效)。
            sqls = [r.get("sql") for r in x.get("rounds", [])
                    if r.get("sql") and not r.get("truncated")
                    and not looks_truncated(r["sql"])
                    # 第三道: 抽取环节偶尔把模型的思考文本当成 SQL 收下来, 且恰好
                    # 引号括号配平, 前两道拦不住。实测快照里 14 轮是这种货色 ——
                    # 与 truncated 同理, 属测量假象而非 agent 行为, 必须排除。
                    and looks_like_sql(r["sql"])
                    and (x["qid"], r.get("round", 1)) not in _slow]
            provs = {r.get("provider") for r in x.get("rounds", []) if r.get("provider")}
            if provs: trace_prov[x["qid"]] = sorted(provs)
            if sqls: trace[x["qid"]] = sqls
        TRACE_INFO["n_questions"] = len(trace)
        print(f"  载入多轮轨迹 {len(trace)} 题, 平均 "
              f"{sum(len(v) for v in trace.values())/max(len(trace),1):.2f} 轮", flush=True)

    if args.replay_only:
        # 只保留缓存里有响应的题。原来未命中会抛异常搞死 worker ——
        # 而缓存是逐步长起来的, 不该让扫描依赖"缓存必须完整"。
        from llm import ResponseCache
        cache = ResponseCache("llm_cache.sqlite")
        keep = []
        # 缓存键含 provider+model, 多 provider 采集后同一道题可能是 gemini/cerebras 答的
        # —— 只查 "groq" 会把它们全丢掉。按轨迹记录的 provider 优先, 再兜底试全部免费档。
        from llm import FREE_TIER
        # FREE_TIER 现在是 {"provider:model": 每日次数}, 不再是 (模型, TPM) 元组
        cands = [("groq", args.model)] + [tuple(k.split(":", 1)) for k in FREE_TIER]
        # 2026-10-14: FREE_TIER 是【限流表】, 管的是还能不能发新请求, 不该用来筛
        #   【缓存读取】—— 一条缓存就是一条缓存, 跟那个模型今天还在不在免费档无关。
        #   实测后果: llm.py 在 2026-10-03 把 groq/compound-mini 从表里删掉, 它答过的
        #   255 条缓存就此查不到, 题池无声变小 —— testset_100 从 100 题掉到 85 题,
        #   pdb_superhero 129->114, pdb_formula_1 174->156。每个点都一样地少, 所以
        #   组内仍自洽, 但跨 10-03 的测量不可比, 而且 --questions N 给不出 N。
        #   这里不改默认行为 (改了 rc5 就和 rc3/rc4 不可比); 要补回用环境变量显式开。
        _extra = os.environ.get("BIRD_CACHE_MODELS", "").strip()
        if _extra:
            cands += [tuple(x.split(":", 1)) for x in _extra.split(",") if ":" in x]
            print(f"  缓存候选追加: {_extra}", flush=True)
        seen = set(); cands = [c for c in cands if not (c in seen or seen.add(c))]
        prov_used = collections.Counter()
        for q in d:
            hit = None
            order = [(p, m) for p, m in cands
                     if p in (trace_prov.get(q["question_id"]) or [])] + cands
            for _p, _m in order:
                hit = cache.get(ResponseCache.key(_p, _m, q["_system"], q["_prompt"]))
                if hit: prov_used[_p] += 1; break
            if hit:
                from prompts import extract_sql
                q["_llm_sql"] = extract_sql(hit[0])
                # 有轨迹就用完整的多轮 SQL 序列, 没有就退回单轮
                q["_sql_rounds"] = trace.get(q["question_id"]) or [q["_llm_sql"]]
                keep.append(q)
        dropped = len(d) - len(keep)
        if dropped:
            print(f"  replay 模式: 丢弃 {dropped} 道无缓存的题, 保留 {len(keep)}", flush=True)
        if len(prov_used) > 1 or (prov_used and "groq" not in prov_used):
            print(f"  provider 分布: {dict(prov_used)}", flush=True)
        if not keep:
            sys.exit("缓存里没有任何可回放的题 —— 先跑 prewarm.py")
        d = keep
    d = d[:args.questions]        # 截断放在筛缓存之后
    if not args.replay_only:
        d = d[:args.questions]
    conn = psycopg2.connect(DSN); conn.autocommit = True
    # 争用旋钮(预注册 prereg_district.json): 默认 77 = 改动前行为, 不影响在跑的采集
    _dr = int(os.environ.get("BIZ_DISTRICT_ROWS", "77"))
    conn.cursor().execute(MEM_DDL.replace("__DISTRICT_ROWS__", str(_dr)))
    return d

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=8)
    p.add_argument("--duration", type=int, default=180, help="固定时间窗模式(秒)")
    p.add_argument("--steps", type=int, default=0,
                   help="固定步数(随机抽样)。已弃用, 请用 --passes")
    p.add_argument("--use-trace", type=int, default=1,
                   help="1=用 retry_trace.json 的多轮返工轨迹(一道题 1-3 步), 0=每题单轮")
    p.add_argument("--trace-file", default="",
                   help="固定用哪份轨迹快照。网格必须传, 否则采集在跑时不同点会读到不同题集")
    p.add_argument("--passes", type=int, default=0,
                   help="固定问题数模式: 完整跑 K 遍题集, 每题恰好 K 次, 确定性顺序。"
                        "主 target 是 seconds_per_pass = 跑完一遍题集要多久")
    p.add_argument("--questions", type=int, default=100)
    p.add_argument("--dbs", default="")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--topk-mem", type=int, default=5)
    p.add_argument("--writes-per-step", type=int, default=1, help="每个 TP 事务里插入几条记忆")
    p.add_argument("--tp-updates", type=int, default=1,
                   help="1=开 agent_state / 记忆访问计数的 UPDATE, 0=只做 INSERT")
    # 0.2 来自规格 §5.1.1 的默认值, 但规格自己标了 "calibration open, Appendix C"
    # (C.1 第二条: "Realistic read/write ratios within cognitive steps")—— 属未校准参数。
    # 2026-10-09 预注册 prereg_drop_bizwrite.json (md5 7480b940): 默认改为 0.0。
    #   注入的业务写被去掉, 理由见预注册: (B) agent 循环本身已产生 TP 写, 不需注入;
    #   (C) 它不是争用主力, 5b 热行才是; (D) 它唯一独有的跨域原子性作用从未被测过
    #   (3405/3405 点全 atomic, 无 deferred 对照); (E) 论文定义的四件事里本就没有它。
    #   代码块保留, 需要时可作为敏感性维度开启。
    p.add_argument("--business-write-pct", type=float, default=0.0,
                   help="在同一事务内做 TPC-B 式业务写入的步骤比例(规格 §5.1.1 默认 20%%, "
                        "规格自标待校准)。争用点是 financial.district 的 77 行")
    # 语义驱动的删除。默认 0.01 的依据: MemTrace(arXiv:2605.28732)脚注 5 分析
    # Mem0 执行日志得出 "delete operations account for only 1.02% of all add, update,
    # and delete operations"。我原先拍的 0.1 高了 10 倍。
    # 注意这只是语义删除;保留期驱动的批量裁剪(规格 §6.6, 每天一次)是另一类,
    # 稳态下其速率 = 插入速率(90 天保留窗口 + 守恒), 尚未实现。
    p.add_argument("--delete-rate", type=float, default=0.01,
                   help="语义驱动的记忆删除比例(C 类: 墓碑 + autovacuum 压力)。"
                        "默认 0.01 来自 Mem0 实测 1.02%%")
    # 默认 0 的三条依据:
    #  ① BIRD 是 text-to-SQL 基准, 本身没有记忆概念 —— 记忆侧全是按规格构造的
    #  ② 我们的记忆是情景型(kind='attempt': "第N次尝试报错X"), 是历史事实,
    #     永不改写。语义型记忆("客户偏好邮件联系")才会被新信息修正 —— 那是 Mem0 的场景
    #  ③ 文献里查不到 UPDATE 的比例(只有 DELETE 的 1.02%), 规格 Appendix C.1 也把
    #     "realistic read/write ratios" 列为待校准
    # 保留它作为敏感度维度 {0, 0.01, 0.1}, 供语义记忆型应用参考, 但不声称某值真实。
    # value linking 索引的维护模式。
    #   atomic  : 向量插入与业务写入同一事务 -> staleness = 0。
    #             embedding 在事务外算好(真实 agent 用等 LLM 的空闲窗口, 不占额外时间)。
    #             独立向量库做不到这一档 —— 跨系统无法原子提交。
    #   deferred: 只写业务表, 索引留给异步补 -> staleness > 0。
    #             这是非 agent 发起的业务写入(人工/批处理)必然落到的一档 —— 没人替它算向量。
    #   off     : 不引入新业务值(旧行为)
    p.add_argument("--value-sync", choices=("atomic","deferred","off"), default="atomic",
                   help="value linking 索引怎么跟上业务写入")
    p.add_argument("--embedding-update-rate", type=float, default=0.0,
                   help="改写一条记忆 embedding 的比例(A 类)。默认 0: 情景记忆不改写。"
                        "无实证依据, 仅作敏感度维度")
    p.add_argument("--ef-search", type=int, default=64)
    p.add_argument("--stmt-timeout-ms", type=int, default=30000)
    p.add_argument("--dynamic-work", type=int, default=1,
                   help="1=动态领题(默认): 所有 agent 从共享票号领下一步, 工作量不变但"
                        "没人空等 —— 实测静态分片下 88%% 的墙钟是跑完的 agent 在等最慢那个。"
                        "0=静态分片 questions[agent_id::agents], agent 的记忆更聚焦但墙钟"
                        "由最重分片决定, 完成时间失去意义。")
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--replay-only", action="store_true",
                   help="只用缓存的 LLM 响应, 不真调 —— 高并发扫描用这个")
    p.add_argument("--out", default="")
    args = p.parse_args()

    print(f"准备 {args.questions} 题 ...", flush=True)
    qs = prepare(args)
    print(f"  实际 {len(qs)} 题, 覆盖 {len(set(q['db_id'] for q in qs))} 个库", flush=True)

    import tempfile, shutil
    resdir = tempfile.mkdtemp(prefix="wl_res_")
    barrier = mp.Barrier(args.agents + 1); stop = mp.Event()
    # 共享票号: 动态领题时所有 agent 从它领下一步。用 mp.Value 而不是 Queue ——
    # Queue 在 8 agents/60s 就因原始延迟列表塞爆管道死锁过(见 worker 末尾注释),
    # 这里只传一个整数, 加锁自增, 没有缓冲区问题。
    _ticket = mp.Value("l", 0) if args.dynamic_work else None
    procs = [mp.Process(target=worker,
                        args=(i, args, qs, barrier, stop, resdir, _ticket))
             for i in range(args.agents)]
    for pr in procs: pr.start()
    try:
        barrier.wait(120)
    except Exception as e:
        for pr in procs: pr.terminate()
        sys.exit(f'barrier 超时: 有 worker 未能启动 ({e})')
    t0 = time.time()
    mode = (f"{args.passes} 遍题集 x {len(qs)} 题(固定问题数)" if args.passes
            else (f"{args.steps} 步(固定步数)" if args.steps else f"{args.duration}s(固定时间)"))
    print(f"开跑: {args.agents} agents x {mode}  ef={args.ef_search} "
          f"writes/step={args.writes_per_step}", flush=True)
    join_to = (args.duration + 120) if not (args.steps or args.passes) else 3600
    for pr in procs: pr.join(join_to)
    el = time.time() - t0

    res = []
    for fn in sorted(os.listdir(resdir)):
        with open(os.path.join(resdir, fn)) as fh: res.append(json.load(fh))
    # 时间去向核算: 已计时的数据库时间 vs agent 墙钟时间。
    # 为什么必须记: 若"未计时"占比很高, 说明瓶颈在客户端(负载生成器)而不是
    # 数据库 —— 那样核数/内存轴会被压平, 网格数据是废的。每个点自报这一项,
    # 事后能直接判定该点是否服务端受限。
    _lat_s = {}
    for r in res:
        for k, v in (r.get("lat") or {}).items():
            _lat_s[k] = _lat_s.get(k, 0.0) + sum(v) / 1000.0
    _agent_wall = sum(r.get("elapsed", 0.0) for r in res)
    # 2026-10-09: tp_sql/tp_vec/tp_commit 是 tp_txn 的子分量, 绝不能进 db_time_s,
    #   否则主 target 被重复计入。db_time 的定义始终是这四项之和。
    _TOP = ("sql", "tp_txn", "vec_mem", "vec_value")
    _SUB = ("tp_biz", "tp_mem", "tp_vlink", "tp_state", "tp_commit")
    _db_s = sum(v for k, v in _lat_s.items() if k in _TOP)
    assert set(_lat_s) <= set(_TOP) | set(_SUB), f"未归类的分量: {set(_lat_s)-set(_TOP)-set(_SUB)}"
    shutil.rmtree(resdir, ignore_errors=True)
    if len(res) != args.agents:
        print(f"  ⚠ 只收到 {len(res)}/{args.agents} 个 agent 的结果")
    agg = {"vec_value": [], "vec_mem": [], "sql": [], "tp_txn": [],
           "tp_biz": [], "tp_mem": [], "tp_vlink": [], "tp_state": [], "tp_commit": []}
    # 2026-10-07: 逐条查询延迟, 跨 agent 合并后每题取中位。
    _byq = {}
    for r in res:
        for qid, v in (r.get("lat_by_q") or {}).items():
            _byq.setdefault(int(qid), []).extend(v)
    steps = sqlerr = llmerr = tperr = 0
    biz = dele = embupd = vsync = vstale = 0
    for r in res:
        steps += r["steps"]; sqlerr += r["sql_err"]; llmerr += r["llm_err"]
        tperr += r.get("tp_err", 0)
        biz += r.get("biz", 0); dele += r.get("del", 0); embupd += r.get("embupd", 0)
        vsync += r.get("vsync", 0); vstale += r.get("vstale", 0)
        for k in agg: agg[k] += r["lat"][k]

    def pct(v, q):
        if not v: return None
        v = sorted(v); return v[min(int(len(v)*q/100), len(v)-1)]

    # A 组 = agent 自身的数据库动作(value 检索 + 记忆检索 + TP 事务)
    _a_lat = agg["vec_value"] + agg["vec_mem"] + agg["tp_txn"]
    _a_n = len(_a_lat)
    # B 组的尾部集中度: 最慢 1% 占 AP 总时间多少
    _sq = sorted(agg["sql"], reverse=True)
    _ap_tail = None
    if _sq:
        _k = max(1, len(_sq)//100)
        _tot = sum(_sq)
        _ap_tail = round(100.0*sum(_sq[:_k])/_tot, 2) if _tot else None
    out = {"agents": args.agents, "duration_s": round(el,1), "ef_search": args.ef_search,
           "mode": ("fixed_passes" if args.passes else
                    ("fixed_steps" if args.steps else "fixed_time")),
           "passes": args.passes or None,
           "questions_per_pass": len(qs),
           "use_trace": args.use_trace,
           "dynamic_work": args.dynamic_work,
           # 时间去向: db_time_pct 低 = 客户端受限, 该点的资源轴不可信
           "agent_wall_s": round(_agent_wall, 2),
           "db_time_s": round(_db_s, 2),
           "db_time_pct": round(100.0 * _db_s / max(_agent_wall, 1e-9), 2),
           "db_time_breakdown_s": {k: round(v, 2) for k, v in sorted(
               ((k, v) for k, v in _lat_s.items() if k in _TOP), key=lambda x: -x[1])},
           # tp_txn 的内部拆分(不计入 db_time_s), 按来源分:
           #   tp_biz   业务事务(TPC-B 形态) —— BIRD 全是 AP, 这是为此补上的 TP
           #   tp_mem   记忆循环: 5b 检索后强化 / 5d 修剪 / 5e 信念修正 /
           #            5f 把 agent<->LLM 这一轮的结果写回成记忆(每步必做)
           #   tp_vlink 业务写的新文本值进 value-linking 向量索引(跨域原子性)
           #   tp_state agent 步数簿记
           #   tp_commit COMMIT 本身(含 WAL flush)
           # 五者之和略小于 tp_txn, 差额(tp_txn_residual_s)是 Python 侧与 BEGIN 的开销。
           "tp_txn_breakdown_s": {k: round(_lat_s.get(k, 0.0), 4) for k in _SUB},
           "tp_txn_residual_s": round(_lat_s.get("tp_txn", 0.0)
                                      - sum(_lat_s.get(k, 0.0) for k in _SUB), 4),
           "trace_file": TRACE_INFO["file"],
           "trace_n_questions": TRACE_INFO["n_questions"],
           "trace_questions": sum(1 for q in qs if len(q.get("_sql_rounds") or [])>1),
           "avg_rounds_per_question": round(
               sum(len(q.get("_sql_rounds") or [1]) for q in qs)/max(len(qs),1), 3),
           "steps_target": args.steps or None,
           # 主 target: db_time_s —— 这份固定工作量在数据库里花掉的总服务时间。
           # 为什么不是完成时间: my_qs = questions[agent_id::agents] 是静态分片,
           # 而 1,521 道题的开销差 4 个数量级(p50 1.6ms, max 20.8s)。于是各 agent
           # 的分片轻重悬殊, 完成时间由分到最重分片的那个 agent 决定 —— 那是分片
           # 运气, 不是资源配置。实测 2 核 -> 18 核:
           #   完成时间        522.4s -> 493.1s   1.06x   <- 被分片运气支配, 废指标
           #   agent 干活时间  5439s  -> 1920s    2.83x
           #   db_time_s      4356s  -> 1435s    3.04x   <- 干净
           #   A 组 p50        13.4ms -> 4.3ms    3.10x
           # 18 核那点 agent 只干了 1920s 活, 却要等到 493x32=15779s 才收工 ——
           # 88% 的墙钟是跑完的 agent 在空等。db_time_s 与空等、与调度都无关。
           #
           # completion_seconds / seconds_per_pass 保留作参考, 但不能当主 target:
           # 实测 1,502 条 LLM 生成的 SQL 里有两条是笛卡尔积(21.2s 和 18.1s),
           # 最慢 0.5% 的查询占 AP 总时间的 78.7%。32 个 agent 里被这两条卡住的 2 个
           # 决定墙钟, 于是端到端时间在 2 核 -> 18 核只快 1.06x, 而分项各自快
           # 1.63x ~ 8.20x。混在一起的指标被少数离群查询绑架, 资源轴就被压平了。
           "completion_seconds": round(el, 3) if (args.steps or args.passes) else None,
           "seconds_per_pass": round(el/args.passes, 4) if args.passes else None,
           # ── 分两组报: A 组 agent 自身的数据库动作, B 组 LLM 生成的 AP 查询 ──
           # A 组每步必做、量大而均匀, 是资源 -> 性能的主 target;
           # B 组重尾极端, 单独报, 它的尾部占比本身就是结论。
           "agent_ops": _a_n,
           "agent_ops_per_second": round(_a_n/el, 2),
           "agent_p50_ms": round(pct(_a_lat, 50), 3) if _a_lat else None,
           "agent_p95_ms": round(pct(_a_lat, 95), 3) if _a_lat else None,
           "agent_p99_ms": round(pct(_a_lat, 99), 3) if _a_lat else None,
           "agent_seconds_per_pass": round(sum(_a_lat)/1000.0/args.agents/args.passes, 4)
                                     if (_a_lat and args.passes) else None,
           # 每题的实测 SQL 延迟中位(毫秒), 与离线探针的逐条代价一一对应。
           # 用于算与文献同口径的单查询 Q-error。
           "sql_ms_by_qid": {str(k): round(pct(v, 50), 3)
                             for k, v in sorted(_byq.items()) if v},
           "sql_n_by_qid": {str(k): len(v) for k, v in sorted(_byq.items()) if v},
           "ap_queries": len(agg["sql"]),
           "ap_seconds_per_pass": round(sum(agg["sql"])/1000.0/args.agents/args.passes, 4)
                                  if (agg["sql"] and args.passes) else None,
           # 最慢 1% 的 AP 查询占 AP 总时间的比例 —— 越高说明分析开销越被少数查询主导
           "ap_tail_share_pct": _ap_tail,
           "writes_per_step": args.writes_per_step, "topk": args.topk,
           "questions": len(qs), "replay_only": args.replay_only,
           "mem_jitter": float(os.environ.get("MEM_JITTER", "0.05")),
           "steps": steps, "steps_per_second": round(steps/el, 2),
           "sql_errors": sqlerr, "llm_errors": llmerr, "tp_errors": tperr,
           "tp_updates": args.tp_updates,
           # 三类 TP 写操作的实际发生次数(按代价特征分类, 见  / 写代价实测)
           "business_writes": biz,          "business_write_pct": args.business_write_pct,
           "memory_deletes": dele,          "delete_rate": args.delete_rate,
           "embedding_updates": embupd,     "embedding_update_rate": args.embedding_update_rate,
           # value linking 索引的维护: atomic = 与业务写入同事务(staleness 0);
           # deferred = 业务值已提交但索引里还没有(staleness > 0)
           "value_sync_mode": args.value_sync,
           "value_syncs": vsync, "value_stale": vstale}
    for k, label in [("vec_value","vector_search"), ("vec_mem","memory_search"),
                     ("sql","sql_exec"), ("tp_txn","tp_txn"),
                     ("tp_biz","tp_biz"), ("tp_mem","tp_mem"), ("tp_vlink","tp_vlink"),
                     ("tp_state","tp_state"), ("tp_commit","tp_commit")]:
        out[f"{label}_n"] = len(agg[k])
        for q in (50, 95, 99):
            out[f"{label}_p{q}_ms"] = round(pct(agg[k], q), 3) if agg[k] else None
        out[f"{label}_per_second"] = round(len(agg[k])/el, 2)
    # TP:AP 实际配比 = 写入次数 : 分析型查询次数
    ap = out["sql_exec_n"]; tp = out["tp_txn_n"]
    out["tp_ap_ratio"] = round(tp/max(ap,1), 3)
    # 每个 TP 事务里的语句数: state UPDATE + (记忆 UPDATE + db_stats upsert) + N 个 INSERT
    # 每个 TP 事务的语句数: state UPDATE + 记忆访问计数 + (业务写入 4 条) + 修剪 + 改写 + N 条 INSERT
    out["tp_stmts_per_txn"] = round(
        1 + (1 if args.tp_updates else 0) + 4*args.business_write_pct
        + args.delete_rate + args.embedding_update_rate + args.writes_per_step, 2)
    out["business_writes_per_second"] = round(biz/el, 2)

    print(f"\n{'='*70}")
    for k, v in out.items(): print(f"  {k:26s} {v}")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"\n已写 {args.out}")

if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
