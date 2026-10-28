"""LLM 适配层 + 响应缓存。

为什么要缓存:核数扫描要在 4/8/16/24/32 核下跑同一批题。如果每遍都真调 LLM,
模型的随机性和网络抖动会盖过核数效应(核数只影响端到端时间的 5% 以下)。
缓存让第 2 遍起 prompt 完全一致时直接回放,既省钱又把 LLM 变成常量。

关键副产物:检索结果变了 -> prompt 变了 -> 缓存 miss。所以 miss 率本身就是
"资源不足伤害了检索质量"的观测量,直接记进指标。
"""
from __future__ import annotations
import hashlib, json, os, re, sqlite3, sys, threading, time
from dataclasses import dataclass

PROVIDERS = {
    # name:        (base_url,                                    env var,            默认模型)
    "anthropic":   ("https://api.anthropic.com/v1/messages",      "ANTHROPIC_API_KEY", "claude-opus-5"),
    "gemini":      ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                                                                  "GEMINI_API_KEY",    "gemini-2.0-flash"),
    "groq":        ("https://api.groq.com/openai/v1/chat/completions",
                                                                  "GROQ_API_KEY",      "llama-3.3-70b-versatile"),
    "cerebras":    ("https://api.cerebras.ai/v1/chat/completions", "CEREBRAS_API_KEY",  "llama-3.3-70b"),
    "openrouter":  ("https://openrouter.ai/api/v1/chat/completions",
                                                                  "OPENROUTER_API_KEY","meta-llama/llama-3.3-70b-instruct:free"),
    "together":    ("https://api.together.xyz/v1/chat/completions","TOGETHER_API_KEY",
                                                                  "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
}


class QuotaExhausted(RuntimeError):
    """撞的是日配额(TPD/RPD), 不是分钟配额。

    区别很关键: 分钟桶睡几十秒就回来, 日桶不会 —— Groq 实测 TPD=200 000,
    满了以后 retry-after 只给 1m46s(滚动窗口刚好挤出一个请求的空间),
    照睡的话吞吐就被钉在"每 2 分钟一个请求"。有多个 provider 时,
    立刻换一个才对; 全都满了才由外层等。
    """


@dataclass
class LLMResult:
    text: str
    cached: bool
    latency_ms: float
    in_tokens: int = 0
    out_tokens: int = 0
    provider: str = ""      # 多 provider 采集时记录是谁答的(论文里要说明异质性)
    model: str = ""


class ResponseCache:
    """prompt 的 sha256 -> 响应。SQLite,进程间安全。"""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        c = sqlite3.connect(path, timeout=30)
        c.execute("""CREATE TABLE IF NOT EXISTS cache(
                       key TEXT PRIMARY KEY, provider TEXT, model TEXT,
                       prompt TEXT, response TEXT,
                       in_tokens INT, out_tokens INT, created REAL)""")
        c.execute("PRAGMA journal_mode=WAL")
        c.commit(); c.close()

    @staticmethod
    def key(provider: str, model: str, system: str, prompt: str) -> str:
        h = hashlib.sha256()
        for part in (provider, model, system, prompt):
            h.update(part.encode()); h.update(b"\x00")
        return h.hexdigest()

    def get(self, k: str):
        c = sqlite3.connect(self.path, timeout=30)
        r = c.execute("SELECT response,in_tokens,out_tokens FROM cache WHERE key=?", (k,)).fetchone()
        c.close()
        return r

    def put(self, k, provider, model, system, prompt, response, in_tok, out_tok):
        with self._lock:
            c = sqlite3.connect(self.path, timeout=30)
            c.execute("INSERT OR REPLACE INTO cache VALUES(?,?,?,?,?,?,?,?)",
                      (k, provider, model, system + "\n---\n" + prompt, response,
                       in_tok, out_tok, time.time()))
            c.commit(); c.close()

    def stats(self):
        c = sqlite3.connect(self.path, timeout=30)
        n, i, o = c.execute("SELECT count(*),coalesce(sum(in_tokens),0),"
                            "coalesce(sum(out_tokens),0) FROM cache").fetchone()
        c.close()
        return {"entries": n, "in_tokens": i, "out_tokens": o}


class LLM:
    """统一入口。所有 provider 走 OpenAI 兼容的 chat/completions,除了 anthropic 走原生。"""

    def __init__(self, provider: str, model: str | None = None,
                 cache_path: str = "llm_cache.sqlite",
                 max_tokens: int = 512, temperature: float = 0.0,
                 replay_only: bool = False, tpm: int = 0):
        if provider not in PROVIDERS:
            raise ValueError(f"未知 provider {provider};可选 {list(PROVIDERS)}")
        self.provider = provider
        self.url, env, default_model = PROVIDERS[provider]
        self.model = model or default_model
        self.key = os.environ.get(env)
        self.max_tokens, self.temperature = max_tokens, temperature
        self.replay_only = replay_only          # True = 只读缓存,miss 就报错(用于核数扫描)
        # 客户端节流。关键: TPM 配额按 max_tokens 预留量 扣, 不是按实际输出量。
        # 实测 max_tokens=1500 时每次调用被扣 prompt(700)+预留(1500)=2200,
        # TPM 8000 只允许 3.6 次/分钟; 而按实际用量(1100)算只等 8.25 秒 = 7.3 次/分钟
        # -> 持续超额 -> retry-after 67 秒 -> 单次调用要烧 15 分钟。
        self.tpm = tpm
        self._last_call = 0.0
        self._last_reserved = 0
        self.throttle_s = 0.0        # 累计被限流等待(秒), 用于区分等配额 vs 干活
        self._call_throttle = 0.0    # 本次调用内的累计等待
        self.pace_safety = 1.15
        self.cache = ResponseCache(cache_path)
        self.n_hit = self.n_miss = 0
        if not self.key and not replay_only:
            raise RuntimeError(f"环境变量 {env} 未设置")

    def complete(self, system: str, prompt: str) -> LLMResult:
        k = ResponseCache.key(self.provider, self.model, system, prompt)
        hit = self.cache.get(k)
        if hit:
            self.n_hit += 1
            return LLMResult(hit[0], True, 0.0, hit[1], hit[2])
        self.n_miss += 1
        if self.replay_only:
            raise KeyError(f"replay_only 模式下缓存未命中 (key={k[:12]}…) "
                           f"—— 说明检索结果变了,prompt 与首遍不同")
        # 预留量 = prompt 估算 + max_tokens(服务端就是按这个扣配额的)
        self._last_reserved = (len(system) + len(prompt)) // 4 + self.max_tokens
        self._pace()
        t0 = time.perf_counter()
        text, in_tok, out_tok = self._call(system, prompt)
        dt = (time.perf_counter() - t0) * 1000
        self.cache.put(k, self.provider, self.model, system, prompt, text, in_tok, out_tok)
        return LLMResult(text, False, dt, in_tok, out_tok)

    def _pace(self):
        """按本次将预留的 token 量算该等多久, 保证不超 TPM。"""
        if not self.tpm:
            self._last_call = time.time(); return
        need = self._last_reserved / self.tpm * 60.0 * self.pace_safety
        wait = need - (time.time() - self._last_call)
        if wait > 0: time.sleep(wait)
        self._last_call = time.time()

    def _call(self, system: str, prompt: str):
        import urllib.request, urllib.error
        if self.provider == "anthropic":
            body = {"model": self.model, "max_tokens": self.max_tokens,
                    "system": system, "messages": [{"role": "user", "content": prompt}]}
            headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01",
                       "content-type": "application/json"}
        else:
            body = {"model": self.model, "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": prompt}]}
            headers = {"Authorization": f"Bearer {self.key}",
                       "content-type": "application/json"}
        # Cloudflare 拦默认的 Python-urllib UA(403 code 1010),必须显式设置
        headers["User-Agent"] = "bird-agent/0.1"
        headers["Accept"] = "application/json"
        req = urllib.request.Request(self.url, json.dumps(body).encode(), headers)
        last = None
        self._call_throttle = 0.0
        for attempt in range(10):                        # 免费档限流严,必须退避重试
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    d = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                last = e
                # 必须把 body 读掉并关闭, 否则连接停在 CLOSE-WAIT ——
                # 实测泄漏 6 个 socket, 13.7 小时只推进 80 题。
                try: body = e.read()[:300].decode(errors="replace")
                except Exception: body = ""
                finally:
                    try: e.close()
                    except Exception: pass
                if e.code in (429, 500, 502, 503, 529):
                    low = body.lower()
                    # "Request too large / reduce max_tokens" 是参数问题, 不是配额:
                    # 退避重试多少次都一样。直接抛, 让上层看到并修配置。
                    if "reduce max_tokens" in low or "request too large" in low:
                        sys.stderr.write(f"[llm] {self.provider}/{self.model} "
                                         f"max_tokens={self.max_tokens} 超模型上限, "
                                         f"需在 MAX_TOKENS_OVERRIDE 里下调: {body[:180]}\n")
                        sys.stderr.flush()
                        raise RuntimeError(f"max_tokens 超限: {body[:200]}") from e
                    # 只认明确的措辞。裸的 "tpd"/"rpd" 会误命中随机 id
                    # (org_01m1ba0fw0e4ct83ma1vgpz3jh 这种), 把分钟级 429
                    # 误判成日配额 -> 白白冷却这个 provider 30 分钟。
                    # Gemini 的 429 文案里没有 "per day" 字样, 只有
                    # "You exceeded your current quota" + QuotaFailure 里的
                    # quotaId=GenerateRequestsPerDayPerProjectPerModel-FreeTier。
                    # 漏判的后果: 每道题白做 10 次指数退避(约 6 分钟), 实测限流等待
                    # 占了 96% 的时间, 25.7 小时只成功 70 次调用。
                    if e.code == 429 and any(m in low for m in
                            ("tokens per day", "requests per day", "(tpd)", "(rpd)",
                             "per-day", "daily limit", "daily quota",
                             "exceeded your current quota", "resource_exhausted",
                             "generaterequestsperday", "quotavalue")):
                        lim = {k[len("x-ratelimit-"):]: v for k, v in (e.headers or {}).items()
                               if k.lower().startswith("x-ratelimit-")}
                        sys.stderr.write(f"[llm] {self.provider} 日配额耗尽, 不睡直接抛 "
                                         f"limits={lim} body={body[:200]}\n")
                        sys.stderr.flush()
                        raise QuotaExhausted(f"{self.provider} 日配额: {body[:200]}") from e
                    ra = e.headers.get("retry-after") if e.headers else None
                    if not ra:
                        # Google 不发 retry-after 头, 而是把秒数放在 body 里:
                        #   "retryDelay": "53s"  /  "Please retry in 53.502061294s."
                        m_ = (re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s?"', body)
                              or re.search(r'retry in (\d+(?:\.\d+)?)s', body))
                        if m_: ra = m_.group(1)
                    try: delay = float(ra) if ra else min(2 ** attempt, 60)
                    except ValueError: delay = min(2 ** attempt, 60)
                    delay = max(delay, 1.0) + 0.5
                    # 可见性: 之前限流是静默 sleep, 分不清"在等配额"和"在干活"
                    lim = {k[len("x-ratelimit-"):]: v for k, v in (e.headers or {}).items()
                           if k.lower().startswith("x-ratelimit-")}
                    # 单次 sleep 封顶 330s: 实测 Groq 在分钟桶还满(remaining-tokens 8000)
                    # 时仍会以 retry_after=279s 拒请求 —— 那是容量/日配额桶, 头里不报。
                    # 封得比服务端给的值小(之前 120s)只会白烧一次请求再被拒,
                    # 所以按服务端说的等; 只挡住"距 UTC 零点"那种小时级数值。
                    slept = min(delay, 330.0)
                    sys.stderr.write(
                        f"[llm] {e.code} attempt={attempt} retry_after={ra} "
                        f"sleep={slept:.0f}s(服务端要 {delay:.0f}s) "
                        f"limits={lim} body={body[:300]}\n")
                    sys.stderr.flush()
                    self.throttle_s += slept
                    self._call_throttle += slept
                    time.sleep(slept)
                    # 单次调用累计等超过 15 分钟 = 撞的是日配额, 不是分钟配额。
                    # 抛出去让 daily_collect.sh 的"等 30 分钟"接管, 缓存不丢。
                    # 累计阈值 1800s: 单次 sleep 已放到 330s, 900s 会在第 3 次退避就误判
                    if self._call_throttle > 1800:
                        raise RuntimeError(
                            f"QUOTA_EXHAUSTED: 单次调用已等 {self._call_throttle:.0f}s, "
                            f"retry_after={ra}, limits={lim}") from e
                    continue
                raise RuntimeError(f"{e.code} {body}") from e
            except Exception as e:
                last = e; time.sleep(min(2 ** attempt, 60))
        else:
            raise RuntimeError(f"10 次重试后仍失败: {last}")

        if self.provider == "anthropic":
            text = "".join(b.get("text", "") for b in d.get("content", []))
            u = d.get("usage", {})
            return text, u.get("input_tokens", 0), u.get("output_tokens", 0)
        text = d["choices"][0]["message"]["content"] or ""
        u = d.get("usage", {})
        return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

    def hit_rate(self):
        t = self.n_hit + self.n_miss
        return self.n_hit / t if t else 0.0


# 各免费档的默认模型与 TPM。TPD 是硬墙(Groq 实测 200 000/天), 撞墙后 sleep 无意义
# —— 只有换 provider 有用。所以 LLMPool 在 TPD 429 上不睡, 直接轮转。
# 第三项 = 每日可用调用数估计, 用作加权分配的权重。
# groq: 实测 TPD 200 000 / 每次约 1 212 token = 165 次/天(这是量出来的, 不是查的)
# gemini: 免费档公开资料约 1 500 RPD, 但 Google 砍过额度且按地区浮动 ——
#         实际值以 AI Studio 控制台为准; 这里只用于分配权重, 估偏不影响正确性,
#         只影响两边谁先跑完(归属一旦写入清单就不再改)。
FREE_TIER = {
    # "provider:model": 每日可用调用数(实测, 不是查资料)
    #
    # Groq 的 TPD 是 200 000 token/天, 且报错文案是 "for model X" -> 按模型算。
    # 每日调用数 = 200000 / 每次实测 token 数:
    #   gpt-oss-120b   in 825 + out 351 = 1176  -> 170
    #   gpt-oss-20b    in 399 + out 349 =  748  -> 267
    #   qwen3.8-27b    in 369 + out  64 =  433  -> 462   (最省; 不吐推理 token)
    #   compound-mini  in1087 + out 326 = 1413  -> 142   (in 高: 它自带 system 脚手架)
    # Gemini 是按请求限, 实测 quotaValue=20/天/模型(不是 1500 —— 那是旧的
    #   gemini-2.5-flash 的公开数字, 照搬吃了 75 倍的亏)。
    #
    # 被排除的(实测):
    #   qwen/qwen3.6-27b  吐 <think> 推理, out 撞 1500 上限被截断
    #   allam-2-7b        标识符不加引号 -> PG 必然报错, 产生不了真实查询开销
    "groq:openai/gpt-oss-120b":  170,
    "groq:openai/gpt-oss-20b":   267,
    "groq:qwen/qwen3.8-27b":     462,
    # "groq:groq/compound-mini": 142,   # 2026-10-03 实测 404 下线(第二次了)
    "gemini:gemini-3.6-flash":    20,
    "gemini:gemini-3.8-flash":    20,
    "gemini:gemini-3.7-flash":    20,
    "gemini:gemini-3.5-flash":    20,
    "gemini:gemini-3.5-flash-lite": 20,
    "gemini:gemini-3.1-flash-lite": 20,
}

# 每个 provider 的 TPM(分钟级节流); 0 = 不做客户端节流
PROVIDER_TPM = {"groq": 8000, "gemini": 0, "cerebras": 0,
                "openrouter": 0, "together": 0}

# 个别模型的 max_tokens 上限覆盖。
# 实测 qwen/qwen3.8-27b 有 OTPM(每分钟输出 token)上限 1000, 传 1500 会被
# 整个请求拒掉("Request too large ... reduce max_tokens"), 一次都过不去 ——
# 而它是权重最大的一项(462/天), 不修则 402 道题永远采不到。
# 900 远高于它实测的输出量(64 token), 不会截断。
MAX_TOKENS_OVERRIDE = {
    "groq:qwen/qwen3.8-27b": 900,
}



class LLMPool:
    """多 provider 轮转采集。

    为什么需要: 单个免费档的日 token 配额(TPD)是硬墙。实测 Groq
    TPD=200 000 -> 每题约 2 109 tokens(1 212 x 1.74 轮)-> 每天 95 题
    -> BIRD dev 全量 1534 题要 16.2 天。撞 TPD 后 retry-after 给的是分钟级
    数值但配额直到 UTC 零点才回, 睡着等纯属浪费; 换一个 provider 才有用。

    缓存键含 provider+model, 所以同一道题在不同 provider 下是不同条目。
    采集阶段只要有一个答案就够 -> complete 先扫全部 client 的缓存,
    全 miss 才真调一个未冷却的 client。
    """

    TPD_COOLDOWN = 1800.0      # 撞 TPD 后冷却 30 分钟再探(避免做时区换算)

    def __init__(self, providers, cache_path="llm_cache.sqlite",
                 max_tokens=512, temperature=0.0, models=None):
        """providers 可以是 "groq"(用 FREE_TIER 里该家的全部模型)
        或 "groq:openai/gpt-oss-20b"(只用这一个模型)。

        每个 (provider, model) 是池里独立一项, 各自的额度、冷却、权重都独立 ——
        因为两家的日配额都是按模型算的(Groq 文案 "for model X";
        Gemini quotaId=GenerateRequestsPerDayPerProjectPerModel)。
        """
        self.clients, self.skipped, self.weights, self.keys = [], [], [], []
        want = []
        for spec in providers:
            if ":" in spec:
                want.append(spec)
            else:
                got = [k for k in FREE_TIER if k.split(":", 1)[0] == spec]
                want += got or [f"{spec}:{FREE_TIER.get(spec, ('',))[0]}"]
        for spec in want:
            prov, _, mdl = spec.partition(":")
            if prov not in PROVIDERS:
                self.skipped.append((spec, "未知 provider")); continue
            env = PROVIDERS[prov][1]
            if not os.environ.get(env):
                self.skipped.append((spec, f"{env} 未设置")); continue
            self.clients.append(LLM(prov, mdl or None, cache_path=cache_path,
                                    max_tokens=MAX_TOKENS_OVERRIDE.get(spec, max_tokens),
                                    temperature=temperature,
                                    tpm=PROVIDER_TPM.get(prov, 0)))
            self.keys.append(spec)
            self.weights.append(max(1, int(FREE_TIER.get(spec, 100))))
        if not self.clients:
            raise RuntimeError(f"没有可用模型(全部缺 key): {self.skipped}")
        self.cool_until = [0.0] * len(self.clients)
        self.n_call = [0] * len(self.clients)
        self.rr = 0
        # 配额耗尽后置 True: 只回放缓存, 不再发请求。
        # 为什么需要: 撞配额时直接中止整轮会把后面已经缓存的题全丢掉
        # (实测只收到 6 题就停, 而缓存里有 315 题可用)。
        self.cache_only = False

    @property
    def throttle_s(self):
        return sum(c.throttle_s for c in self.clients)

    @property
    def max_tokens(self):
        return self.clients[0].max_tokens

    def all_cooling(self):
        """全部池项都在冷却吗。按模型独立配额后, 单项撞限不代表池子没额度 ——
        采集侧只能在这个返回 True 时才转"只收缓存"。"""
        now = time.time()
        return all(t > now for t in self.cool_until)

    def status(self):
        now = time.time()
        return ", ".join(
            f"{k}:{self.n_call[i]}次"
            + (f"(冷却{(self.cool_until[i]-now)/60:.0f}分)" if self.cool_until[i] > now else "")
            for i, k in enumerate(self.keys))

    def names(self):
        return list(self.keys)

    def pick_for(self, key) -> str:
        """按 key 确定性地选 provider —— 不是轮转。

        为什么必须确定性: 论文要能说"哪个模型做了哪些题"。轮转的话谁答哪道题
        取决于当时的配额状态和到达顺序, 换一次运行就变了, 不可复现。
        用 sha1(key) 而不是 hash: Python 的 str hash 每进程随机化(PYTHONHASHSEED),
        同一道题在两次运行里会分到不同 provider。
        """
        # 按每日额度加权: 快的多分题, 否则总时长被最慢的那家拖住。
        # 实测 groq 165 次/天 vs gemini 约 1 500 次/天, 均分会让 groq 那半边慢 9 倍。
        # 仍然是确定性的 —— 同一 qid 永远落在同一家, 与配额当时状态无关。
        tot = sum(self.weights)
        h = hashlib.sha1(str(key).encode()).digest()
        x = int.from_bytes(h[:8], "big") % tot
        acc = 0
        for k, w in zip(self.keys, self.weights):
            acc += w
            if x < acc: return k
        return self.keys[-1]

    def weight_table(self):
        tot = sum(self.weights)
        return {k: {"每日调用估计": w, "分配占比": round(w / tot, 4)}
                for k, w in zip(self.keys, self.weights)}

    def complete(self, system: str, prompt: str, pin: str | None = None) -> LLMResult:
        """pin 给定时只用这个 provider —— 缓存也只查它的键。

        为什么缓存也要钉住: 一道题的所有返工轮必须出自同一个模型, 否则重试序列
        里模型换了人, 这条轨迹就不代表任何真实 agent 的行为。
        """
        if pin is not None:
            idx = next((i for i, k in enumerate(self.keys) if k == pin), None)
            if idx is None:
                raise RuntimeError(f"归属 provider {pin} 不在池里(缺 key?): {self.names()}")
            c = self.clients[idx]
            hit = c.cache.get(ResponseCache.key(c.provider, c.model, system, prompt))
            if hit:
                c.n_hit += 1
                r = LLMResult(hit[0], True, 0.0, hit[1], hit[2])
                r.provider, r.model = c.provider, c.model
                return r
            if self.cache_only or self.cool_until[idx] > time.time():
                raise QuotaExhausted(
                    f"{pin} 配额耗尽/冷却中, **不换人顶上** —— 这道题留到它恢复再采")
            try:
                r = c.complete(system, prompt)
                self.n_call[idx] += 1
                r.provider, r.model = c.provider, c.model
                return r
            except QuotaExhausted:
                self.cool_until[idx] = time.time() + self.TPD_COOLDOWN
                sys.stderr.write(f"[pool] {pin} 撞日配额, 冷却 "
                                 f"{self.TPD_COOLDOWN/60:.0f} 分钟 [{self.status()}]\n")
                sys.stderr.flush()
                raise
        # 1) 无 pin: 先扫全部 client 的缓存 —— 哪个 provider 答过都算答过
        for c in self.clients:
            hit = c.cache.get(ResponseCache.key(c.provider, c.model, system, prompt))
            if hit:
                c.n_hit += 1
                r = LLMResult(hit[0], True, 0.0, hit[1], hit[2])
                r.provider, r.model = c.provider, c.model
                return r
        # 2) 全 miss
        if self.cache_only:
            raise KeyError("cache_only: 配额已耗尽, 只收缓存")
        now = time.time()
        order = [(self.rr + i) % len(self.clients) for i in range(len(self.clients))]
        live = [i for i in order if self.cool_until[i] <= now]
        if not live:
            wait = min(self.cool_until) - now
            raise RuntimeError(f"QUOTA_EXHAUSTED: 全部 {len(self.clients)} 个 provider 都在冷却, "
                               f"最早 {wait/60:.0f} 分钟后可用 [{self.status()}]")
        last = None
        for i in live:
            c = self.clients[i]
            try:
                r = c.complete(system, prompt)
                self.n_call[i] += 1
                self.rr = (i + 1) % len(self.clients)
                r.provider, r.model = c.provider, c.model
                return r
            except RuntimeError as e:
                last = e
                # 日配额: 冷却这个 provider, 立刻换下一个(不睡)
                if isinstance(e, QuotaExhausted) or "QUOTA_EXHAUSTED" in str(e):
                    self.cool_until[i] = time.time() + self.TPD_COOLDOWN
                    sys.stderr.write(f"[pool] {c.provider} 撞日配额, 冷却 "
                                     f"{self.TPD_COOLDOWN/60:.0f} 分钟, 换下一个 "
                                     f"[{self.status()}]\n")
                    sys.stderr.flush()
                    continue
                raise
        raise RuntimeError(f"QUOTA_EXHAUSTED: 全部 provider 均失败, 最后: {last}")
