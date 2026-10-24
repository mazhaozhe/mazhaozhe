"""
agent/token_tracker.py —— Token 用量统计与预估
================================================
两个核心功能（对标 WorkBuddy 的 token 显示）：

1. 事前预估  estimate_run()
   在 Agent 开跑前，基于【系统提示 + 工具定义 + 当前对话历史 + 本次输入】
   估算下一轮的输入 token；输出 token 按本会话历史输出均值预估（首轮用默认值）。
   再按当前模型的价格折算预计成本。

2. 事后统计  actual_usage()
   一轮结束后，从 checkpointer 状态里 diff 出本轮【新增的 AI 消息】，
   汇总 API 真实返回的 usage_metadata（输入/输出 token，含 ReAct 中间步），
   折算实际成本。真实值与预估值的误差即预估精度。

估算原理（不依赖任何 tokenizer 库的启发式）：
    DeepSeek 官方口径 ≈ 1 个英文字符 0.3 token、1 个中文字符 0.6 token。
    对纯"预估"用途足够（误差一般在 ±20% 内）；
    真实用量以 API 返回的 usage_metadata 为准（见事后统计）。

成本口径：
    pricing 字段在 config/models.json 中按模型配置（元 / 百万 token），
    输入价按"缓存未命中"计（偏保守，即偏高报），可随时在 JSON 里修改。
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

from agent.observability import log_step

# ---------------------------------------------------------------------------
# 常量：估算系数（DeepSeek 官方口径的近似）
# ---------------------------------------------------------------------------
ASCII_TOKENS_PER_CHAR = 0.3      # 1 个英文字符 ≈ 0.3 token
CJK_TOKENS_PER_CHAR = 0.6        # 1 个中文字符 ≈ 0.6 token
PER_MESSAGE_OVERHEAD = 8         # 每条消息的协议开销（角色标记等）
DEFAULT_OUTPUT_ESTIMATE = 256    # 首轮没有历史可参照时，预估输出 token
REASONER_OUTPUT_FACTOR = 2.5     # 推理模型的思考链也是输出 token，按经验约 2.5 倍
EXTRA_CALL_INPUT_FACTOR = 1.05    # 工具任务里第 2、3... 次模型调用，每次比首次多 ~5%
                                 # （历史+静态要重发，只新增少量工具调用参数与返回结果；
                                 #  经 DeepSeek 真实调用标定）

# 静态开销标定系数：系统提示与工具定义实际发送时，外层还有
# LangChain 注入的框架指令 + OpenAI function 格式包装（{"type":"function",...}），
# 经 DeepSeek 真实调用标定约为启发式估算的 2 倍（2,878 实测 / 1,445 估算）。
STATIC_CALIBRATION = 2.0


# ---------------------------------------------------------------------------
# 纯函数：文本 / 消息 → token 估算
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """启发式估算一段文本的 token 数（中英文混合）。空文本返回 0。"""
    if not text:
        return 0
    ascii_len = sum(1 for ch in text if ord(ch) < 128)
    other_len = len(text) - ascii_len
    return int(ascii_len * ASCII_TOKENS_PER_CHAR + other_len * CJK_TOKENS_PER_CHAR)


def message_text(msg: Any) -> str:
    """
    把一条 LangChain 消息压成"可用于估算的纯文本"：
    - 文本内容直接取；
    - 内容块列表（图片/多段）取其中文本部分；
    - AI 消息的 tool_calls（函数名+参数）也算进去——它们真实占 token。
    """
    parts: List[str] = []
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    for tc in getattr(msg, "tool_calls", None) or []:
        try:
            parts.append(json.dumps(tc, ensure_ascii=False))
        except (TypeError, ValueError):
            parts.append(str(tc))
    return "\n".join(p for p in parts if p)


def messages_tokens(messages: Iterable[Any]) -> int:
    """估算一批消息的 token 总量（含每条消息的协议开销）。只算消息本身，
    不含系统提示与工具定义（那是每次请求的固定开销，见 static_overhead_tokens）。"""
    total = 0
    for msg in messages or []:
        total += estimate_tokens(message_text(msg)) + PER_MESSAGE_OVERHEAD
    return total


# ---------------------------------------------------------------------------
# 工具定义的 token 占用（静态，进程内只算一次）
# ---------------------------------------------------------------------------
_TOOLS_SCHEMA_CACHE: Optional[str] = None


def _tool_schema_json(tool: Any) -> str:
    """把一个工具的参数 Schema 序列化成 JSON 文本（与发给模型的格式一致）。"""
    try:
        schema = tool.args_schema.model_json_schema()   # Pydantic v2
    except Exception:
        try:
            schema = tool.args or {}                    # 兜底：参数字典
        except Exception:
            schema = {}
    try:
        return json.dumps(schema, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(schema)


def _tools_schema_text() -> str:
    """
    把所有已注册工具的"名字+描述+参数 Schema"序列化成文本。
    这部分会随每次模型请求一起发送，是输入 token 的大头之一。
    注意：必须用 model_json_schema() 取真实 Schema——
    直接 json.dumps(pydantic 类) 只会得到类名字符串，严重低估。
    """
    global _TOOLS_SCHEMA_CACHE
    if _TOOLS_SCHEMA_CACHE is not None:
        return _TOOLS_SCHEMA_CACHE
    try:
        from tools.registry import all_tools
        chunks: List[str] = []
        for t in all_tools():
            chunks.append(f"{t.name}: {t.description} {_tool_schema_json(t)}")
        _TOOLS_SCHEMA_CACHE = "\n".join(chunks)
    except Exception:                       # 工具层不可用时按 0 处理，不阻断
        _TOOLS_SCHEMA_CACHE = ""
    return _TOOLS_SCHEMA_CACHE


def static_overhead_tokens() -> int:
    """每次请求的固定输入：系统提示 + 全部工具定义（含框架包装，已标定）。"""
    from agent.prompts import SYSTEM_PROMPT
    raw = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(_tools_schema_text())
    return int(raw * STATIC_CALIBRATION)


# ---------------------------------------------------------------------------
# 成本计算
# ---------------------------------------------------------------------------
def estimate_cost(pricing: Optional[Dict[str, float]],
                  input_tokens: int, output_tokens: int) -> Optional[float]:
    """
    按模型价格折算成本（元）。pricing 形如：
        {"input_per_mtok": 2.0, "output_per_mtok": 3.0}   # 元 / 百万 token
    未配置价格的模型（如本地 Ollama）返回 None（界面上显示"免费/未知"）。
    """
    if not pricing or (not input_tokens and not output_tokens):
        return None
    in_price = float(pricing.get("input_per_mtok", 0) or 0)
    out_price = float(pricing.get("output_per_mtok", 0) or 0)
    if in_price == 0 and out_price == 0:
        return 0.0
    return input_tokens / 1_000_000 * in_price + output_tokens / 1_000_000 * out_price


def fmt_cost(cost: Optional[float]) -> str:
    """成本显示：未配置价格的模型如实说明，正常值保留 4 位小数。"""
    if cost is None:
        return "价格未配置"
    if cost == 0:
        return "¥0"
    return f"¥{cost:.4f}"


# ---------------------------------------------------------------------------
# 事前预估
# ---------------------------------------------------------------------------
@dataclass
class RunEstimate:
    """一轮对话开跑前的用量预估。"""
    input_tokens: int          # 预计输入（系统提示+工具+历史+本次输入，含多步放大）
    output_tokens: int         # 预计输出
    total_tokens: int
    history_tokens: int        # 其中历史对话占多少（可判断"该 /new 了"）
    static_tokens: int         # 其中系统提示+工具定义占多少
    cost: Optional[float]      # 预计成本（元），免费模型为 None
    model_id: str = ""
    expected_calls: int = 1    # 预计模型调用次数（工具任务 > 1）

    def describe(self) -> str:
        total = self.input_tokens + self.output_tokens
        calls_note = f"，预计 ~{self.expected_calls} 次模型调用" if self.expected_calls > 1 else ""
        return (f"预计消耗 ~{total:,} tokens "
                f"(输入 ~{self.input_tokens:,} / 输出 ~{self.output_tokens:,}，"
                f"其中历史对话 ~{self.history_tokens:,}{calls_note})"
                f"{' · ' + fmt_cost(self.cost) if self.cost else ''}")


async def estimate_run(agent: Any, thread_id: str, user_input: str,
                       pricing: Optional[Dict[str, float]] = None,
                       model_id: str = "",
                       avg_output: Optional[int] = None,
                       expected_calls: int = 1,
                       model_kind: str = "chat") -> RunEstimate:
    """
    预估"接下来这一轮"的 token 消耗。

    输入 = 系统提示 + 工具定义（静态开销）
         + 当前会话已积累的历史消息（从 checkpointer 状态读取）
         + 用户本次输入
         工具任务（expected_calls > 1）时，每次后续调用都要把静态开销+历史
         连同新增的工具结果重发一遍，按 EXTRA_CALL_INPUT_FACTOR 逐次放大。
    输出 = 本会话历史输出均值（无历史时用默认值）；
         推理模型（kind=reasoner）的思考链按 REASONER_OUTPUT_FACTOR 放大。
    """
    static_tok = static_overhead_tokens()
    history_tok = 0
    try:
        snapshot = await agent.aget_state({"configurable": {"thread_id": thread_id}})
        msgs = (snapshot.values or {}).get("messages", []) if snapshot else []
        history_tok = messages_tokens(msgs)
    except Exception:
        history_tok = 0          # 拿不到状态就按"无历史"估，宁可低估不可报错

    # 首次调用的输入
    first_call_input = static_tok + history_tok + estimate_tokens(user_input) + PER_MESSAGE_OVERHEAD
    # 工具任务：第 2、3... 次调用在首次基础上逐次累加（静态+历史+逐步变大的上下文）
    input_tok = first_call_input
    for _ in range(max(expected_calls, 1) - 1):
        input_tok += int(first_call_input * EXTRA_CALL_INPUT_FACTOR)

    output_tok = avg_output if avg_output and avg_output > 0 else DEFAULT_OUTPUT_ESTIMATE
    if model_kind == "reasoner":
        output_tok = int(output_tok * REASONER_OUTPUT_FACTOR)

    cost = estimate_cost(pricing, input_tok, output_tok)
    est = RunEstimate(
        input_tokens=input_tok,
        output_tokens=output_tok,
        total_tokens=input_tok + output_tok,
        history_tokens=history_tok,
        static_tokens=static_tok,
        cost=cost,
        model_id=model_id,
        expected_calls=max(expected_calls, 1),
    )
    log_step("usage_estimated", model=model_id,
             input_tokens=input_tok, output_tokens=output_tok,
             expected_calls=est.expected_calls)
    return est


# ---------------------------------------------------------------------------
# 事后统计：基于 checkpointer 状态差分
# ---------------------------------------------------------------------------
async def message_id_snapshot(agent: Any, thread_id: str) -> Set[str]:
    """开跑前记录当前状态里已有的消息 id 集合（用于 diff 本轮新增）。"""
    try:
        snapshot = await agent.aget_state({"configurable": {"thread_id": thread_id}})
        msgs = (snapshot.values or {}).get("messages", []) if snapshot else []
        return {getattr(m, "id", None) or id(m) for m in msgs}
    except Exception:
        return set()


def _usage_of(msg: Any) -> Dict[str, int]:
    """从一条 AI 消息里取 usage_metadata（缺字段时补 0）。"""
    um = getattr(msg, "usage_metadata", None) or {}
    return {
        "input_tokens": int(um.get("input_tokens", 0) or 0),
        "output_tokens": int(um.get("output_tokens", 0) or 0),
    }


@dataclass
class TurnUsage:
    """一轮对话的真实用量（API 返回，含 ReAct 中间步的所有模型调用）。"""
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0              # 本轮模型调用次数（含中间步）
    cached_calls: int = 0       # 其中命中 LLM 缓存的调用数（这些调用几乎不耗 token）
    cost: Optional[float] = None
    model_id: str = ""
    estimated: bool = False     # True=拿不到真实值，用估算兜底

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def describe(self, session: "SessionUsage") -> str:
        src = "估算" if self.estimated else "实耗"
        cache_note = f"，缓存命中 {self.cached_calls} 次" if self.cached_calls else ""
        return (f"本次{src} {self.total_tokens:,} tokens "
                f"(输入 {self.input_tokens:,} / 输出 {self.output_tokens:,}，"
                f"模型调用 {self.calls} 次{cache_note})"
                f"{' · ' + fmt_cost(self.cost) if self.cost is not None else ''} · "
                f"会话累计 {session.total_tokens:,} tokens / {fmt_cost(session.cost)}")

    def deviation(self, estimate: Optional["RunEstimate"]) -> str:
        """相对事前预估的偏差百分比（预估偏保守时为正）。"""
        if not estimate or estimate.total_tokens <= 0:
            return ""
        pct = (self.total_tokens - estimate.total_tokens) / estimate.total_tokens * 100
        if abs(pct) < 3:
            return "，与预估基本一致"
        return f"，比预估{'多' if pct > 0 else '少'} {abs(pct):.0f}%"


async def actual_usage(agent: Any, thread_id: str, before_ids: Set[str],
                       pricing: Optional[Dict[str, float]] = None,
                       model_id: str = "") -> Optional[TurnUsage]:
    """
    统计本轮真实用量：读取最新状态，筛出【开跑后新增的 AI 消息】，
    汇总其 usage_metadata。拿不到真实值时退化为文本估算（estimated=True）。
    """
    try:
        snapshot = await agent.aget_state({"configurable": {"thread_id": thread_id}})
        msgs = (snapshot.values or {}).get("messages", []) if snapshot else []
    except Exception:
        return None              # 状态都读不到就不展示，不误导用户

    new_ai = [m for m in msgs
              if getattr(m, "type", "") == "ai"
              and (getattr(m, "id", None) or id(m)) not in before_ids]

    if not new_ai:
        return None              # 本轮没有新增（如中断/失败），不显示

    usage = TurnUsage(model_id=model_id)
    has_real = False
    for m in new_ai:
        u = _usage_of(m)
        if u["input_tokens"] or u["output_tokens"]:
            has_real = True
            usage.input_tokens += u["input_tokens"]
            usage.output_tokens += u["output_tokens"]
        else:
            # 消息存在但 usage 为空 → 命中了 LLM 缓存（缓存响应不回传用量，
            # 也几乎不产生计费 token），如实标注而不是当作 0 混淆视听
            usage.cached_calls += 1
        usage.calls += 1

    if not has_real:
        # 流式模式下个别供应商不回传 usage → 用估算兜底并如实标注
        usage.input_tokens = static_overhead_tokens() + messages_tokens(
            [m for m in msgs if (getattr(m, "id", None) or id(m)) not in before_ids])
        usage.output_tokens = sum(estimate_tokens(message_text(m)) for m in new_ai)
        usage.cached_calls = 0
        usage.estimated = True

    usage.cost = estimate_cost(pricing, usage.input_tokens, usage.output_tokens)
    log_step("usage_actual", model=model_id, estimated=usage.estimated,
             input_tokens=usage.input_tokens,
             output_tokens=usage.output_tokens, calls=usage.calls,
             cached_calls=usage.cached_calls)
    return usage


# ---------------------------------------------------------------------------
# 上下文压缩：触发点计算 + 压缩检测
# ---------------------------------------------------------------------------
# 目录没声明输出上限时，按 8K 估算（主流模型的常见上限，宁可低估触发点）
UNKNOWN_OUTPUT_CAP = 8192
# SummarizationMiddleware 生成的摘要消息带这个标记（见 langchain 源码
# _build_new_messages：additional_kwargs={"lc_source": "summarization"}）
SUMMARY_LC_SOURCE = "summarization"


def compute_summary_trigger(
    max_input_tokens: Optional[int],
    max_output_tokens: Optional[int] = None,
    static_tokens: int = 0,
    safety_margin: int = 8192,
) -> Optional[int]:
    """
    按"模型窗口预算"计算上下文压缩的触发点（history 消息达到多少 token 时压缩）。

        触发点 = 模型窗口 - 单次最大输出 - 静态开销 - 安全余量

    这样一次完整请求（静态 + 历史 + 输出）永远落在窗口内：
    窗口越大触发点越高（用满大窗口），输出上限越大触发点越低（给长输出留位）。
    窗口未知时返回 None，由调用方退化为绝对阈值（.env 的
    SUMMARIZE_TRIGGER_TOKENS）——宁早勿晚，压缩频繁好过请求溢出。
    """
    if not isinstance(max_input_tokens, int) or max_input_tokens <= 0:
        return None
    output_cap = (max_output_tokens
                  if isinstance(max_output_tokens, int) and max_output_tokens > 0
                  else UNKNOWN_OUTPUT_CAP)
    budget = max_input_tokens - output_cap - max(static_tokens, 0) - max(safety_margin, 0)
    # 兜底下限 1024：预算被余量吃光（极端小窗口）时也保持一个正触发点，
    # 不用"窗口比例"当下限——那本身就可能顶破窗口
    return max(int(budget), 1024)


def is_summary_message(msg: Any) -> bool:
    """判断一条消息是否是 SummarizationMiddleware 生成的压缩摘要。"""
    return bool((getattr(msg, "additional_kwargs", None) or {}).get(
        "lc_source") == SUMMARY_LC_SOURCE)


async def detect_compaction(agent: Any, thread_id: str,
                            before_ids: Set[str]) -> int:
    """
    统计本轮对话是否发生了上下文压缩（读取最新状态，找出开跑后新增的摘要消息）。

    注意：一次运行内多次压缩时，旧摘要会被折叠进新摘要，最终状态里只剩一份，
    因此返回值是"至少压缩次数"（0 或 1），用于向用户提示，不做精确计数。
    """
    try:
        snapshot = await agent.aget_state({"configurable": {"thread_id": thread_id}})
        msgs = (snapshot.values or {}).get("messages", []) if snapshot else []
    except Exception:
        return 0
    return sum(
        1 for m in msgs
        if is_summary_message(m) and (getattr(m, "id", None) or id(m)) not in before_ids
    )


# ---------------------------------------------------------------------------
# 会话累计（进程内，/new 时清零）
# ---------------------------------------------------------------------------
@dataclass
class SessionUsage:
    """一次进程内、一个会话(thread)的累计用量。线程安全。
    中途切换模型时，按模型分别累计（/usage 可看到各模型各花了多少）。"""
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    _output_history: List[int] = field(default_factory=list)
    _by_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage: TurnUsage) -> None:
        with self._lock:
            self.turns += 1
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
            if usage.cost:
                self.cost += usage.cost
            self._output_history.append(usage.output_tokens)
            # 按模型累计（切换模型后各花多少一目了然）
            mid = usage.model_id or "(未知模型)"
            slot = self._by_model.setdefault(
                mid, {"turns": 0, "input": 0, "output": 0, "cost": 0.0})
            slot["turns"] += 1
            slot["input"] += usage.input_tokens
            slot["output"] += usage.output_tokens
            if usage.cost:
                slot["cost"] += usage.cost

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def avg_output(self) -> int:
        """历史每轮平均输出 token——用作下一轮的输出预估。"""
        with self._lock:
            if not self._output_history:
                return DEFAULT_OUTPUT_ESTIMATE
            return max(int(sum(self._output_history) / len(self._output_history)), 32)

    def per_model(self) -> List[tuple]:
        """[(model_id, {turns,input,output,cost}), ...]，按用量降序。"""
        with self._lock:
            rows = [(mid, dict(s)) for mid, s in self._by_model.items()]
        rows.sort(key=lambda r: r[1]["input"] + r[1]["output"], reverse=True)
        return rows

    def reset(self) -> None:
        with self._lock:
            self.turns = 0
            self.input_tokens = 0
            self.output_tokens = 0
            self.cost = 0.0
            self._output_history.clear()
            self._by_model.clear()

    def describe(self) -> str:
        lines = [f"本会话累计：{self.turns} 轮 · "
                 f"输入 {self.input_tokens:,} / 输出 {self.output_tokens:,} / "
                 f"共 {self.total_tokens:,} tokens · 成本约 {fmt_cost(self.cost)}"]
        rows = self.per_model()
        if len(rows) > 1:        # 用过多个模型时才展开明细，避免噪音
            lines.append("分模型：")
            for mid, s in rows:
                lines.append(
                    f"  · {mid:<22} {s['turns']} 轮 · "
                    f"{s['input'] + s['output']:,} tokens · {fmt_cost(s['cost'])}")
        return "\n".join(lines)
