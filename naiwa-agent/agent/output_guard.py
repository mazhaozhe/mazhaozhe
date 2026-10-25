"""
agent/output_guard.py —— 模型输出退化检测（防"复读机"）
======================================================

现象
----
模型陷入"决策瘫痪"时（反复权衡 A 还是 B、该不该执行），自回归输出会退化成
短句循环：

    好。执行。好。（执行）好。执行。好。我调用。好。执行。……

实测一次 17,193 字符，其中「执行」出现 935 次、「好」1,234 次，2570 行里只有
149 种，直到把 max_tokens（8192）烧光，实耗 3.4 万 tokens；这段垃圾还会被写进
会话历史，下一轮又喂回模型，**自我强化**。

为什么不能只靠 prompt
--------------------
这是 LLM 的神经文本退化（neural text degeneration）：低 temperature +
无 repetition penalty + 模型卡在二选一时最容易触发。提示词能降低概率，
但只有工程护栏能保证"烧不了几万 token"。

三道防线
--------
1. 采样层：frequency_penalty 抑制重复（core.py 的 build_model）
2. 流式层：退化一出现立刻中止生成（client/bridge.py 的 consume_stream）
3. 历史层：把已膨胀的历史消息截断，别污染下一轮（history_repair.clean_degenerate）

两个检测器（取更早的那个起点）
------------------------------
A. 多样性塌缩（主判据）
   正常文本的 4-gram 去重率约 0.4~0.9；退化文本跌到 0.02 量级。
   在一个滑动窗口上算这个比值，找"从某处起一路塌缩到底"的边界。
   能抓住不严格的循环（中间夹「（执行）」「（停止过度思考）」等变体）。

B. 严格周期（兜底）
   某个 4~40 字符单元连续重复 ≥10 次、总长 ≥100 字符。
   能抓住短促的死循环，包括没有换行、整段挤在一行的情况。

阈值取值见下方常量，实测数据见 tests/test_output_guard.py 的注释。
"""

from __future__ import annotations

# ---------- 检测器 A：多样性塌缩 ----------
_WINDOW = 600                # 滑动窗口（字符）
_STEP = 150                 # 检查点步长
_RATIO_THRESHOLD = 0.18      # 4-gram 去重率低于此值视为退化（正常文本 >= 0.33）
_MIN_TAIL = 900              # 退化段至少这么长才认定（避免一闪而过的重复）

# ---------- 检测器 B：严格周期 ----------
_MIN_UNIT = 4                # 最短重复单元（"（执行）" 正好 4 字符）
_MAX_UNIT = 40               # 最长重复单元（整句话）
_MIN_REPEATS = 10            # 至少连续重复 10 次
_MIN_RUN_CHARS = 100         # 且重复段总长 ≥100 字符

# 太短的文本不值得检测（正常答复经常只有一两百字）
_MIN_LEN = 400

# 流式场景下严格周期检测的扫描窗口 / 触发间隔（控制单次开销）
_PERIODIC_TAIL = 3000
_PERIODIC_EVERY = 400


def _dgram_ratio(seg: str, k: int = 4) -> float:
    """4-gram 去重率：不同 4-gram 数 / 总 4-gram 数。越接近 1 越"新鲜"。"""
    if len(seg) <= k:
        return 1.0
    grams = {seg[i:i + k] for i in range(len(seg) - k + 1)}
    return len(grams) / (len(seg) - k + 1)


def _collapse_start(text: str, points: list[tuple[int, float]] | None = None) -> int | None:
    """
    定位"多样性塌缩"起点。

    points 是 (检查点位置, 该处窗口的 4-gram 去重率) 列表，按位置升序。
    返回最早的 p，使 p 之后所有检查点都低于阈值；不足 _MIN_TAIL 长则不认定。
    """
    if points is None:
        points = []
        for pos in range(0, max(0, len(text) - _WINDOW + _STEP), _STEP):
            points.append((pos, _dgram_ratio(text[pos:pos + _WINDOW])))
        if not points:
            return None

    # 从尾部往前找连续低于阈值的最后一个断点
    tail_start = len(points)
    for idx in range(len(points) - 1, -1, -1):
        if points[idx][1] < _RATIO_THRESHOLD:
            tail_start = idx
        else:
            break
    if tail_start >= len(points):
        return None                              # 尾部没有塌缩
    start = points[tail_start][0]
    if len(text) - start < _MIN_TAIL:
        return None                              # 塌缩段太短，不作为退化
    return start


def _periodic_start(text: str, tail_limit: int = 0) -> int | None:
    """
    检测严格周期重复段，返回其起点（取最早的）。

    tail_limit > 0 时只看尾部这么多字符（流式场景：退化必然发生在当前输出末尾，
    命中后靠向前扩展拿回真正的起点），把开销压成常数级。
    """
    n = len(text)
    if n < max(_MIN_UNIT * _MIN_REPEATS, _MIN_RUN_CHARS):
        return None
    offset = 0
    if tail_limit and n > tail_limit:
        offset = n - tail_limit
        text = text[offset:]
        n = tail_limit
    best: int | None = None
    for unit_len in range(_MIN_UNIT, _MAX_UNIT + 1):
        limit = n - unit_len
        if best is not None and best < limit:
            limit = best
        i = 0
        while i < limit:
            # 单字符快速否决：周期成立必然 text[i] == text[i+unit_len]
            if text[i] != text[i + unit_len]:
                i += 1
                continue
            unit = text[i:i + unit_len]
            if not unit.strip() or len(set(unit)) < 2:   # 空白 / "----" 不算
                i += unit_len
                continue
            blocks = 1
            j = i + unit_len
            while j + unit_len <= n and text[j:j + unit_len] == unit:
                blocks += 1
                j += unit_len
            if blocks >= _MIN_REPEATS and blocks * unit_len >= _MIN_RUN_CHARS:
                if best is None or i < best:
                    best = i
                break
            i = j if blocks > 1 else i + 1
    return None if best is None else best + offset


def detect_degeneration(text: str) -> int | None:
    """返回退化起点索引（该位置往后都是重复垃圾）；没有退化返回 None。"""
    if not text or len(text) < _MIN_LEN:
        return None
    candidates = [p for p in (_collapse_start(text), _periodic_start(text)) if p is not None]
    return min(candidates) if candidates else None


def trim_degeneration(text: str) -> tuple[str, bool]:
    """
    截断退化尾巴，返回 (保留内容, 是否发生过退化)。

    保留部分是退化开始前的有效输出（可能为空或只有半句话），调用方据此
    决定是"当成答案用"还是"判定本次没答出来"。
    """
    at = detect_degeneration(text)
    if at is None:
        return text, False
    return text[:at].rstrip(), True


def is_degenerate(message_text: str, min_chars: int = _MIN_LEN) -> bool:
    """历史消息是否已被退化污染（用于历史清洗的快速判断）。"""
    if not message_text or len(message_text) < min_chars:
        return False
    return detect_degeneration(message_text) is not None


class StreamGuard:
    """
    流式护栏：逐块喂入模型输出，检测到退化就返回起点，调用方据此中止生成。

    增量计算检查点，总开销随文本长度线性增长（一次流式只扫一遍）；
    严格周期检测只看尾部 _PERIODIC_TAIL 字符，避免每块都做全文扫描。
    """

    __slots__ = ("_buf", "_points", "_scanned", "_hit", "_periodic_at")

    def __init__(self) -> None:
        self._buf = ""
        self._points: list[tuple[int, float]] = []
        self._scanned = 0            # 已算过检查点的文本长度
        self._periodic_at = 0        # 上次做严格周期检测时的文本长度
        self._hit: int | None = None

    def feed(self, text: str) -> int | None:
        """
        用最新的完整累积文本调用；发现退化时返回退化起点（之后持续返回同一个值）。
        """
        if self._hit is not None:
            return self._hit
        length = len(text)
        if length < _MIN_LEN:
            self._buf = text
            return None

        self._buf = text
        # A. 多样性塌缩：只补算新增检查点（整体 O(n)，不会重复劳动）
        pos = self._scanned
        while pos + _WINDOW <= length:
            self._points.append((pos, _dgram_ratio(text[pos:pos + _WINDOW])))
            pos += _STEP
        self._scanned = pos
        if self._points:
            start = _collapse_start(text, self._points)
            if start is not None:
                self._hit = start
                return self._hit

        # B. 严格周期：每积累 _PERIODIC_EVERY 字符做一次，只看尾部
        if length - self._periodic_at >= _PERIODIC_EVERY:
            self._periodic_at = length
            periodic = _periodic_start(text, tail_limit=_PERIODIC_TAIL)
            if periodic is not None and length - periodic >= _MIN_TAIL:
                self._hit = periodic
        return self._hit

    @property
    def hit(self) -> int | None:
        """已判定退化时返回起点。"""
        return self._hit

    def text(self) -> str:
        """截止到干预点的文本（未命中时为全部累积文本）。"""
        if self._hit is None:
            return self._buf
        return self._buf[: self._hit].rstrip()
