"""
middleware/rate_limit.py —— 自定义限流中间件
=============================================
LangChain 内置中间件覆盖了"调用次数上限"（安全阀），
但没有覆盖"频率 + 并发"限制 —— 本模块补上这块：
- 令牌间隔：保证相邻两次工具调用至少间隔 60/calls_per_minute 秒；
- 并发信号量：同时执行的工具调用不超过 max_concurrency 个。
对应需求："可以适当限制并发和调用频率以减小开销"。
"""

import asyncio
import time

from langchain.agents.middleware import AgentMiddleware


class RateLimitMiddleware(AgentMiddleware):
    """工具调用限流：令牌间隔 + 并发上限（对应 .env 的 CALLS_PER_MINUTE / MAX_CONCURRENCY）。"""

    def __init__(self, calls_per_minute: int = 30, max_concurrency: int = 4):
        super().__init__()
        if calls_per_minute < 1:
            raise ValueError("calls_per_minute 必须 >= 1")
        self._min_interval = 60.0 / calls_per_minute  # 相邻调用最小间隔（秒）
        self._semaphore = asyncio.Semaphore(max_concurrency)  # 并发上限
        self._last_call = 0.0                            # 上次放行的时间戳
        self._lock = asyncio.Lock()                      # 保护 _last_call 的更新

    async def awrap_tool_call(self, request, handler):
        """
        包裹每一次工具调用：先拿并发名额，再按令牌间隔排队，最后放行。
        （同步版 wrap_tool_call 不实现：本项目入口为 asyncio，全异步链路。）
        """
        async with self._semaphore:
            async with self._lock:
                # 计算需要等待的时间：距上次放行不足一个间隔就 sleep
                wait = self._last_call + self._min_interval - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_call = time.monotonic()
            return await handler(request)
