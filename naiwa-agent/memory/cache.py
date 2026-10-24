"""
memory/cache.py —— 缓存层（重复信息不重复花钱）
================================================
两层缓存，对应需求"我这个 agent 要学会对重复的信息进行缓存调用"：

1. LLM 缓存（SQLiteCache）：完全相同的模型请求直接命中，零 token 消耗；
2. 工具结果缓存（TTLCache）：幂等工具（搜索、读文件等）在 TTL 内复用结果。

注意：只用于幂等操作；写文件、发邮件等有副作用的调用绝不能缓存。
"""

import functools
import hashlib
import json
import threading
import time
from pathlib import Path

from agent.observability import log_step


# ---------------------------------------------------------------------------
# 1. LLM 全局缓存
# ---------------------------------------------------------------------------
def enable_llm_cache(db_path: str | Path) -> bool:
    """开启 LangChain 全局 LLM 缓存（SQLite 落盘，跨进程可用）。"""
    try:
        from langchain_core.globals import set_llm_cache
        from langchain_community.cache import SQLiteCache

        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        set_llm_cache(SQLiteCache(database_path=str(db_path)))
        log_step("llm_cache_enabled", path=str(db_path))
        return True
    except Exception as exc:  # 缓存不可用不应该阻断主流程
        print(f"[cache] LLM 缓存启用失败（不影响主功能）: {type(exc).__name__}: {exc}")
        return False


# ---------------------------------------------------------------------------
# 2. 工具结果 TTL 缓存
# ---------------------------------------------------------------------------
class TTLCache:
    """线程安全的 TTL 内存缓存（LRU 淘汰交给上层，此处只做过期控制）。"""

    def __init__(self, ttl_seconds: int = 600, max_items: int = 500):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._data: dict[str, tuple[float, object]] = {}   # key -> (过期时刻, 值)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(fn_name: str, args: tuple, kwargs: dict) -> str:
        """由函数名 + 参数生成稳定哈希键（参数顺序不影响结果）。"""
        payload = json.dumps(
            {"fn": fn_name, "args": args, "kwargs": kwargs},
            ensure_ascii=False, sort_keys=True, default=str,
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def get(self, key: str):
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            expire_at, value = item
            if time.monotonic() > expire_at:
                # 已过期：顺手清理
                del self._data[key]
                self.misses += 1
                return None
            self.hits += 1
            return value

    def set(self, key: str, value) -> None:
        with self._lock:
            # 简单容量控制：超过上限时清掉最早的 10%
            if len(self._data) >= self.max_items:
                for k in list(self._data)[: max(1, self.max_items // 10)]:
                    self._data.pop(k, None)
            self._data[key] = (time.monotonic() + self.ttl, value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0


def cached(ttl_seconds: int = 600):
    """
    工具函数缓存装饰器。

    用法：
        @cached(ttl_seconds=600)
        def get_weather(city: str): ...
        get_weather("广州")   # 第一次真正执行
        get_weather("广州")   # 直接命中缓存

    被装饰函数暴露 .cache 属性（TTLCache 实例），便于统计命中率。
    """
    def decorator(fn):
        cache = TTLCache(ttl_seconds=ttl_seconds)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = TTLCache.make_key(fn.__name__, args, kwargs)
            hit = cache.get(key)
            if hit is not None:
                log_step("cache_hit", fn=fn.__name__)
                return hit
            value = fn(*args, **kwargs)
            cache.set(key, value)
            return value

        wrapper.cache = cache          # 暴露缓存实例
        return wrapper

    return decorator
