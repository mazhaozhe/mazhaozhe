"""
tests/test_cache_and_config.py —— 缓存与配置测试
=================================================
1. TTL 缓存：命中 / 过期 / 参数区分；
2. 配置：默认值加载 + 缺 Key 时的启动校验提示。
"""

import time

from memory.cache import TTLCache, cached
from agent.config import Settings


# ---------------------------------------------------------------------------
# TTLCache
# ---------------------------------------------------------------------------
def test_ttl_cache_hit_and_expire():
    """TTL 内命中，过期后未命中。"""
    cache = TTLCache(ttl_seconds=1)
    cache.set("k", "v")
    assert cache.get("k") == "v"       # 命中
    time.sleep(1.1)
    assert cache.get("k") is None      # 过期


def test_ttl_cache_key_stability():
    """相同参数 → 相同 key；不同参数 → 不同 key。"""
    k1 = TTLCache.make_key("fn", ("a", 1), {"x": 1})
    k2 = TTLCache.make_key("fn", ("a", 1), {"x": 1})
    k3 = TTLCache.make_key("fn", ("a", 2), {"x": 1})
    assert k1 == k2 and k1 != k3


def test_cached_decorator():
    """装饰器：第二次调用不再执行原函数（用调用计数验证）。"""
    calls = {"n": 0}

    @cached(ttl_seconds=60)
    def fetch(city: str) -> str:
        calls["n"] += 1
        return f"weather of {city}"

    assert fetch("广州") == "weather of 广州"
    assert fetch("广州") == "weather of 广州"   # 命中缓存
    assert calls["n"] == 1                      # 原函数只执行了 1 次
    assert fetch.cache.hits == 1


# ---------------------------------------------------------------------------
# 配置校验
# ---------------------------------------------------------------------------
def test_settings_defaults():
    """未配置 .env 时使用安全默认值。"""
    s = Settings(_env_file=None)  # 显式跳过 .env，纯默认值
    assert s.main_model == "deepseek-chat"
    assert s.model_run_limit == 60
    assert s.tool_run_limit == 100
    assert s.calls_per_minute == 60
    assert s.summarize_trigger_tokens == 16000
    assert s.summarize_keep_messages == 20
    assert s.summarize_safety_margin == 8192
    assert s.summarize_trim_tokens == 0


def test_validate_runtime_reports_missing_key():
    """缺 API Key → validate_runtime 给出明确修复指引。"""
    s = Settings(_env_file=None, deepseek_api_key="")
    problems = s.validate_runtime()
    assert len(problems) == 1
    assert "DEEPSEEK_API_KEY" in problems[0]

    s_ok = Settings(_env_file=None, deepseek_api_key="sk-real-key")
    assert s_ok.validate_runtime() == []
