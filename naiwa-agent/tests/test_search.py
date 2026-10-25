"""
tests/test_search.py —— 联网搜索工具测试
========================================
覆盖 tools/search.py 的多源级联与熔断冷却（全部 mock 网络，不真实联网）：

- 天气类查询 → Open-Meteo 结构化数据（识别城市、输出实况/今日预报）
- 通用查询 → Bing 中国站 RSS 解析
- Bing 失败 → DuckDuckGo 兜底
- 全源失败 → 返回说明文本（不抛异常，不触发工具重试）
- 熔断：连续失败达阈值后熔断，60 秒冷却后自动半开
- 失败说明不进缓存（恢复后能立刻拿到真实结果）

背景：旧实现只用 DuckDuckGo 即时问答，国内不可达（实测 10 秒超时），
连续失败两次即永久熔断到进程重启——用户看到的就是「连查天气都不行」。
"""

import pytest

import tools.search as search  # noqa: E402

BING_XML = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel>
<item><title>结果一</title><link>https://example.com/1</link>
<description>摘要一</description></item>
<item><title>结果二</title><link>https://example.com/2</link>
<description>摘要二</description></item>
</channel></rss>"""

WEATHER_JSON = {
    "current": {
        "time": "2026-10-25T13:00", "temperature_2m": 30.2,
        "relative_humidity_2m": 46, "apparent_temperature": 32.6,
        "weather_code": 3, "wind_speed_10m": 9.2,
    },
    "daily": {
        "temperature_2m_max": [31.2], "temperature_2m_min": [21.8],
        "precipitation_probability_max": [0],
    },
}

GEO_JSON = {"results": [{"name": "测试城", "latitude": 30.0, "longitude": 120.0}]}


class FakeResp:
    def __init__(self, text="", payload=None, status=200):
        self.text = text
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise search.requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def reset_state():
    """隔离熔断全局状态与模块级缓存，避免用例互相污染。"""
    search._FAIL_STREAK = 0
    search._CIRCUIT_OPEN = False
    search._CIRCUIT_OPENED_AT = 0.0
    # 清掉搜索模块的三个 TTL 缓存（天气/搜索/地理编码），否则上个用例的
    # 成功结果会盖住当前用例 mock 的失败场景
    for fn in (search._fetch_weather, search._fetch_bing_rss, search._geocode):
        cache = getattr(fn, "cache", None)
        if cache is not None:
            with cache._lock:
                cache._data.clear()
    yield
    search._FAIL_STREAK = 0
    search._CIRCUIT_OPEN = False
    search._CIRCUIT_OPENED_AT = 0.0


def install_fake_net(monkeypatch, *, bing="ok", ddg="fail", weather="ok", calls=None):
    """安装假的 requests.get：按 URL 分派，可分别控制各源成败。"""
    calls = calls if calls is not None else []

    def fake_get(url, params=None, headers=None, timeout=None, **kwargs):
        calls.append(url)
        if url == search.METEO_FORECAST:
            if weather == "fail":
                raise search.requests.ConnectionError("weather down")
            return FakeResp(payload=WEATHER_JSON)
        if url == search.METEO_GEOCODE:
            return FakeResp(payload=GEO_JSON)
        if url == search.BING_RSS:
            if bing == "fail":
                raise search.requests.ConnectionError("bing down")
            if bing == "garbage":
                return FakeResp(text="<html>被拦截</html>")
            return FakeResp(text=BING_XML)
        if url == search.DDG_ENDPOINT:
            if ddg == "fail":
                raise search.requests.ConnectionError("ddg unreachable")
            return FakeResp(payload={"AbstractText": "DDG 摘要", "AbstractURL": "https://ddg.example"})
        raise AssertionError(f"未预期的请求: {url}")

    monkeypatch.setattr(search.requests, "get", fake_get)
    return calls


# ---------------------------------------------------------------------------
# 天气通道
# ---------------------------------------------------------------------------
class TestWeatherChannel:
    def test_guangzhou_weather_from_open_meteo(self, monkeypatch):
        install_fake_net(monkeypatch)
        out = search.search_web.invoke({"query": "广州今天天气怎么样"})
        assert "广州" in out
        assert "30.2°C" in out and "体感 32.6°C" in out
        assert "阴" in out                       # weather_code=3
        assert "最高 31.2°C / 最低 21.8°C" in out
        assert "降水概率 0%" in out
        assert "Open-Meteo" in out

    def test_city_from_geocoding_when_not_builtin(self, monkeypatch):
        install_fake_net(monkeypatch)
        out = search.search_web.invoke({"query": "测试城天气"})
        assert "测试城" in out and "30.2°C" in out

    def test_weather_failure_returns_text_not_exception(self, monkeypatch):
        """天气源故障返回说明文本：抛异常会被重试放大成数分钟等待。"""
        install_fake_net(monkeypatch, weather="fail")
        out = search.search_web.invoke({"query": "广州天气"})
        assert "暂时不可用" in out or "无法" in out
        assert "30.2°C" not in out

    def test_weather_without_city_falls_back_to_search(self, monkeypatch):
        """只说"天气怎么样"识别不出城市 → 走通用搜索，不误报天气。"""
        install_fake_net(monkeypatch)
        out = search.search_web.invoke({"query": "天气怎么样"})
        assert "结果一" in out


# ---------------------------------------------------------------------------
# 通用搜索通道
# ---------------------------------------------------------------------------
class TestGeneralSearch:
    def test_bing_rss_results(self, monkeypatch):
        install_fake_net(monkeypatch)
        out = search.search_web.invoke({"query": "Python 3.13 新特性"})
        assert "结果一" in out and "https://example.com/1" in out and "摘要一" in out
        assert "结果二" in out

    def test_bing_failure_falls_back_to_ddg(self, monkeypatch):
        install_fake_net(monkeypatch, bing="fail", ddg="ok")
        out = search.search_web.invoke({"query": "冷门关键词"})
        assert "DDG 摘要" in out

    def test_all_sources_fail_returns_hint(self, monkeypatch):
        install_fake_net(monkeypatch, bing="fail", ddg="fail")
        out = search.search_web.invoke({"query": "冷门关键词二"})
        assert "没有检索到" in out and "无法联网核实" in out

    def test_bing_garbage_html_not_parsed_as_results(self, monkeypatch):
        """返回非 XML（被拦截）时不崩，走兜底或给说明。"""
        install_fake_net(monkeypatch, bing="garbage", ddg="fail")
        out = search.search_web.invoke({"query": "冷门关键词三"})
        assert "没有检索到" in out


# ---------------------------------------------------------------------------
# 熔断：带冷却，不再永久挂死
# ---------------------------------------------------------------------------
class TestCircuitBreaker:
    def test_opens_after_threshold(self, monkeypatch):
        install_fake_net(monkeypatch, bing="fail", ddg="fail")
        for i in range(search._CIRCUIT_THRESHOLD):
            search.search_web.invoke({"query": f"连续失败{i}"})
        assert search._CIRCUIT_OPEN is True
        out = search.search_web.invoke({"query": "熔断后查询"})
        assert "暂时不可用" in out and "冷却" in out

    def test_recovers_after_cooldown(self, monkeypatch):
        """核心回归：冷却到点必须自动恢复，不能像旧版那样等到进程重启。"""
        install_fake_net(monkeypatch, bing="fail", ddg="fail")
        for i in range(search._CIRCUIT_THRESHOLD):
            search.search_web.invoke({"query": f"失败{i}"})
        assert search._CIRCUIT_OPEN is True

        # 时间跳到冷却之后：应当放行并成功
        now = search.time.monotonic() + search._CIRCUIT_COOLDOWN + 1
        monkeypatch.setattr(search.time, "monotonic", lambda: now)
        install_fake_net(monkeypatch)          # 网络恢复
        out = search.search_web.invoke({"query": "恢复后查询"})
        assert "结果一" in out
        assert search._CIRCUIT_OPEN is False

    def test_success_resets_streak(self, monkeypatch):
        install_fake_net(monkeypatch, bing="fail", ddg="fail")
        search.search_web.invoke({"query": "失败一次"})
        assert search._FAIL_STREAK == 1
        install_fake_net(monkeypatch)          # 网络恢复
        search.search_web.invoke({"query": "成功一次"})
        assert search._FAIL_STREAK == 0


# ---------------------------------------------------------------------------
# 缓存：失败说明不能被缓存住
# ---------------------------------------------------------------------------
class TestCaching:
    def test_failure_text_not_cached(self, monkeypatch):
        """
        旧实现把"接口不可用"文本也缓存了 10 分钟：网络恢复后仍返回坏结果。
        现在缓存只包真实数据请求，失败不写入。
        """
        calls = install_fake_net(monkeypatch, bing="fail", ddg="fail", weather="fail")
        first = search.search_web.invoke({"query": "广州天气"})
        second = search.search_web.invoke({"query": "广州天气"})
        assert first.startswith("天气服务暂时不可用")
        assert second.startswith("天气服务暂时不可用")   # 两次都真实请求，不缓存失败

        # 网络恢复后同一查询必须能拿到真实数据（不被失败文本缓存挡住）
        install_fake_net(monkeypatch)
        out = search.search_web.invoke({"query": "广州天气"})
        assert "30.2°C" in out
        assert len(calls) >= 2
