"""
tools/search.py —— 联网搜索工具
================================
两条主通道 + 一条兜底（2026-10 重构，修复"连查天气都不行"）：

1. 天气类查询 → Open-Meteo（免费、无需 Key、中国大陆可直连）
   先地理编码定位城市，再取实况 + 今日高低温/降水概率，输出结构化中文数据。
2. 其他查询 → Bing 中国站 RSS（免费、无需 Key、结构稳定、可直连）
   解析 <item> 得到标题 / 链接 / 摘要，交给模型组织答案。
3. 兜底 → DuckDuckGo 即时问答（国内通常不可达，开代理时才有用）。

为什么不再用 DDG 作主源：本机实测 api.duckduckgo.com 10 秒超时（HTTP 000），
连续失败即触发熔断，之后**所有**搜索都返回"已熔断"——用户看到的就是
"连查天气都不行"，且只能重启客户端才恢复。熔断现在改为 60 秒冷却自动
半开；失败说明文本也不再进缓存（否则修好了还会返回旧的坏结果）。

失败处理：网络类异常抛内置 ConnectionError / TimeoutError（交给重试中间件，
只重试这两类）；搜索源不可用则返回说明文本，让模型换路径而不是白等。
"""

import re
import time
import xml.etree.ElementTree as ET

import requests
from langchain.tools import tool
from pydantic import BaseModel, Field

from memory.cache import cached
from tools.registry import register

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}
BING_RSS = "https://cn.bing.com/search"
DDG_ENDPOINT = "https://api.duckduckgo.com/"
METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

# ---------------------------------------------------------------------------
# 失败熔断（带冷却）：连续失败达到阈值后短暂停用，冷却结束自动半开试探
# ---------------------------------------------------------------------------
_FAIL_STREAK = 0
_CIRCUIT_OPEN = False
_CIRCUIT_OPENED_AT = 0.0
_CIRCUIT_THRESHOLD = 3
_CIRCUIT_COOLDOWN = 60.0        # 秒


def _circuit_blocked() -> bool:
    """是否仍在熔断冷却期。冷却到期后自动半开（放行一次试探）。"""
    global _CIRCUIT_OPEN
    if not _CIRCUIT_OPEN:
        return False
    if time.monotonic() - _CIRCUIT_OPENED_AT >= _CIRCUIT_COOLDOWN:
        _CIRCUIT_OPEN = False
        return False
    return True


def _record_result(ok: bool) -> None:
    global _FAIL_STREAK, _CIRCUIT_OPEN, _CIRCUIT_OPENED_AT
    if ok:
        _FAIL_STREAK = 0
        _CIRCUIT_OPEN = False
        return
    _FAIL_STREAK += 1
    if _FAIL_STREAK >= _CIRCUIT_THRESHOLD:
        _CIRCUIT_OPEN = True
        _CIRCUIT_OPENED_AT = time.monotonic()


# ---------------------------------------------------------------------------
# 天气通道：Open-Meteo
# ---------------------------------------------------------------------------
_WEATHER_KEYWORDS = (
    "天气", "气温", "温度", "下雨", "降雨", "降水", "台风", "风力", "风速",
    "湿度", "冷不冷", "热不热", "多少度", "穿什么", "下雪",
)

# 常见城市内置坐标（省一次地理编码请求；用户输入里的城市名直接命中）
_CITY_COORDS = {
    "广州": (23.13, 113.26), "深圳": (22.54, 114.06), "北京": (39.90, 116.41),
    "上海": (31.23, 121.47), "杭州": (30.27, 120.16), "成都": (30.57, 104.07),
    "武汉": (30.59, 114.31), "南京": (32.06, 118.80), "西安": (34.34, 108.94),
    "重庆": (29.56, 106.55), "天津": (39.13, 117.20), "苏州": (31.30, 120.62),
    "长沙": (28.23, 112.94), "郑州": (34.75, 113.63), "青岛": (36.07, 120.38),
    "厦门": (24.48, 118.09), "福州": (26.07, 119.30), "昆明": (25.04, 102.72),
    "合肥": (31.82, 117.23), "济南": (36.65, 117.12), "沈阳": (41.81, 123.43),
    "大连": (38.91, 121.61), "哈尔滨": (45.80, 126.53), "长春": (43.82, 125.32),
    "石家庄": (38.04, 114.51), "太原": (37.87, 112.55), "南昌": (28.68, 115.86),
    "南宁": (22.82, 108.37), "贵阳": (26.65, 106.63), "兰州": (36.06, 103.83),
    "海口": (20.04, 110.32), "东莞": (23.02, 113.75), "佛山": (23.02, 113.12),
    "珠海": (22.27, 113.58), "汕头": (23.35, 116.68), "无锡": (31.49, 120.31),
    "宁波": (29.87, 121.54), "温州": (28.00, 120.67), "常州": (31.81, 119.97),
    "乌鲁木齐": (43.83, 87.62), "拉萨": (29.65, 91.14), "银川": (38.49, 106.23),
    "西宁": (36.62, 101.78), "呼和浩特": (40.84, 111.75), "三亚": (18.25, 109.51),
}

# 港澳台按规范显示名输出（查询时仍接受常用写法）
_CITY_DISPLAY = {
    "香港": "中国香港", "澳门": "中国澳门", "台北": "中国台湾·台北",
    "高雄": "中国台湾·高雄", "台中": "中国台湾·台中",
}

# WMO 天气代码 → 中文
_WMO_TEXT = {
    0: "晴", 1: "晴间多云", 2: "多云", 3: "阴",
    45: "有雾", 48: "雾凇", 51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    56: "冻毛毛雨", 57: "强冻毛毛雨", 61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨", 71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "阵雨", 82: "强阵雨", 85: "小阵雪", 86: "强阵雪",
    95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "雷阵雨伴大冰雹",
}


def _looks_like_weather(query: str) -> bool:
    return any(kw in query for kw in _WEATHER_KEYWORDS)


def _find_city(query: str) -> tuple[str, float, float] | None:
    """
    从查询里定位城市：先匹配内置城市表（长名优先），再退回地理编码接口。
    返回 (显示名, 纬度, 经度)；识别不出城市时返回 None（走通用搜索）。
    """
    hits = [name for name in _CITY_COORDS if name in query]
    if not hits:
        hits = [name for name in _CITY_DISPLAY if name in query]
    if hits:
        name = max(hits, key=len)
        if name in _CITY_COORDS:
            lat, lon = _CITY_COORDS[name]
            return _CITY_DISPLAY.get(name, name), lat, lon
        # 港澳台：内置表里没有就交给地理编码
        geo = _geocode(name)
        return (geo[0], geo[1], geo[2]) if geo else None

    # 去掉天气/疑问类词，剩下的中文片段拿去地理编码
    cleaned = query
    for word in ("今天", "明日", "明天", "现在", "当前", "最新", "一下", "怎么样",
                 "如何", "查询", "查一下", "多少度", "的", "呢", "吗", "？", "?", "，"):
        cleaned = cleaned.replace(word, " ")
    for word in _WEATHER_KEYWORDS:
        cleaned = cleaned.replace(word, " ")
    candidates = [t for t in re.split(r"[\s,，、.。!！?？]+", cleaned) if 2 <= len(t) <= 8]
    for candidate in candidates:
        geo = _geocode(candidate)
        if geo:
            return geo
    return None


@cached(ttl_seconds=86400)      # 城市坐标基本不变，缓存一天
def _geocode(name: str) -> tuple[str, float, float] | None:
    """城市名 → (规范名, 纬度, 经度)。Open-Meteo 地理编码，免费无需 Key。"""
    try:
        resp = requests.get(METEO_GEOCODE, params={
            "name": name, "count": 1, "language": "zh", "format": "json",
        }, headers=UA, timeout=8)
        resp.raise_for_status()
        results = (resp.json() or {}).get("results") or []
    except (requests.RequestException, ValueError):
        return None
    if not results:
        return None
    item = results[0]
    display = item.get("name") or name
    return (display, float(item["latitude"]), float(item["longitude"]))


@cached(ttl_seconds=600)        # 实况 10 分钟内可复用
def _fetch_weather(lat: float, lon: float) -> dict:
    """Open-Meteo 实况 + 今日预报（原始 JSON，失败会抛异常，不进缓存）。"""
    resp = requests.get(METEO_FORECAST, params={
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                   "weather_code,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "Asia/Shanghai", "forecast_days": 1,
    }, headers=UA, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _weather_answer(city: str, lat: float, lon: float) -> str:
    try:
        data = _fetch_weather(lat, lon)
    except requests.Timeout as exc:
        raise TimeoutError(f"天气服务请求超时: {city}") from exc
    except requests.RequestException as exc:
        raise ConnectionError(f"天气服务请求失败: {exc}") from exc

    current = data.get("current") or {}
    daily = data.get("daily") or {}
    temp = current.get("temperature_2m")
    feels = current.get("apparent_temperature")
    humidity = current.get("relative_humidity_2m")
    wind = current.get("wind_speed_10m")
    code = current.get("weather_code")
    desc = _WMO_TEXT.get(int(code), "未知") if code is not None else "未知"
    stamp = current.get("time") or ""

    lines = [f"【实况】{city} {temp}°C（体感 {feels}°C），{desc}"]
    extra = []
    if humidity is not None:
        extra.append(f"湿度 {humidity}%")
    if wind is not None:
        extra.append(f"风速 {wind} km/h")
    if extra:
        lines.append("　" + "，".join(extra))

    highs = (daily.get("temperature_2m_max") or [None])[0]
    lows = (daily.get("temperature_2m_min") or [None])[0]
    rain = (daily.get("precipitation_probability_max") or [None])[0]
    if highs is not None and lows is not None:
        today = f"【今日】最高 {highs}°C / 最低 {lows}°C"
        if rain is not None:
            today += f"，降水概率 {rain}%"
        lines.append(today)

    lines.append(f"数据源：Open-Meteo（{stamp} 北京时间，实况+当日预报）")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 通用搜索：Bing 中国站 RSS
# ---------------------------------------------------------------------------
@cached(ttl_seconds=600)        # 相同查询 10 分钟内命中缓存；失败不进缓存
def _fetch_bing_rss(query: str) -> str:
    resp = requests.get(BING_RSS, params={"q": query, "format": "rss"},
                        headers=UA, timeout=12)
    resp.raise_for_status()
    return resp.text


def _bing_results(query: str, max_results: int) -> list[tuple[str, str, str]]:
    """解析 Bing RSS：返回 [(标题, 链接, 摘要)]。"""
    try:
        root = ET.fromstring(_fetch_bing_rss(query))
    except ET.ParseError as exc:
        raise ValueError(f"搜索结果不是合法 XML，可能被网络拦截: {exc}") from exc
    items = root.findall(".//item")[:max_results]
    results = []
    for item in items:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = re.sub(r"\s+", " ", (item.findtext("description") or "")).strip()
        if title:
            results.append((title, link, desc))
    return results


# ---------------------------------------------------------------------------
# 兜底：DuckDuckGo 即时问答（国内通常不可达）
# ---------------------------------------------------------------------------
def _ddg_answer(query: str, max_results: int) -> list[str]:
    resp = requests.get(DDG_ENDPOINT, params={
        "q": query, "format": "json", "no_html": 1,
        "no_redirect": 1, "skip_disambig": 1,
    }, headers=UA, timeout=6)       # 国内会超时，别等 15 秒
    resp.raise_for_status()
    data = resp.json()

    lines: list[str] = []
    abstract = (data.get("AbstractText") or "").strip()
    if abstract:
        lines.append(f"【摘要】{abstract}")
        if data.get("AbstractURL"):
            lines.append(f"来源: {data['AbstractURL']}")
    for topic in data.get("RelatedTopics", []):
        if len(lines) >= max_results:
            break
        if isinstance(topic, dict) and topic.get("Text"):
            lines.append(f"- {topic['Text'].strip()}\n  {topic.get('FirstURL', '')}")
    return lines


# ---------------------------------------------------------------------------
# 对外工具
# ---------------------------------------------------------------------------
class SearchWebArgs(BaseModel):
    query: str = Field(..., description="搜索关键词，尽量具体")
    max_results: int = Field(5, ge=1, le=10, description="最多返回条数")


@register
@tool(args_schema=SearchWebArgs)
def search_web(query: str, max_results: int = 5) -> str:
    """联网搜索资料，返回摘要与相关链接。需要最新信息（新闻、天气、价格、
    版本等）时使用。天气类查询会直接返回实况与当日预报。"""
    if _circuit_blocked():
        return (
            "搜索服务暂时不可用（此前连续失败，正在冷却，约 1 分钟后自动恢复）。"
            "可以先回答其他问题，稍后再试；若必须联网核实，请说明「未能联网核实」。"
        )

    # 1) 天气：走结构化数据源，比搜索结果靠谱得多
    if _looks_like_weather(query):
        city = _find_city(query)
        if city:
            name, lat, lon = city
            try:
                answer = _weather_answer(name, lat, lon)
            except (ConnectionError, TimeoutError):
                _record_result(False)
                return (f"天气服务暂时不可用（网络请求失败），未能获取{name}的实时天气。"
                        "请稍后再试，或直接告诉用户当前无法联网核实。")
            except Exception:
                _record_result(False)
                return (f"天气数据解析失败，未能获取{name}的实时天气。"
                        "请告诉用户暂时无法联网核实。")
            _record_result(True)
            return answer

    # 2) 通用搜索：Bing RSS
    results: list[tuple[str, str, str]] = []
    bing_error = ""
    try:
        results = _bing_results(query, max_results)
    except Exception as exc:        # 连接失败 / 解析失败 / 被拦截
        bing_error = f"{type(exc).__name__}: {exc}"
    if results:
        _record_result(True)
        return "\n".join(
            f"- {title}\n  {link}\n  {desc[:200]}" for title, link, desc in results)

    # 3) 兜底：DDG（国内通常也失败，快速放弃并给出说明）
    try:
        lines = _ddg_answer(query, max_results)
        if lines:
            _record_result(True)
            return "\n".join(lines)
    except Exception:
        pass

    _record_result(False)
    return (
        f"没有检索到与「{query}」相关的结果（搜索源：Bing 中国站"
        f"{'，错误：' + bing_error[:80] if bing_error else ''}）。\n"
        "可以换更通用的关键词重试，或直接告诉用户当前无法联网核实。"
    )
