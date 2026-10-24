"""
tools/search.py —— 联网搜索工具
================================
使用 DuckDuckGo 即时问答接口（无需 API Key）。
结果用于"查资料 / 核实信息"场景；接口只提供摘要与相关主题，
不是完整网页抓取，够用且稳定。

失败处理：网络错误抛 ConnectionError（交给重试中间件），
查询无结果返回明确提示（避免模型编造答案）。
"""

import requests
from langchain.tools import tool
from pydantic import BaseModel, Field

from memory.cache import cached
from tools.registry import register

DDG_ENDPOINT = "https://api.duckduckgo.com/"
HEADERS = {"User-Agent": "GeneralTaskAgent/1.0 (+https://github.com/)"}

# 失败熔断：搜索源（网关/代理）持续故障时，避免每条任务都白等完整超时。
# 连续失败达到阈值后，本轮进程内直接返回不可用说明，不再发起真实请求。
_FAIL_STREAK = 0
_CIRCUIT_OPEN = False
_CIRCUIT_THRESHOLD = 2


class SearchWebArgs(BaseModel):
    query: str = Field(..., description="搜索关键词，尽量具体")
    max_results: int = Field(5, ge=1, le=10, description="最多返回条数")


@cached(ttl_seconds=600)   # 相同查询 10 分钟内命中缓存，不重复请求
def _search_raw(query: str, max_results: int) -> str:
    """实际执行搜索（被缓存装饰器包裹）。"""
    try:
        resp = requests.get(
            DDG_ENDPOINT,
            params={
                "q": query,
                "format": "json",
                "no_html": 1,
                "no_redirect": 1,
                "skip_disambig": 1,
            },
            headers=HEADERS,
            timeout=15,
        )
        if resp.status_code >= 500:
            # 网关/代理故障（502/503 等）几秒内不会自愈。
            # raise 会进 ToolRetryMiddleware 的指数退避（1s+2s+4s），
            # 模型拿错误消息再调一次又是一轮，实测单条任务因此被拖到近百秒。
            # 改为直接返回说明文本：跳过工具层重试，让模型立刻换路径。
            return (
                f"搜索接口暂时不可用（HTTP {resp.status_code}，可能是网络代理故障）。"
                "请改用其他途径获取信息（本地资料、命令行查询），"
                "并在回答中注明「未能联网核实」。"
            )
        resp.raise_for_status()
    except requests.Timeout as exc:
        raise TimeoutError(f"搜索请求超时（15s）: {query}") from exc
    except requests.RequestException as exc:
        raise ConnectionError(f"搜索请求失败: {exc}") from exc

    try:
        data = resp.json()
    except ValueError as exc:
        raise ValueError("搜索接口返回了非 JSON 内容，可能被网络拦截。") from exc

    lines: list[str] = []

    # 1) 直接答案
    abstract = (data.get("AbstractText") or "").strip()
    if abstract:
        lines.append(f"【摘要】{abstract}")
        if data.get("AbstractURL"):
            lines.append(f"来源: {data['AbstractURL']}")

    # 2) 相关主题（取前 max_results 条）
    for topic in data.get("RelatedTopics", []):
        if len(lines) >= max_results + (1 if abstract else 0):
            break
        if isinstance(topic, dict) and topic.get("Text"):
            text = topic["Text"].strip()
            url = topic.get("FirstURL", "")
            lines.append(f"- {text}\n  {url}")

    if not lines:
        return (
            f"没有检索到与「{query}」直接相关的结果。"
            "可以换更通用的关键词重试，或直接告诉用户数据不足。"
        )
    return "\n".join(lines)


@register
@tool(args_schema=SearchWebArgs)
def search_web(query: str, max_results: int = 5) -> str:
    """联网搜索资料（DuckDuckGo），返回摘要与相关链接。需要最新信息时使用。
    注意：搜索源连续失败会触发熔断，之后请改用本地资料或命令行查询。"""
    global _FAIL_STREAK, _CIRCUIT_OPEN
    if _CIRCUIT_OPEN:
        return (
            "搜索服务已熔断（此前连续请求失败，短时间内不会恢复）。"
            "本轮对话请改用其他信息源（本地文件、命令行查询），"
            "并在回答中注明「未能联网核实」。"
        )
    try:
        result = _search_raw(query, max_results)
    except Exception:
        # 超时 / 连接失败也计入熔断 streak
        _FAIL_STREAK += 1
        if _FAIL_STREAK >= _CIRCUIT_THRESHOLD:
            _CIRCUIT_OPEN = True
        raise
    if result.startswith("搜索接口暂时不可用"):
        _FAIL_STREAK += 1
        if _FAIL_STREAK >= _CIRCUIT_THRESHOLD:
            _CIRCUIT_OPEN = True
    else:
        _FAIL_STREAK = 0
    return result
