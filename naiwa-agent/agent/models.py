"""
agent/models.py —— 模型管理器（多模型灵活切换的核心）
======================================================
职责：
    1. 维护一份"可用模型目录"（ModelSpec 列表），支持从 JSON 动态加载 / 热重载；
    2. 用统一的"供应商构造器"把模型规格实例化为 LangChain 聊天模型；
    3. 提供 查询列表 / 获取当前 / 切换 三个核心操作，切换失败自动回滚；
    4. 全程线程安全，任何一步出错都抛出语义化异常，绝不让 Agent 处于"半切换"状态。

扩展方式（后续加新模型，零改代码）：
    - 加一家已有供应商的新模型  → 只改 config/models.json，加一条记录；
    - 加一家全新供应商          → 调用 register_provider("xxx", builder) 注册一个构造器即可。

被 agent/runtime.py 使用；对外暴露给 agent/api.py。
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.config import settings
from agent.observability import log_step


# ---------------------------------------------------------------------------
# 语义化异常：上层（CLI / Web）可据此给用户精确反馈
# ---------------------------------------------------------------------------
class ModelError(Exception):
    """模型相关错误的基类。"""
    pass


class ModelNotFoundError(ModelError):
    """请求的模型 id 不在目录里。"""
    def __init__(self, model_id: str, available: List[str]):
        self.model_id = model_id
        self.available = available
        super().__init__(
            f"模型 `{model_id}` 不存在。可用模型：{', '.join(available) or '(空)'}"
        )


class ModelConfigError(ModelError):
    """模型配置不完整（如缺 API Key）。"""
    pass


class ModelBuildError(ModelError):
    """实例化模型失败（依赖缺失 / 参数非法等）。"""
    pass


class ModelSwitchError(ModelError):
    """切换过程中校验失败，已回滚到旧模型。"""
    pass


# ---------------------------------------------------------------------------
# 模型规格：目录里一条模型的完整描述
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    """
    一个可选模型的全部信息。

    id            唯一标识，切换时用它（如 "deepseek-chat"）
    label         展示名（给人看）
    provider      供应商类型，决定用哪个构造器（如 "deepseek" / "openai-compatible"）
    model         供应商侧的真实模型名（如 "deepseek-chat" / "qwen-plus"）
    base_url      API 地址
    api_key_env   存放密钥的环境变量名（密钥不落盘到 JSON，只存"变量名"）
    temperature   采样温度
    kind          "chat" 普通对话 / "reasoner" 深度推理
    supports_tools 是否支持工具调用（Agent 必需，False 的模型切换时会被拦截）
    supports_vision 是否支持图片（多模态）输入。自动发现的模型按模型名猜测，
                  可在目录里显式声明覆盖；决定上传照片时是走视觉输入还是降级提示
    requires_key  是否需要密钥（本地 Ollama 之类可为 False）
    description   一句话说明
    pricing       价格（元/百万token）：{"input_per_mtok": .., "output_per_mtok": ..}
                  用于 token 成本估算；本地免费模型可不填
    max_input_tokens  上下文窗口大小（token）。注入到模型 profile，供
                  SummarizationMiddleware 按窗口比例计算压缩触发点；
                  不填则模型用内置 profile（部分新模型自带），再没有则走绝对阈值兜底
    max_output_tokens 单次回复最大输出 token。不填用供应商默认值
                  （如 deepseek-chat 默认 4K，显式设 8192 可用满 8K 上限）
    extra         供应商私有参数，原样透传给构造器
    """
    id: str
    label: str
    provider: str
    model: str
    base_url: str = ""
    api_key_env: str = ""
    temperature: float = 0.2
    kind: str = "chat"
    supports_tools: bool = True
    supports_vision: bool = False
    requires_key: bool = True
    description: str = ""
    pricing: Dict[str, float] = field(default_factory=dict)
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSpec":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        # 目录里多出来的键收进 extra，避免扩展字段导致解析失败
        extra = dict(kwargs.get("extra", {}))
        for k, v in data.items():
            if k not in known:
                extra[k] = v
        kwargs["extra"] = extra
        return cls(**kwargs)

    def to_public_dict(self, available: bool, current: bool) -> Dict[str, Any]:
        """对外暴露的安全视图（绝不含密钥）。"""
        return {
            "id": self.id,
            "label": self.label,
            "provider": self.provider,
            "model": self.model,
            "kind": self.kind,
            "supports_tools": self.supports_tools,
            "supports_vision": self.supports_vision,
            "available": available,     # 密钥是否已配置（能不能用）
            "current": current,         # 是否为当前在用模型
            "description": self.description,
            "pricing": dict(self.pricing),   # 元/百万token，成本估算用
            "max_input_tokens": self.max_input_tokens,    # 上下文窗口（/context 展示）
            "max_output_tokens": self.max_output_tokens,  # 单次输出上限
        }


# ---------------------------------------------------------------------------
# 供应商构造器：把 ModelSpec + 密钥 → LangChain 聊天模型实例
# 全部懒加载（用到才 import），避免为切换而装一堆用不到的 SDK
# ---------------------------------------------------------------------------
def _window_kwargs(spec: ModelSpec) -> Dict[str, Any]:
    """
    把目录里声明的窗口/输出上限转成模型构造参数：

    - max_output_tokens → max_tokens：抬高单次回复上限
      （如 deepseek-chat 默认只给 4K，显式设 8192 才能用满 8K）；
    - max_input_tokens  → profile：SummarizationMiddleware 靠它按窗口比例
      算压缩触发点；模型自带 profile 时以目录声明为准（更可控）。

    两项都没声明时返回空 dict——不硬塞任何参数，保持供应商默认行为。
    """
    kwargs: Dict[str, Any] = {}
    if isinstance(spec.max_output_tokens, int) and spec.max_output_tokens > 0:
        kwargs["max_tokens"] = spec.max_output_tokens
    if isinstance(spec.max_input_tokens, int) and spec.max_input_tokens > 0:
        kwargs["profile"] = {"max_input_tokens": spec.max_input_tokens}
    return kwargs


def _build_deepseek(spec: ModelSpec, api_key: str):
    from langchain_deepseek import ChatDeepSeek
    return ChatDeepSeek(
        model=spec.model,
        api_key=api_key,
        api_base=spec.base_url or "https://api.deepseek.com",
        temperature=spec.temperature,
        max_retries=2,
        stream_usage=True,        # 流式输出时也在最后一个 chunk 回传真实 token 用量
        **{**_window_kwargs(spec),
           **anti_degeneration_kwargs(spec.model),
           **spec.extra},
    )


# 推理型模型不接受采样参数（DeepSeek reasoner / R1 / OpenAI o 系列 / thinking 变体）
_NO_PENALTY_HINTS = ("reasoner", "-r1", "o1", "o3", "o4", "thinking")


def anti_degeneration_kwargs(model_name: str = "") -> Dict[str, Any]:
    """
    反"复读机"采样参数。

    模型卡在决策瘫痪（反复权衡该不该执行）时，自回归输出会退化成
    「好。执行。好。执行。」的死循环——一次能白烧 3 万 tokens。
    frequency_penalty 按词频惩罚已出现过的 token，是最直接的压制手段；
    presence_penalty 轻推模型引入新内容。

    推理型模型不支持采样参数，一律不传（传了可能被接口拒绝）。
    """
    name = (model_name or "").lower()
    if any(hint in name for hint in _NO_PENALTY_HINTS):
        return {}
    kwargs: Dict[str, Any] = {}
    if settings.frequency_penalty:
        kwargs["frequency_penalty"] = settings.frequency_penalty
    if settings.presence_penalty:
        kwargs["presence_penalty"] = settings.presence_penalty
    return kwargs


def _build_openai_compatible(spec: ModelSpec, api_key: str):
    """
    OpenAI 兼容端点通用构造器。
    覆盖：OpenAI 官方、通义千问(兼容模式)、Kimi、本地 Ollama 等——
    只要对方提供 OpenAI 风格的 /v1 接口，就都能用这一个构造器。
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=spec.model,
        api_key=api_key or "not-needed",   # Ollama 等免密钥端点给个占位值
        base_url=spec.base_url,
        temperature=spec.temperature,
        max_retries=2,
        stream_usage=True,        # 流式输出时也在最后一个 chunk 回传真实 token 用量
        **{**_window_kwargs(spec),
           **anti_degeneration_kwargs(spec.model),
           **spec.extra},
    )


# 内置供应商注册表：provider 名 → 构造器
_BUILTIN_PROVIDERS: Dict[str, Callable[[ModelSpec, str], Any]] = {
    "deepseek": _build_deepseek,
    "openai": _build_openai_compatible,
    "openai-compatible": _build_openai_compatible,
    "ollama": _build_openai_compatible,     # Ollama 自带 OpenAI 兼容端点
}


# ---------------------------------------------------------------------------
# 模型自动发现：用 Key 调各家 /models 接口，拉出"这家全部可用模型"
# ---------------------------------------------------------------------------
def _guess_kind(model_id: str) -> str:
    """从模型名猜类型：带 reasoner/think/r1 字样的按深度推理处理。"""
    low = model_id.lower()
    if any(k in low for k in ("reasoner", "think", "thinking", "r1")):
        return "reasoner"
    return "chat"


def _guess_vision(model_id: str, patterns: list[str] | None = None) -> bool:
    """
    从模型名猜测是否支持图片（多模态）输入。

    命中 vision_patterns（见 agent/config.py）任一子串即视为支持，
    覆盖 qwen-vl-plus / moonshot-v1-*-vision / gpt-4o / gemini / claude-3 等
    常见命名。patterns 传 None 时用配置里的默认列表；目录里显式声明的
    supports_vision 始终优先（ModelSpec 字段本身即最终值）。
    """
    from agent.config import settings   # 延迟导入，避免模块加载顺序问题

    words = patterns if patterns is not None else settings.vision_pattern_list
    low = model_id.lower()
    return any(word and word in low for word in words)


def _fetch_provider_models(base_url: str, api_key: str,
                           timeout: float = 8.0) -> List[str]:
    """
    调 OpenAI 兼容端点的 GET {base_url}/models，返回模型 id 列表。
    DeepSeek / OpenAI / 通义(兼容模式) / Kimi 等都支持这个接口。失败抛异常。
    """
    import requests
    url = base_url.rstrip("/") + "/models"
    resp = requests.get(url,
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=timeout)
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", []) if m.get("id")]


# ---------------------------------------------------------------------------
# 模型管理器
# ---------------------------------------------------------------------------
class ModelManager:
    """
    管理"有哪些模型可用、当前在用哪个、怎么切换"。

    线程安全：所有会改动状态的操作都在同一把锁内完成。
    """

    def __init__(self, catalog_path: Optional[Path] = None):
        self._lock = threading.RLock()
        self._catalog_path = Path(catalog_path) if catalog_path else None
        self._specs: Dict[str, ModelSpec] = {}
        self._providers: Dict[str, Callable[[ModelSpec, str], Any]] = dict(_BUILTIN_PROVIDERS)
        self._default_id: Optional[str] = None
        self._current_id: Optional[str] = None
        self._current_model: Any = None   # 当前已实例化的聊天模型
        if self._catalog_path:
            self.reload()

    # ---- 目录加载 / 热重载 ----
    def reload(self) -> int:
        """重新读取 JSON 目录（改完配置不用重启进程）。返回加载到的模型数。"""
        with self._lock:
            if not self._catalog_path or not self._catalog_path.exists():
                raise ModelConfigError(
                    f"模型目录文件不存在：{self._catalog_path}。"
                    "请创建 config/models.json。"
                )
            try:
                raw = json.loads(self._catalog_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ModelConfigError(f"模型目录 JSON 解析失败：{exc}") from exc

            specs: Dict[str, ModelSpec] = {}
            for item in raw.get("models", []):
                spec = ModelSpec.from_dict(item)
                if not spec.id:
                    raise ModelConfigError("目录中存在缺少 id 的模型条目。")
                specs[spec.id] = spec

            self._specs = specs
            self._default_id = raw.get("default") or (next(iter(specs)) if specs else None)
            log_step("models_reloaded", count=len(specs), default=self._default_id)
            return len(specs)

    # ---- 扩展：运行时注册新模型 / 新供应商 ----
    def register_model(self, spec: ModelSpec) -> None:
        """运行时临时加入一个模型（不落盘）。便于动态扩展。"""
        with self._lock:
            self._specs[spec.id] = spec

    def register_provider(self, name: str, builder: Callable[[ModelSpec, str], Any]) -> None:
        """注册新供应商构造器。builder(spec, api_key) -> 聊天模型实例。"""
        with self._lock:
            self._providers[name] = builder

    # ---- 自动发现：一家 Key → 这家全部可用模型 ----
    def discover(self, timeout: float = 8.0,
                 cache_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        对每个"已配置密钥"的供应商调 GET /models，把返回的全部模型自动注册进目录。

        设计要点：
            - 静态目录(config/models.json)优先：id 已存在的不覆盖
              （人工标注的 label/价格/kind 得以保留）；
            - 新模型只进进程内目录 + 缓存文件，不改 models.json；
            - 网络失败时自动回退到缓存文件（上次发现的结果），离线也能列出；
            - 免密钥端点(Ollama)跳过（它的列表接口格式不同）。

        返回每个供应商的报告：{provider: {"ok", "models", "new", "error"}}
        """
        with self._lock:
            prov_info: Dict[str, Dict[str, str]] = {}
            for spec in self._specs.values():
                prov_info.setdefault(spec.provider, {
                    "base_url": spec.base_url,
                    "api_key_env": spec.api_key_env,
                })

        report: Dict[str, Any] = {}
        for provider, info in prov_info.items():
            base_url, api_key_env = info["base_url"], info["api_key_env"]
            key = os.environ.get(api_key_env, "").strip() if api_key_env else ""
            if not base_url or not key:
                report[provider] = {"ok": False, "models": [], "new": 0,
                                    "error": "未配置密钥，跳过"}
                continue
            try:
                ids = _fetch_provider_models(base_url, key, timeout=timeout)
            except Exception as exc:
                cached = self._load_discover_cache(provider, cache_path)
                note = f"（已用缓存 {len(cached)} 个）" if cached else "（无缓存）"
                report[provider] = {"ok": False, "models": cached, "new": 0,
                                    "error": f"{type(exc).__name__}: {exc} {note}"}
                continue
            new = 0
            with self._lock:
                for mid in ids:
                    if mid in self._specs:
                        continue                    # 静态条目优先，不覆盖
                    self._specs[mid] = ModelSpec(
                        id=mid, label=f"{provider} · {mid}", provider=provider,
                        model=mid, base_url=base_url, api_key_env=api_key_env,
                        temperature=0.2, kind=_guess_kind(mid),
                        supports_tools=True,
                        supports_vision=_guess_vision(mid),
                        description=f"自动发现（{provider} /models）",
                    )
                    new += 1
            report[provider] = {"ok": True, "models": ids, "new": new,
                                "error": None}
            self._save_discover_cache(provider, ids, cache_path)

        log_step("models_discovered",
                 **{p: {"ok": r["ok"], "new": r["new"]}
                    for p, r in report.items()})
        return report

    @staticmethod
    def _save_discover_cache(provider: str, ids: List[str],
                             cache_path: Optional[Path]) -> None:
        """把某供应商发现到的模型 id 落盘缓存（不含密钥），供离线启动用。"""
        if not cache_path:
            return
        try:
            path = Path(cache_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            data: Dict[str, Any] = {}
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("providers", {})[provider] = list(ids)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        except OSError:
            pass                        # 缓存写失败不影响功能

    def _load_discover_cache(self, provider: str,
                             cache_path: Optional[Path]) -> List[str]:
        """网络失败时从缓存恢复该供应商的模型列表，并注册进目录。"""
        if not cache_path or not Path(cache_path).exists():
            return []
        try:
            data = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            ids = data.get("providers", {}).get(provider, [])
        except (OSError, json.JSONDecodeError):
            return []
        with self._lock:
            info = next((s for s in self._specs.values()
                         if s.provider == provider), None)
            if info is None:
                return []
            for mid in ids:
                if mid not in self._specs:
                    self._specs[mid] = ModelSpec(
                        id=mid, label=f"{provider} · {mid}", provider=provider,
                        model=mid, base_url=info.base_url,
                        api_key_env=info.api_key_env, temperature=0.2,
                        kind=_guess_kind(mid), supports_tools=True,
                        supports_vision=_guess_vision(mid),
                        description=f"自动发现（{provider} · 缓存）",
                    )
        return list(ids)

    # ---- 查询 ----
    def _resolve_key(self, spec: ModelSpec) -> str:
        """按 api_key_env 从环境变量取密钥。免密钥模型返回空串。"""
        if not spec.requires_key:
            return ""
        key = os.environ.get(spec.api_key_env, "").strip() if spec.api_key_env else ""
        return key

    def _is_available(self, spec: ModelSpec) -> bool:
        return (not spec.requires_key) or bool(self._resolve_key(spec))

    def list_models(self) -> List[Dict[str, Any]]:
        """列出全部模型的安全视图（含是否可用 / 是否当前）。不含任何密钥。"""
        with self._lock:
            return [
                spec.to_public_dict(
                    available=self._is_available(spec),
                    current=(spec.id == self._current_id),
                )
                for spec in self._specs.values()
            ]

    def get_current(self) -> Dict[str, Any]:
        """返回当前模型的安全视图；尚未初始化时返回 None。"""
        with self._lock:
            if not self._current_id:
                return {}
            spec = self._specs[self._current_id]
            return spec.to_public_dict(available=True, current=True)

    @property
    def current_model(self) -> Any:
        """当前已实例化的聊天模型对象（未初始化返回 None）。"""
        with self._lock:
            return self._current_model

    @property
    def current_id(self) -> Optional[str]:
        with self._lock:
            return self._current_id

    # ---- 实例化一个模型（内部）----
    def _instantiate(self, spec: ModelSpec) -> Any:
        builder = self._providers.get(spec.provider)
        if builder is None:
            raise ModelBuildError(
                f"未知供应商 `{spec.provider}`。可用：{', '.join(self._providers)}；"
                "或调用 register_provider() 注册新供应商。"
            )
        api_key = self._resolve_key(spec)
        if spec.requires_key and not api_key:
            raise ModelConfigError(
                f"模型 `{spec.id}` 需要密钥，但环境变量 `{spec.api_key_env}` 未配置。"
                f"请在 .env 中填写 {spec.api_key_env}=你的密钥。"
            )
        try:
            return builder(spec, api_key)
        except ImportError as exc:
            raise ModelBuildError(
                f"实例化 `{spec.id}` 失败：缺少依赖（{exc}）。"
                f"供应商 `{spec.provider}` 可能需要额外安装对应 SDK。"
            ) from exc
        except Exception as exc:
            raise ModelBuildError(
                f"实例化 `{spec.id}` 失败：{type(exc).__name__}: {exc}"
            ) from exc

    # ---- 初始化（启动时选定默认模型）----
    def init_default(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """选定并实例化初始模型。默认用目录里的 default。"""
        target = model_id or self._default_id
        if not target:
            raise ModelConfigError("模型目录为空，无法初始化默认模型。")
        return self.switch(target)

    # ---- 切换（核心）----
    def switch(self, model_id: str, ping: bool = False) -> Dict[str, Any]:
        """
        切换到指定模型，立即生效（无需重启）。

        流程：
            1. 校验存在性      → 不存在抛 ModelNotFoundError
            2. 校验工具调用能力 → 不支持则抛 ModelConfigError（Agent 跑不了）
            3. 实例化新模型     → 失败抛 ModelConfigError/ModelBuildError
            4. (可选) ping 联通性 → 失败抛 ModelSwitchError 并回滚
            5. 原子替换当前模型  → 全程在锁内，旧模型作为后备

        ping=True 会真实调一次 API 验证可用（会产生极少 token 消耗），默认关闭。
        任何一步失败都不会污染"当前模型"，保证 Agent 始终可用。
        """
        with self._lock:
            # 1. 存在性
            spec = self._specs.get(model_id)
            if spec is None:
                raise ModelNotFoundError(model_id, list(self._specs))

            # 已是当前模型：直接返回，不做无谓重建
            if model_id == self._current_id and self._current_model is not None:
                return spec.to_public_dict(available=True, current=True)

            # 2. 能力校验：Agent 依赖工具调用
            if not spec.supports_tools:
                raise ModelConfigError(
                    f"模型 `{model_id}` 声明为不支持工具调用（supports_tools=false），"
                    "无法驱动 Agent。请换一个支持工具调用的模型。"
                )

            # 3. 实例化（在替换当前模型之前先建好，失败则不影响现状）
            new_model = self._instantiate(spec)

            # 4. 可选联通性验证
            if ping:
                try:
                    new_model.invoke("ping")
                except Exception as exc:
                    raise ModelSwitchError(
                        f"切换到 `{model_id}` 失败（联通性验证未通过）："
                        f"{type(exc).__name__}: {exc}。已保留原模型 "
                        f"`{self._current_id}`。"
                    ) from exc

            # 5. 原子替换
            old_id = self._current_id
            self._current_model = new_model
            self._current_id = model_id
            log_step("model_switched", from_id=old_id, to_id=model_id,
                     provider=spec.provider, model=spec.model)
            return spec.to_public_dict(available=True, current=True)
