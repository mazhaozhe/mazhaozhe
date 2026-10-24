"""
agent/api.py —— 对外统一接口（门面 / Facade）
==============================================
这是给"调用方"用的简洁接口层：CLI、未来的网页前端、或其他 Python 代码，
都只跟这里的几个函数打交道，不必关心底层 ModelManager / AgentRuntime 的细节。

设计目标（对应需求）：
    - 查询可用模型列表      → list_models()
    - 自由选择 / 切换模型    → switch_model(model_id)
    - 切换后无需重启生效     → 内部走 AgentRuntime 热切换
    - 查看当前模型           → current_model()
    - 改完目录热重载         → reload_models()
    - 错误处理               → 抛出 agent.models 里的语义化异常
    - 可扩展                 → register_provider() / 编辑 config/models.json

典型用法（CLI 或 Web 后端里）：
    import agent.api as api
    api.init_runtime()                       # 启动时初始化一次
    api.list_models()                        # 模型列表
    api.switch_model("deepseek-reasoner")    # 切换（热生效）
    agent = api.get_agent()                  # 拿当前 Agent 去跑
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.models import ModelSpec  # noqa: F401  （供调用方构造新模型用）
from agent.runtime import AgentRuntime

_runtime: Optional[AgentRuntime] = None
_init_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 生命周期
# ---------------------------------------------------------------------------
def init_runtime(catalog_path: Optional[Path] = None,
                 default_model: Optional[str] = None) -> AgentRuntime:
    """
    初始化全局运行时（幂等：重复调用直接返回已存在的实例）。
    应在服务启动时调用一次。
    """
    global _runtime
    with _init_lock:
        if _runtime is None:
            _runtime = AgentRuntime(catalog_path=catalog_path,
                                    default_model=default_model)
        return _runtime


def _require_runtime() -> AgentRuntime:
    if _runtime is None:
        raise RuntimeError("运行时未初始化，请先调用 init_runtime()。")
    return _runtime


# ---------------------------------------------------------------------------
# 模型查询 / 切换（核心接口）
# ---------------------------------------------------------------------------
def list_models() -> List[Dict[str, Any]]:
    """列出全部可用模型（含是否可用、是否当前）。不含密钥。"""
    return _require_runtime().list_models()


def current_model() -> Dict[str, Any]:
    """返回当前在用模型的信息。"""
    return _require_runtime().current_model()


def switch_model(model_id: str, ping: bool = False) -> Dict[str, Any]:
    """
    切换到指定模型，热生效、无需重启、会话记忆保留。

    成功返回新模型信息；失败抛出语义化异常：
        ModelNotFoundError  模型不存在
        ModelConfigError    配置问题（缺密钥 / 不支持工具调用）
        ModelBuildError     实例化失败（缺依赖等）
        ModelSwitchError    联通性验证失败（已回滚）
    """
    return _require_runtime().switch_model(model_id, ping=ping)


def reload_models() -> int:
    """热重载模型目录 config/models.json（改配置不用重启）。返回模型数。"""
    return _require_runtime().reload_models()


def discover_models() -> Dict[str, Any]:
    """
    从各已配置密钥的供应商自动发现全部可用模型并注册进目录。
    （启动时会自动跑一次；此命令用于手动刷新，或网络恢复后补拉。）
    返回 {provider: {"ok", "models", "new", "error"}} 报告。
    """
    return _require_runtime().discover_models()


# ---------------------------------------------------------------------------
# 拿当前 Agent 去跑
# ---------------------------------------------------------------------------
def get_agent() -> Any:
    """返回当前可用的 Agent 实例（切换模型后自动指向新实例）。"""
    return _require_runtime().agent


# ---------------------------------------------------------------------------
# 扩展口：注册新供应商构造器（全新厂商时调用一次即可）
# ---------------------------------------------------------------------------
def register_provider(name: str, builder: Callable[[ModelSpec, str], Any]) -> None:
    """
    注册一个新的模型供应商构造器。
    builder(spec, api_key) -> LangChain 聊天模型实例。
    之后在 config/models.json 里把某模型的 provider 设为 name 即可使用。
    """
    _require_runtime().model_manager.register_provider(name, builder)
