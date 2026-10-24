"""
agent/runtime.py —— Agent 运行时（支撑模型热切换）
====================================================
把"模型管理器 + 可复用的会话记忆 + Agent 实例"封装成一个运行时对象。

为什么需要它：
    create_agent() 一旦组装好，模型就"焊死"在里面了。
    要做到"切换模型不重启"，就得在切换时用新模型**重建** Agent——
    但记忆（checkpointer）必须**复用**，否则切个模型上下文就丢了。

    AgentRuntime 就是干这个的：
      - 持有唯一的 checkpointer（所有重建共用，记忆贯穿始终）；
      - 切换模型 → 用新模型重建 Agent → 原子替换引用；
      - 切换失败 → 旧 Agent 原封不动，服务不受影响。

对外只暴露三个属性/方法：agent（当前可用 Agent）、model_manager、switch_model()。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from agent.config import BASE_DIR
from agent.core import build_agent, build_checkpointer
from agent.models import ModelManager
from agent.observability import log_step

# 模型目录默认位置：项目根/config/models.json
DEFAULT_CATALOG = BASE_DIR / "config" / "models.json"
# 自动发现的模型列表缓存（网络不可用时离线恢复，不含密钥）
DISCOVER_CACHE = BASE_DIR / "data" / "discovered_models.json"


class AgentRuntime:
    """
    Agent 运行时：一个可热切换模型的 Agent 持有者。

    用法：
        runtime = AgentRuntime()                # 按目录默认模型启动
        runtime.agent                            # 拿去 stream_run / astream
        runtime.switch_model("qwen-plus")        # 热切换，记忆保留
        runtime.discover_models()                # 拉取各家全部可用模型
    """

    def __init__(self, catalog_path: Optional[Path] = None,
                 default_model: Optional[str] = None):
        self._lock = threading.RLock()
        self.model_manager = ModelManager(catalog_path or DEFAULT_CATALOG)
        # 会话记忆只建一次，之后所有模型重建都复用它（切换不丢上下文）
        self._checkpointer = build_checkpointer()
        self._agent: Any = None

        # 初始化默认模型并建首个 Agent
        self.model_manager.init_default(default_model)
        self._rebuild_locked()

        # 启动时自动发现：一家 Key → 这家全部可用模型（失败静默，不拖慢启动）
        if os.environ.get("AUTO_DISCOVER", "1") != "0":
            try:
                report = self.discover_models(timeout=6.0)
                for provider, r in report.items():
                    if r.get("ok") and r.get("new"):
                        log_step("auto_discover_added", provider=provider,
                                 new=r["new"])
            except Exception as exc:
                log_step("auto_discover_failed", error=str(exc))

    def discover_models(self, timeout: float = 8.0) -> Dict[str, Any]:
        """从各已配置密钥的供应商自动发现全部可用模型并注册。返回发现报告。"""
        return self.model_manager.discover(timeout=timeout,
                                           cache_path=DISCOVER_CACHE)

    # ---- 内部：用当前模型重建 Agent ----
    def _rebuild_locked(self) -> None:
        """用 model_manager 里的当前模型重建 Agent（复用 checkpointer）。需在锁内调用。"""
        current = self.model_manager.get_current()
        self._agent = build_agent(
            model=self.model_manager.current_model,
            checkpointer=self._checkpointer,
            model_label=current.get("model"),
        )
        log_step("runtime_rebuilt", model_id=current.get("id"),
                 model=current.get("model"))

    # ---- 对外：当前可用 Agent ----
    @property
    def agent(self) -> Any:
        """当前 Agent 实例。切换模型后此引用自动指向新实例。"""
        with self._lock:
            return self._agent

    # ---- 对外：热切换 ----
    def switch_model(self, model_id: str, ping: bool = False) -> Dict[str, Any]:
        """
        切换模型，立即生效、无需重启、记忆保留。

        先在 ModelManager 层完成校验与实例化（失败会抛语义化异常且不影响现状），
        成功后才重建 Agent 并原子替换——因此旧 Agent 在任何失败下都安然无恙。

        返回新模型的公开信息字典。
        """
        with self._lock:
            # 若切的就是当前模型，直接返回，避免无谓重建
            if model_id == self.model_manager.current_id and self._agent is not None:
                return self.model_manager.get_current()

            info = self.model_manager.switch(model_id, ping=ping)  # 失败在此抛出
            self._rebuild_locked()                                  # 成功后重建
            return info

    # ---- 对外：便捷查询（委托给 model_manager）----
    def list_models(self) -> list[Dict[str, Any]]:
        return self.model_manager.list_models()

    def current_model(self) -> Dict[str, Any]:
        return self.model_manager.get_current()

    def reload_models(self) -> int:
        """热重载模型目录（改了 config/models.json 不用重启）。"""
        return self.model_manager.reload()
