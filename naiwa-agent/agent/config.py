"""
agent/config.py —— 配置中心
============================
从项目根目录的 .env 加载配置（pydantic-settings，类型安全 + 自动校验）。
所有路径统一基于项目根目录解析，避免"在哪个目录启动"导致的相对路径错乱。
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（agent/config.py 的上两级）
BASE_DIR = Path(__file__).resolve().parent.parent

# 把 .env 真正读进进程环境变量（os.environ）。
# 这样模型管理器用 os.environ.get("XXX_API_KEY") 也能取到密钥——
# 否则 pydantic-settings 只把值读进 Settings 对象，环境变量里是空的。
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    """全部可配置项。字段名与 .env 中的键一一对应（大小写不敏感）。"""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",          # .env 里多余的键不报错
    )

    # ---- 密钥 ----
    deepseek_api_key: str = Field(default="", description="DeepSeek API Key")

    # ---- 模型 ----
    main_model: str = "deepseek-chat"
    reasoner_model: str = "deepseek-reasoner"
    deepseek_base_url: str = "https://api.deepseek.com"
    temperature: float = 0.2

    # ---- 安全阀 ----
    model_run_limit: int = 60          # 单次任务最大模型调用次数
    tool_run_limit: int = 100          # 单次任务最大工具调用次数

    # ---- 可靠性 ----
    tool_retries: int = 3              # 工具失败重试次数

    # ---- 限流 ----
    calls_per_minute: int = 60         # 每分钟工具调用上限
    max_concurrency: int = 4           # 最大并发

    # ---- 长对话上下文压缩（SummarizationMiddleware）----
    # 触发点优先按"模型窗口预算"动态计算（见 agent/core.py 的 _summarize_trigger）：
    #     窗口 - 最大输出 - 静态开销 - 安全余量
    # 只有当模型窗口未知（如自动发现且无 profile 的模型）时，才用下面的绝对阈值兜底。
    summarize_trigger_tokens: int = 16000   # 兜底：历史消息达到多少 token 触发压缩
    summarize_keep_messages: int = 20       # 压缩时保留最近多少条原文消息
    summarize_safety_margin: int = 8192     # 预算式触发点保留的安全余量（token）
    summarize_trim_tokens: int = 0          # 喂给摘要模型的历史上限；0 = 不裁剪（推荐，
                                            # 任务目标在历史最开头，裁剪会丢目标）

    # ---- RAG（可选）----
    rag_enabled: bool = False
    rag_top_k: int = 4
    chroma_dir: str = "data/chroma"

    # ---- 运行时文件 ----
    checkpoint_db: str = "data/checkpoint.sqlite"
    cache_db: str = "data/cache.sqlite"
    log_file: str = "data/agent.log"
    trace_file: str = "data/trace.log"   # 思考/工具过程完整流水（/trace 打开）

    # ---- 过程框 UI ----
    process_box_lines: int = 10          # 过程框内容行数（3~30 之间有效）

    # ------------------------------------------------------------------
    def abs_path(self, relative: str) -> Path:
        """把配置里的相对路径解析为项目根目录下的绝对路径，并确保父目录存在。"""
        p = Path(relative)
        if not p.is_absolute():
            p = BASE_DIR / p
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            # 目录建不出来（权限等）不阻断启动，后续写入时会给出更具体的错误
            pass
        return p

    def validate_runtime(self) -> list[str]:
        """
        启动前校验，返回问题列表（空列表=一切正常）。
        不直接抛异常，是为了让 main.py 能一次性把所有问题都提示给用户。
        """
        problems: list[str] = []

        if not self.deepseek_api_key or self.deepseek_api_key.startswith("sk-xxx"):
            problems.append(
                "未配置 DEEPSEEK_API_KEY。请在项目根目录的 .env 中填写真实密钥"
                "（申请：https://platform.deepseek.com）。"
            )

        if self.model_run_limit < 1 or self.tool_run_limit < 1:
            problems.append("MODEL_RUN_LIMIT / TOOL_RUN_LIMIT 必须 >= 1。")

        if self.calls_per_minute < 1:
            problems.append("CALLS_PER_MINUTE 必须 >= 1。")

        if self.summarize_trigger_tokens < 1 or self.summarize_keep_messages < 1:
            problems.append(
                "SUMMARIZE_TRIGGER_TOKENS / SUMMARIZE_KEEP_MESSAGES 必须 >= 1。")

        if self.summarize_trim_tokens < 0:
            problems.append("SUMMARIZE_TRIM_TOKENS 必须 >= 0（0 表示不裁剪）。")

        if sys.version_info >= (3, 14):
            problems.append(
                "当前 Python 3.14+ 部分依赖可能不兼容，建议使用 3.10~3.13。"
            )
        return problems


# 全局单例：其他模块直接 from agent.config import settings
settings = Settings()
