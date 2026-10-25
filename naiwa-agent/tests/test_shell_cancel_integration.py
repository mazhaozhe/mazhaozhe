"""
tests/test_shell_cancel_integration.py —— 「停止任务杀进程树」全链路测试
=====================================================================
复现用户截图场景：模型发起 run_shell（长命令执行中）→ 用户点「停止」
→ 取消必须迅速生效（不能干等 timeout），且杀掉整棵进程树。

链路：create_agent（真 LangGraph + 真 SQLite checkpointer）
      → astream → ToolNode 执行 run_shell（asyncio 子进程）
      → 取消 → run_shell 的 CancelledError 分支杀进程树。

同步 subprocess.run 实现在这个测试下的表现：取消后干等 timeout，
且 flag 文件最终被写下（子进程还活着）——即「卡住」的根因。
"""

import asyncio
import sys
import time
from pathlib import Path

import aiosqlite
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from tools.coding import run_shell


class ToolCallingFakeModel(BaseChatModel):
    """第一轮就返回一个 run_shell 工具调用的假模型（之后不会被再调用——任务会被取消）。"""

    command: str = ""
    timeout: int = 60

    @property
    def _llm_type(self) -> str:
        return "tool-calling-fake"

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        """create_agent 会调 bind_tools；假模型直接返回自身。"""
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessageChunk(
            content="",
            tool_calls=[{"name": "run_shell",
                         "args": {"command": self.command, "timeout": self.timeout},
                         "id": "call-1"}]))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(
            content="",
            tool_calls=[{"name": "run_shell",
                         "args": {"command": self.command, "timeout": self.timeout},
                         "id": "call-1"}]))


async def _drain(stream) -> None:
    async for _ in stream:
        pass


def test_stop_task_kills_shell_process_tree(tmp_path):
    """取消 astream 任务：快速返回 + 子进程树被杀 + 历史不残留坏消息。"""
    flag = tmp_path / "stop_orphan.txt"
    cmd = (f'"{sys.executable}" -c '
           f'"import time; time.sleep(3); open(r\'{flag}\', \'w\').close()"')

    async def scenario():
        conn = aiosqlite.connect(str(tmp_path / "checkpoint.sqlite"))
        agent = create_agent(
            model=ToolCallingFakeModel(command=cmd, timeout=60),
            tools=[run_shell],
            checkpointer=AsyncSqliteSaver(conn),
        )
        config = {"configurable": {"thread_id": "t-stop"}}
        stream = agent.astream(
            {"messages": [("user", "跑一个长命令")]}, config=config,
            stream_mode=["updates"],
        )
        task = asyncio.create_task(_drain(stream))
        await asyncio.sleep(1.5)          # 等模型返回 tool_calls 且子进程跑起来
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # 取消后历史里应留下孤儿 tool_calls（ assistant 声明了调用但没有
        # tool 回复）——history_repair 的存在正是为了清理它
        state = await agent.aget_state(config)
        from agent.history_repair import find_orphans
        orphans = find_orphans(state.values["messages"])
        assert orphans, "取消后应留下孤儿 tool_calls（由 history_repair 清理）"

        await conn.close()

    start = time.monotonic()
    asyncio.run(scenario())
    elapsed = time.monotonic() - start
    assert elapsed < 5, f"取消应在数秒内完成，实际 {elapsed:.1f}s（同步实现会干等 timeout）"

    time.sleep(4)                        # 若子进程还活着，此刻已经写了文件
    assert not flag.exists(), "取消后子进程仍在执行（进程树没杀干净）"
