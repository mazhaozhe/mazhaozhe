"""
tests/test_coding_tools.py —— 代码开发工具测试
===============================================
用 pytest 的 tmp_path 临时目录测试文件读写/编辑/命令执行，
不污染真实文件系统。

run_shell 是异步工具（asyncio 子进程），测试统一走 ainvoke；
重点回归「取消/超时杀整棵进程树」——同步实现下点「停止」杀不掉
子进程，会留着 pip 之类在后台跑满 timeout，表现为卡住。
"""

import asyncio
import sys
import time

import pytest

from tools.coding import edit_file, list_dir, read_file, run_shell, write_file


def test_write_and_read_roundtrip(tmp_path):
    """写入 → 读回，内容一致。"""
    target = tmp_path / "sub" / "a.txt"   # sub 不存在，应自动创建
    result = write_file.invoke({"path": str(target), "content": "你好 Agent"})
    assert "已写入" in result
    assert read_file.invoke({"path": str(target)}) == "你好 Agent"


def test_read_missing_file_raises(tmp_path):
    """读不存在的文件 → FileNotFoundError（中间件会转成重试提示）。"""
    with pytest.raises(FileNotFoundError):
        read_file.invoke({"path": str(tmp_path / "nope.txt")})


def test_edit_file_exact_replace(tmp_path):
    """精确替换成功。"""
    target = tmp_path / "code.py"
    write_file.invoke({"path": str(target), "content": "def f():\n    return 1\n"})
    result = edit_file.invoke({
        "path": str(target),
        "old_text": "return 1",
        "new_text": "return 2",
    })
    assert "已替换" in result
    assert "return 2" in read_file.invoke({"path": str(target)})


def test_edit_file_no_match_raises(tmp_path):
    """原文不存在 → 明确报错并提示先 read_file 核对。"""
    target = tmp_path / "x.txt"
    write_file.invoke({"path": str(target), "content": "hello"})
    with pytest.raises(ValueError, match="找到要替换的原文"):
        edit_file.invoke({"path": str(target), "old_text": "world", "new_text": "!"})


def test_list_dir(tmp_path):
    """目录浏览正常。"""
    write_file.invoke({"path": str(tmp_path / "a.py"), "content": "x=1"})
    listing = list_dir.invoke({"path": str(tmp_path), "pattern": "*.py"})
    assert "a.py" in listing


def test_run_shell_echo():
    """执行 echo 命令并拿到输出。"""
    out = asyncio.run(run_shell.ainvoke({"command": "echo hello_agent", "timeout": 10}))
    assert "hello_agent" in out and "exit_code=0" in out


def test_run_shell_dangerous_blocked():
    """危险命令必须被安全策略拦截（安全阀）。"""
    with pytest.raises(PermissionError, match="安全策略"):
        asyncio.run(run_shell.ainvoke({"command": "shutdown /s", "timeout": 5}))


def test_run_shell_timeout():
    """
    超时命令不抛异常，而是返回结构化结果（exit_code=timeout）。
    抛异常会被 ToolRetryMiddleware 重试 3 次、模型再重试多轮——一条
    timeout=240 的命令能放大成几十分钟等待，这是"卡住"的根因之一。
    """
    out = asyncio.run(run_shell.ainvoke({"command": "ping -n 10 127.0.0.1", "timeout": 2}))
    assert "exit_code=timeout" in out
    assert "超过 2 秒" in out


def test_run_shell_uses_project_venv_python():
    """
    run_shell 里的 python 必须解析到项目 .venv——否则模型 pip install
    装进系统 Python，Agent 自己（跑在 .venv）永远用不上，反复装、反复失败。
    """
    out = asyncio.run(run_shell.ainvoke({
        "command": 'python -c "import sys; print(sys.executable)"', "timeout": 20,
    }))
    assert "exit_code=0" in out
    assert ".venv" in out.lower(), f"python 未指向项目 .venv：{out}"


def test_run_shell_utf8_output():
    """Windows 控制台默认 GBK：chcp 65001 + PYTHONUTF8=1 后中文输出不应乱码。"""
    out = asyncio.run(run_shell.ainvoke({
        "command": f'"{sys.executable}" -c "print(\'中文输出测试\')"',
        "timeout": 10,
    }))
    assert "exit_code=0" in out
    assert "中文输出测试" in out


def test_run_shell_timeout_kills_process_tree(tmp_path):
    """
    超时终止必须杀整棵进程树：shell 下真正干活的子进程（这里是 python）
    不能活过 timeout——否则它会在后台继续跑，下次调用又被它拖住。
    """
    flag = tmp_path / "timeout_orphan.txt"

    async def scenario():
        cmd = (f'"{sys.executable}" -c '
               f'"import time; time.sleep(3); open(r\'{flag}\', \'w\').close()"')
        out = await run_shell.ainvoke({"command": cmd, "timeout": 1})
        assert "exit_code=timeout" in out

    asyncio.run(scenario())
    time.sleep(4)                       # 若子进程还活着，此刻已经写了文件
    assert not flag.exists(), "超时后子进程仍在执行（进程树没杀干净）"


def test_run_shell_cancel_kills_process_tree(tmp_path):
    """
    核心回归：用户点「停止」取消任务时，run_shell 必须立刻杀掉进程树并让出。
    同步实现下取消杀不掉线程里的 subprocess.run，会干等 timeout（数十秒），
    事件循环也被占住——这就是「每次到这个工具就卡住」的根因。
    """
    flag = tmp_path / "cancel_orphan.txt"

    async def scenario():
        cmd = (f'"{sys.executable}" -c '
               f'"import time; time.sleep(2); open(r\'{flag}\', \'w\').close()"')
        task = asyncio.create_task(run_shell.ainvoke({"command": cmd, "timeout": 30}))
        await asyncio.sleep(0.8)        # 让 python 子进程真起来
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    start = time.monotonic()
    asyncio.run(scenario())
    elapsed = time.monotonic() - start
    assert elapsed < 5, f"取消应在数秒内完成，实际 {elapsed:.1f}s"
    time.sleep(3)                       # 若子进程还活着，此刻已经写了文件
    assert not flag.exists(), "取消后子进程仍在执行（进程树没杀干净）"
