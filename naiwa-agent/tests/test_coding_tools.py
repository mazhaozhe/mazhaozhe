"""
tests/test_coding_tools.py —— 代码开发工具测试
===============================================
用 pytest 的 tmp_path 临时目录测试文件读写/编辑/命令执行，
不污染真实文件系统。
"""

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
    out = run_shell.invoke({"command": "echo hello_agent", "timeout": 10})
    assert "hello_agent" in out and "exit_code=0" in out


def test_run_shell_dangerous_blocked():
    """危险命令必须被安全策略拦截（安全阀）。"""
    with pytest.raises(PermissionError, match="安全策略"):
        run_shell.invoke({"command": "shutdown /s", "timeout": 5})


def test_run_shell_timeout():
    """超时命令 → TimeoutError。"""
    with pytest.raises(TimeoutError):
        run_shell.invoke({"command": "ping -n 10 127.0.0.1", "timeout": 2})
