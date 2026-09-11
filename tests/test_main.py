"""
main.py 主入口单元测试

测试内容：
- 命令行参数解析
- 正常流程输出
- 文件不存在异常处理
- 输出格式验证（两位小数）
- 参数数量错误处理
"""

import os
import sys
import subprocess
import pytest


def get_data_path(filename):
    """获取 data 目录下测试文件的路径"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    return os.path.join(data_dir, filename)


def get_main_path():
    """获取 main.py 的绝对路径"""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'main.py')


# ============================================================
# 测试16: 正常流程 - 输出文件包含正确的重复率
# ============================================================
def test_normal_flow(tmp_path):
    """
    集成测试：完整运行 main.py，验证输出文件内容

    构造思路：使用样例数据文件运行程序，检查输出文件
    是否包含浮点数格式的重复率。
    """
    orig_path = get_data_path('orig.txt')
    plagia_path = get_data_path('orig_add.txt')
    answer_path = str(tmp_path / "ans.txt")

    result = subprocess.run(
        [sys.executable, get_main_path(), orig_path, plagia_path, answer_path],
        capture_output=True, text=True
    )

    assert result.returncode == 0
    assert os.path.exists(answer_path)

    # 读取输出并验证格式
    with open(answer_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # 应为浮点数格式，如 0.85
    value = float(content)
    assert 0.00 <= value <= 1.00


# ============================================================
# 测试17: 完全相同的文件 - 输出应为 1.00
# ============================================================
def test_identical_files_output(tmp_path):
    """
    边界值测试：两篇相同文本，输出文件内容应为 1.00

    构造思路：使用相同内容的文件运行程序，验证输出为 1.00。
    """
    orig_path = get_data_path('orig.txt')
    same_path = get_data_path('orig_same.txt')
    answer_path = str(tmp_path / "ans.txt")

    result = subprocess.run(
        [sys.executable, get_main_path(), orig_path, same_path, answer_path],
        capture_output=True, text=True
    )

    assert result.returncode == 0

    with open(answer_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    assert content == "1.00"


# ============================================================
# 测试18: 输出格式验证 - 精确到小数点后两位
# ============================================================
def test_output_format(tmp_path):
    """
    格式验证测试：输出必须精确到小数点后两位

    构造思路：运行程序后检查输出文件内容是否符合 "0.xx" 格式。
    """
    orig_path = get_data_path('orig.txt')
    plagia_path = get_data_path('orig_add.txt')
    answer_path = str(tmp_path / "ans.txt")

    subprocess.run(
        [sys.executable, get_main_path(), orig_path, plagia_path, answer_path],
        capture_output=True, text=True
    )

    with open(answer_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # 验证格式：应为 "0.xx" 或 "1.00" 或 "0.00"
    parts = content.split('.')
    assert len(parts) == 2
    assert len(parts[1]) == 2
    float(content)  # 确保能转为浮点数


# ============================================================
# 测试19: 文件不存在 - 应非正常退出
# ============================================================
def test_file_not_found(tmp_path):
    """
    异常处理测试：输入文件不存在时，程序应以非零状态码退出

    构造思路：传入不存在的文件路径，验证程序不会崩溃，
    而是输出错误信息并以非零状态码退出。
    """
    answer_path = str(tmp_path / "ans.txt")

    result = subprocess.run(
        [sys.executable, get_main_path(), 'C:\\nonexistent\\orig.txt',
         'C:\\nonexistent\\plagia.txt', answer_path],
        capture_output=True, text=True
    )

    assert result.returncode != 0
    assert '错误' in result.stderr


# ============================================================
# 测试20: 参数数量错误 - 应打印用法说明
# ============================================================
def test_wrong_args():
    """
    异常处理测试：参数数量不足时，应打印用法说明并退出

    构造思路：不传入足够的参数，验证程序给出友好的错误提示。
    """
    result = subprocess.run(
        [sys.executable, get_main_path(), 'only_one_arg'],
        capture_output=True, text=True
    )

    assert result.returncode != 0
    assert '用法' in result.stdout or '用法' in result.stderr
