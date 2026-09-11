"""
pytest 配置文件
将项目根目录加入 sys.path，使测试文件能导入项目模块。
"""

import sys
import os

# 将项目根目录（conftest.py 所在目录）加入 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
