#!/usr/bin/env python3
"""Myapp —— 小学四则运算题目生成器（命令行入口）。

用法：
    python Myapp.py -n 10 -r 10
    python Myapp.py -e Exercises.txt -a Answers.txt
"""

from myapp.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
