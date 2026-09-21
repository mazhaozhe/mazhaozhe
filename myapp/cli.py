"""命令行接口：-n/-r 生成题目，-e/-a 判分统计。"""

import argparse
import sys

from .checker import grade
from .generator import generate_questions, write_files

__all__ = ["main"]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="Myapp",
        description="小学四则运算题目生成器",
        epilog="示例：Myapp.py -n 10 -r 10  生成 10 道 10 以内的题目；"
               "Myapp.py -e Exercises.txt -a Answers.txt  判分统计")
    parser.add_argument("-n", type=int, default=10, metavar="N",
                        help="生成题目的个数（默认 10）")
    parser.add_argument("-r", type=int, metavar="R",
                        help="题目中数值（自然数、真分数分子分母）的范围上限，"
                             "必须给定，例如 -r 10 表示 10 以内（不含 10）")
    parser.add_argument("-e", metavar="FILE", help="判题模式：题目文件路径")
    parser.add_argument("-a", metavar="FILE", help="判题模式：答案文件路径")
    return parser


def main(argv=None):
    # 输出重定向到管道/文件时统一使用 UTF-8，避免乱码
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.e or args.a:                               # 判题模式
        if not (args.e and args.a):
            parser.error("判题模式需要同时给定 -e <题目文件> 与 -a <答案文件>")
        correct, wrong = grade(args.e, args.a)
        print(f"共判定 {len(correct) + len(wrong)} 道题目："
              f"正确 {len(correct)} 道，错误 {len(wrong)} 道，"
              f"统计结果已写入 Grade.txt")
        return 0

    if args.r is None:                                 # -r 必须给定
        parser.error("缺少必需参数 -r <数值范围>，该参数必须给定"
                     "（例如 Myapp.py -n 10 -r 10）")
    if args.n < 1:
        parser.error("参数 -n 必须为正整数")
    if args.r < 1:
        parser.error("参数 -r 必须为自然数（大于等于 1）")

    questions, answers = generate_questions(args.n, args.r)
    write_files(questions, answers)
    print(f"已生成 {args.n} 道数值范围 {args.r} 以内的四则运算题目："
          f"题目写入 Exercises.txt，答案写入 Answers.txt")
    return 0
