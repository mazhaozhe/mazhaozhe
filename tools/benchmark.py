"""基准测试：统计不同规模下题目生成的耗时，供效能分析使用。

用法：python tools/benchmark.py
"""

import random
import sys
import time

from myapp.generator import generate_questions


def main():
    for count, limit in [(1000, 10), (10000, 10), (10000, 50)]:
        rng = random.Random(42)                       # 固定种子，保证可复现
        start = time.perf_counter()
        questions, _ = generate_questions(count, limit, rng)
        elapsed = time.perf_counter() - start
        print(f"n={count:<6} r={limit:<4} 用时 {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
