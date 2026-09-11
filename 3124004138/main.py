"""
论文查重程序 - 主入口

用法:
    python main.py [原文文件路径] [抄袭版文件路径] [答案文件路径]

示例:
    python main.py C:\\tests\\orig.txt C:\\tests\\orig_add.txt C:\\tests\\ans.txt

输出:
    在答案文件中写入重复率（浮点数，精确到小数点后两位，如 0.85）
"""

import sys
import os
from similarity_calculator import SimilarityCalculator


def write_result(answer_path, similarity):
    """
    将相似度结果写入答案文件。

    Args:
        answer_path: 答案文件路径
        similarity: 相似度浮点值
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(answer_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 写入结果，保留两位小数
    with open(answer_path, 'w', encoding='utf-8') as f:
        f.write(f"{similarity:.2f}")


def main():
    """
    主函数：解析命令行参数，计算相似度，输出结果。
    """
    # 检查参数数量
    if len(sys.argv) != 4:
        print("用法: python main.py [原文文件路径] [抄袭版文件路径] [答案文件路径]")
        print("示例: python main.py C:\\tests\\orig.txt "
              "C:\\tests\\orig_add.txt C:\\tests\\ans.txt")

        sys.exit(1)

    orig_path = sys.argv[1]
    plagia_path = sys.argv[2]
    answer_path = sys.argv[3]

    try:
        # 创建相似度计算器并计算
        calculator = SimilarityCalculator()
        similarity = calculator.calculate_similarity(orig_path, plagia_path)

        # 写入答案文件
        write_result(answer_path, similarity)

        print(f"查重完成，重复率: {similarity:.2f}")

    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except UnicodeDecodeError as e:
        print(f"错误: 文件编码无法识别 - {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        print(f"错误: 文件权限不足 - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
