"""题目生成器：批量生成不重复的四则运算题目并写出文件。"""

import random

from .expression import (ExpressionError, canonical, evaluate,
                          random_tree, to_text)
from .fraction_utils import format_value

__all__ = ["generate_questions", "write_files"]


def generate_questions(count, value_limit, rng=None):
    """生成 count 道不重复的题目。

    :param count: 题目数量
    :param value_limit: 数值范围（自然数与分子分母均小于该值）
    :param rng: 可选的随机源，便于测试时固定种子复现
    :return: ``(题目文本列表, 答案文本列表)``
    :raises RuntimeError: 数值范围内不重复的合法题目不足 count 道

    生成策略（拒绝采样）：随机生成表达式树，先以规范化键判重，再带约束
    求值；任何一步不满足（重复 / 负数 / 除零 / 整除）即丢弃重试。尝试
    次数设置上限，范围过小时报错而非死循环。
    """
    rng = rng or random.Random()
    questions, answers, seen = [], [], set()
    attempts = 0
    max_attempts = max(200000, count * 200)
    while len(questions) < count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"在数值范围 {value_limit} 以内无法生成 {count} 道不重复的"
                "题目，请增大 -r 参数后重试")
        tree = random_tree(rng, value_limit, rng.randint(1, 3))
        key = canonical(tree)
        if key in seen:                               # 与已有题目重复
            continue
        try:
            answer = evaluate(tree)
        except ExpressionError:                       # 违反生成约束
            continue
        seen.add(key)
        questions.append(to_text(tree))
        answers.append(format_value(answer))
    return questions, answers


def write_files(questions, answers,
                exercise_path="Exercises.txt", answer_path="Answers.txt"):
    """按作业要求的格式写出 Exercises.txt 与 Answers.txt。"""
    with open(exercise_path, "w", encoding="utf-8") as f:
        f.writelines(f"{question} = \n" for question in questions)
    with open(answer_path, "w", encoding="utf-8") as f:
        f.writelines(f"{answer}\n" for answer in answers)
