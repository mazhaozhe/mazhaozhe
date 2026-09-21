"""判题模块：解析题目文件，与答案文件比对判分统计，输出 Grade.txt。"""

import re

from .fraction_utils import parse_value

__all__ = ["tokenize", "evaluate_text", "grade"]

_TOKEN_RE = re.compile(r"\d+'\d+/\d+|\d+/\d+|\d+|[()+\-×÷]")


def tokenize(text):
    """把一行题目切分为记号（数值 / 运算符 / 括号），忽略空格与等号。"""
    return _TOKEN_RE.findall(text.replace("−", "-"))   # 兼容 Unicode 减号


class _Parser:
    """递归下降分析法器。

    文法（四则运算符均为左结合）::

        expr   := term (('+' | '-') term)*
        term   := factor (('×' | '÷') factor)*
        factor := 数值 | '(' expr ')'
    """

    def __init__(self, tokens):
        self._tokens = tokens
        self._pos = 0

    def parse(self):
        value = self._expr()
        if self._pos != len(self._tokens):
            raise ValueError("表达式末尾存在多余记号")
        return value

    def _peek(self):
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _next(self):
        token = self._peek()
        self._pos += 1
        return token

    def _expr(self):
        value = self._term()
        while self._peek() in ("+", "-"):
            if self._next() == "+":
                value = value + self._term()
            else:
                value = value - self._term()
        return value

    def _term(self):
        value = self._factor()
        while self._peek() in ("×", "÷"):
            if self._next() == "×":
                value = value * self._factor()
            else:
                divisor = self._factor()
                if divisor == 0:
                    raise ZeroDivisionError("除数为 0")
                value = value / divisor
        return value

    def _factor(self):
        token = self._next()
        if token is None:
            raise ValueError("表达式不完整")
        if token == "(":
            value = self._expr()
            if self._next() != ")":
                raise ValueError("缺少右括号")
            return value
        if token in ("+", "-", "×", "÷", ")"):
            raise ValueError(f"记号位置错误: {token}")
        return parse_value(token)


def evaluate_text(text):
    """解析并计算一行题目，返回 :class:`~fractions.Fraction`。"""
    return _Parser(tokenize(text)).parse()


def grade(exercise_path, answer_path, output_path="Grade.txt"):
    """判分统计。

    逐题解析计算标准答案，与答案文件比对；结果写入 ``Grade.txt`` 并
    返回 ``(正确题号列表, 错误题号列表)``（题号从 1 开始）。
    """
    with open(exercise_path, "r", encoding="utf-8") as f:
        exercises = [line for line in (s.strip() for s in f) if line]
    with open(answer_path, "r", encoding="utf-8") as f:
        answers = [line for line in (s.strip() for s in f) if line]
    if len(exercises) != len(answers):
        raise ValueError("题目数量与答案数量不一致")

    correct, wrong = [], []
    for index, (exercise, answer) in enumerate(zip(exercises, answers), 1):
        expected = evaluate_text(exercise)
        actual = parse_value(answer)
        (correct if expected == actual else wrong).append(index)

    lines = []
    for label, numbers in (("Correct", correct), ("Wrong", wrong)):
        if numbers:
            joined = ", ".join(str(n) for n in numbers)
            lines.append(f"{label}: {len(numbers)} ({joined})")
        else:
            lines.append(f"{label}: 0")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return correct, wrong
