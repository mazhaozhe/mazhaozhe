"""算术表达式模块：随机生成、带约束求值、序列化与规范化（判重）。

表达式采用二叉树表示：

- :class:`Value` 叶子节点保存一个非负数值（自然数 / 真分数 / 带分数）；
- :class:`Op` 内部节点保存一个二元运算符 ``+ - × ÷`` 及左右子树。

判重策略：两道题目等价，当且仅当其表达式树能通过有限次交换 ``+`` / ``×``
的左右子树相互变换得到（结合律体现为树的结合方式）。因此把表达式树
序列化为**规范化键**：``+`` / ``×`` 节点的两个子键按字典序排序后拼接，
``-`` / ``÷`` 节点保持左右顺序。规范化键相同的题目视为重复。
"""

import random
from fractions import Fraction

from .fraction_utils import format_value

__all__ = ["Value", "Op", "ExpressionError",
           "random_tree", "evaluate", "to_text", "canonical"]

_OPERATORS = ("+", "-", "×", "÷")
_PRECEDENCE = {"+": 1, "-": 1, "×": 2, "÷": 2}


class ExpressionError(Exception):
    """表达式违反题目生成约束（负数 / 除零 / 除法结果不是分数）。"""


class Value:
    """叶子节点：一个非负数值。"""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value if isinstance(value, Fraction) else Fraction(value)


class Op:
    """内部节点：二元运算符及其左右子树。"""

    __slots__ = ("op", "left", "right")

    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right


def random_operand(rng, limit):
    """在 (0, limit) 范围内随机生成一个操作数。

    三类操作数：自然数、真分数、带分数（带分数按题目定义同样属于"真分数"）。
    真分数 / 带分数的分子、分母均小于 limit。
    """
    roll = rng.random()
    if limit >= 3 and roll < 0.35:                   # 真分数 n/d，1 <= n < d < limit
        den = rng.randrange(2, limit)
        num = rng.randrange(1, den)
        return Fraction(num, den)
    if limit >= 3 and roll < 0.50:                   # 带分数 w'n/d
        whole = rng.randrange(1, limit)
        den = rng.randrange(2, limit)
        num = rng.randrange(1, den)
        return Fraction(whole) + Fraction(num, den)
    return Fraction(rng.randrange(0, limit))         # 自然数 0 ~ limit-1


def random_tree(rng, limit, operator_count):
    """生成恰好含 operator_count 个运算符的随机表达式树。"""
    if operator_count == 0:
        return Value(random_operand(rng, limit))
    if operator_count == 1:
        return Op(rng.choice(_OPERATORS),
                  Value(random_operand(rng, limit)),
                  Value(random_operand(rng, limit)))
    left_ops = rng.randint(0, operator_count - 1)     # 剩余运算符分给左右子树
    right_ops = operator_count - 1 - left_ops
    return Op(rng.choice(_OPERATORS),
              random_tree(rng, limit, left_ops),
              random_tree(rng, limit, right_ops))


def evaluate(node):
    """带约束求值，返回 :class:`Fraction`。

    违反约束时抛出 :class:`ExpressionError`：

    - 任何 ``e1 - e2`` 子表达式要求 e1 >= e2（计算过程不能出现负数）；
    - 任何 ``e1 ÷ e2`` 子表达式要求 e2 != 0 且结果为分数（不能整除）。
    """
    if isinstance(node, Value):
        return node.value
    left = evaluate(node.left)
    right = evaluate(node.right)
    op = node.op
    if op == "+":
        return left + right
    if op == "-":
        if left < right:
            raise ExpressionError("减法出现负数")
        return left - right
    if op == "×":
        return left * right
    if op == "÷":
        if right == 0:
            raise ExpressionError("除数为 0")
        quotient = left / right
        if quotient.denominator == 1:
            raise ExpressionError("除法结果不是分数")
        return quotient
    raise ValueError(f"未知运算符: {op}")


def to_text(node):
    """把表达式树序列化为题目文本。

    运算符前后带空格；按四个运算符均为左结合的规则输出**最少括号**：
    例如 ``(1 + 2) + 3`` 输出为 ``1 + 2 + 3``，而 ``3 + (2 + 1)``
    的右子树与父级同级且处于右侧，需要保留括号。
    """
    return _to_text(node, parent=None, is_right=False)


def _to_text(node, parent, is_right):
    if isinstance(node, Value):
        return format_value(node.value)
    text = (f"{_to_text(node.left, node.op, False)} {node.op} "
            f"{_to_text(node.right, node.op, True)}")
    if parent is not None and (
            _PRECEDENCE[node.op] < _PRECEDENCE[parent]
            or (_PRECEDENCE[node.op] == _PRECEDENCE[parent] and is_right)
    ):
        return f"({text})"                           # 需要括号保证结合顺序
    return text


def canonical(node):
    """计算表达式树的规范化键，用于题目判重。"""
    if isinstance(node, Value):
        return format_value(node.value)
    left = canonical(node.left)
    right = canonical(node.right)
    if node.op in ("+", "×") and right < left:        # 交换律：子键排序
        left, right = right, left
    return f"({left}{node.op}{right})"
