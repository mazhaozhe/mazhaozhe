"""真分数 / 带分数的文本格式化与解析。

约定（与作业要求一致）：
- 自然数直接输出，例如 ``7``；
- 真分数输出为 ``分子/分母``，例如 ``3/5``；
- 大于 1 的分数输出为带分数 ``整数'分子/分母``，例如 ``2'3/8``。
"""

from fractions import Fraction

__all__ = ["format_value", "parse_value"]


def format_value(value):
    """把非负数值格式化为题目要求的文本形式。

    :param value: ``int`` 或非负 :class:`~fractions.Fraction`
    """
    if not isinstance(value, Fraction):
        value = Fraction(value)
    if value.denominator == 1:                       # 自然数（含 0）
        return str(value.numerator)
    if value < 1:                                    # 真分数
        return f"{value.numerator}/{value.denominator}"
    whole, num = divmod(value.numerator, value.denominator)   # 带分数
    return f"{whole}'{num}/{value.denominator}"


def parse_value(text):
    """把 ``7``、``3/5``、``2'3/8`` 形式的文本解析为 :class:`Fraction`。"""
    text = text.strip()
    if "'" in text:                                  # 带分数 2'3/8
        whole, frac = text.split("'", 1)
        num, den = frac.split("/", 1)
        return Fraction(int(whole) * int(den) + int(num), int(den))
    if "/" in text:                                  # 真分数 3/5
        num, den = text.split("/", 1)
        return Fraction(int(num), int(den))
    return Fraction(int(text))                       # 自然数 7
