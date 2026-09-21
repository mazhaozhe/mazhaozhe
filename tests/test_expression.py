"""expression 的单元测试：约束校验、判重等价性、序列化格式。"""

import random
import unittest
from fractions import Fraction

from myapp.expression import (ExpressionError, Op, Value, canonical,
                              evaluate, random_tree, to_text)


def num(value):
    """构造自然数叶子。"""
    return Value(Fraction(value))


class ConstraintTest(unittest.TestCase):
    """随机生成大量表达式，验证全局约束性质。"""

    def test_no_negative_result(self):
        """凡通过约束求值的表达式，结果必非负。"""
        rng = random.Random(42)
        passed = 0
        for _ in range(3000):
            tree = random_tree(rng, 10, rng.randint(1, 3))
            try:
                result = evaluate(tree)
            except ExpressionError:
                continue
            passed += 1
            self.assertGreaterEqual(result, 0)
        self.assertGreater(passed, 500)               # 确实有大量合法样本

    def test_division_result_is_fraction(self):
        """任何 ÷ 子表达式的结果都不是整数（真分数/带分数）。"""
        rng = random.Random(2024)
        for _ in range(3000):
            tree = random_tree(rng, 10, rng.randint(1, 3))
            try:
                evaluate(tree)
            except ExpressionError:
                continue
            self._check_division(tree)

    def _check_division(self, node):
        if isinstance(node, Op):
            if node.op == "÷":
                left = evaluate(node.left)
                right = evaluate(node.right)
                self.assertNotEqual(right, 0)
                self.assertNotEqual((left / right).denominator, 1)
            self._check_division(node.left)
            self._check_division(node.right)


class CanonicalTest(unittest.TestCase):
    """判重等价性，采用作业说明中给出的判定示例。"""

    def test_associative_swap_equivalent(self):
        """3 + (2 + 1) 与 1 + 2 + 3 重复（+ 左结合，交换律）。"""
        a = Op("+", num(3), Op("+", num(2), num(1)))
        b = Op("+", Op("+", num(1), num(2)), num(3))
        self.assertEqual(canonical(a), canonical(b))

    def test_different_structure_not_equivalent(self):
        """1 + 2 + 3 与 3 + 2 + 1 不重复（结合结构不同）。"""
        b = Op("+", Op("+", num(1), num(2)), num(3))
        c = Op("+", Op("+", num(3), num(2)), num(1))
        self.assertNotEqual(canonical(b), canonical(c))

    def test_multiplication_swap_equivalent(self):
        """6 × 8 与 8 × 6 重复。"""
        a = Op("×", num(6), num(8))
        b = Op("×", num(8), num(6))
        self.assertEqual(canonical(a), canonical(b))

    def test_addition_swap_equivalent(self):
        """23 + 45 与 45 + 23 重复。"""
        a = Op("+", num(23), num(45))
        b = Op("+", num(45), num(23))
        self.assertEqual(canonical(a), canonical(b))

    def test_subtraction_not_swappable(self):
        """减法不满足交换律：3 - 2 与 2 - 3 的规范化键不同。"""
        a = Op("-", num(3), num(2))
        b = Op("-", num(2), num(3))
        self.assertNotEqual(canonical(a), canonical(b))


class ToTextTest(unittest.TestCase):

    def test_left_associative_no_parens(self):
        tree = Op("+", Op("+", num(1), num(2)), num(3))
        self.assertEqual(to_text(tree), "1 + 2 + 3")

    def test_right_child_same_precedence_keeps_parens(self):
        tree = Op("+", num(3), Op("+", num(2), num(1)))
        self.assertEqual(to_text(tree), "3 + (2 + 1)")

    def test_lower_precedence_child_keeps_parens(self):
        tree = Op("×", Op("+", num(1), num(2)), num(3))
        self.assertEqual(to_text(tree), "(1 + 2) × 3")

    def test_fraction_text(self):
        tree = Op("+", Value(Fraction(1, 6)), Value(Fraction(1, 8)))
        self.assertEqual(to_text(tree), "1/6 + 1/8")


class EvaluateTest(unittest.TestCase):

    def test_spec_example(self):
        """作业示例：1/6 + 1/8 = 7/24。"""
        tree = Op("+", Value(Fraction(1, 6)), Value(Fraction(1, 8)))
        self.assertEqual(evaluate(tree), Fraction(7, 24))

    def test_negative_rejected(self):
        tree = Op("-", num(1), num(2))
        with self.assertRaises(ExpressionError):
            evaluate(tree)

    def test_division_by_zero_rejected(self):
        tree = Op("÷", num(1), num(0))
        with self.assertRaises(ExpressionError):
            evaluate(tree)

    def test_exact_division_rejected(self):
        """4 ÷ 2 结果为自然数，违反"除法结果必须是分数"。"""
        tree = Op("÷", num(4), num(2))
        with self.assertRaises(ExpressionError):
            evaluate(tree)

    def test_mixed_division_accepted(self):
        """3 ÷ 2 = 1'1/2，结果是带分数（题目定义的"真分数"），合法。"""
        tree = Op("÷", num(3), num(2))
        self.assertEqual(evaluate(tree), Fraction(3, 2))


if __name__ == "__main__":
    unittest.main()
