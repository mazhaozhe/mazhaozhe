"""fraction_utils 的单元测试：分数格式化与解析。"""

import unittest
from fractions import Fraction

from myapp.fraction_utils import format_value, parse_value


class FormatValueTest(unittest.TestCase):

    def test_natural(self):
        """自然数（含 0）直接输出。"""
        self.assertEqual(format_value(Fraction(7)), "7")
        self.assertEqual(format_value(Fraction(0)), "0")

    def test_proper_fraction(self):
        """真分数输出为 分子/分母。"""
        self.assertEqual(format_value(Fraction(3, 5)), "3/5")

    def test_mixed_fraction(self):
        """大于 1 的分数输出为带分数。"""
        self.assertEqual(format_value(Fraction(19, 8)), "2'3/8")
        self.assertEqual(format_value(Fraction(3, 2)), "1'1/2")


class ParseValueTest(unittest.TestCase):

    def test_parse_natural(self):
        self.assertEqual(parse_value("7"), Fraction(7))
        self.assertEqual(parse_value(" 0 "), Fraction(0))

    def test_parse_proper_fraction(self):
        self.assertEqual(parse_value("3/5"), Fraction(3, 5))

    def test_parse_mixed_fraction(self):
        self.assertEqual(parse_value("2'3/8"), Fraction(19, 8))

    def test_roundtrip(self):
        """解析后再格式化应得到原文。"""
        for text in ("7", "0", "3/5", "2'3/8", "1'1/2"):
            self.assertEqual(format_value(parse_value(text)), text)


if __name__ == "__main__":
    unittest.main()
