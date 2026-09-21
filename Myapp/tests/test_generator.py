"""generator 的单元测试：数量、判重、约束、退化范围。"""

import random
import unittest

from myapp.checker import evaluate_text, tokenize
from myapp.fraction_utils import format_value, parse_value
from myapp.generator import generate_questions, write_files


class GenerateTest(unittest.TestCase):

    def test_count_and_no_duplicate(self):
        """生成的题目数量正确且互不重复。"""
        questions, answers = generate_questions(1000, 20, random.Random(7))
        self.assertEqual(len(questions), 1000)
        self.assertEqual(len(answers), 1000)
        self.assertEqual(len(set(questions)), 1000)

    def test_answers_match_reparse(self):
        """重新解析每道题计算答案，应与写出的答案一致。"""
        questions, answers = generate_questions(500, 10, random.Random(3))
        for question, answer in zip(questions, answers):
            self.assertEqual(format_value(evaluate_text(question)), answer)

    def test_operator_count_limit(self):
        """每道题运算符个数不超过 3。"""
        questions, _ = generate_questions(500, 10, random.Random(11))
        for question in questions:
            operators = sum(question.count(op) for op in "+-×÷")
            self.assertLessEqual(operators, 3, question)

    def test_operand_range(self):
        """所有操作数的数值（自然数、分子、分母、带分数整数部分）均小于 r。"""
        questions, _ = generate_questions(500, 10, random.Random(5))
        for question in questions:
            for token in tokenize(question):
                if not token[0].isdigit():
                    continue
                if "'" in token:                       # 带分数 2'3/8
                    whole, frac = token.split("'", 1)
                    self.assertLess(int(whole), 10)
                    num, den = frac.split("/", 1)
                    self.assertLess(int(num), 10)
                    self.assertLess(int(den), 10)
                elif "/" in token:                     # 真分数 3/5
                    num, den = token.split("/", 1)
                    self.assertLess(int(num), 10)
                    self.assertLess(int(den), 10)
                else:                                  # 自然数
                    self.assertLess(int(token), 10)

    def test_tiny_range_r_1(self):
        """r = 1 时范围内仍有少量可生成的不重复题目。"""
        questions, _ = generate_questions(10, 1, random.Random(1))
        self.assertEqual(len(questions), 10)

    def test_tiny_range_exhausted(self):
        """范围过小、题目数超出可生成上限时应当报错而非死循环。"""
        with self.assertRaises(RuntimeError):
            generate_questions(500, 1, random.Random(1))

    def test_ten_thousand_questions(self):
        """压测：一次生成一万道不重复题目。"""
        questions, answers = generate_questions(10000, 10, random.Random(2024))
        self.assertEqual(len(questions), 10000)
        self.assertEqual(len(set(questions)), 10000)
        self.assertEqual(len(answers), 10000)


class WriteFilesTest(unittest.TestCase):

    def test_write_files(self):
        import os
        import tempfile
        questions, answers = generate_questions(5, 10, random.Random(9))
        with tempfile.TemporaryDirectory() as tmp:
            exercise_path = os.path.join(tmp, "Exercises.txt")
            answer_path = os.path.join(tmp, "Answers.txt")
            write_files(questions, answers, exercise_path, answer_path)
            with open(exercise_path, encoding="utf-8") as f:
                lines = f.read().splitlines()
            self.assertEqual(len(lines), 5)
            for line, question in zip(lines, questions):
                self.assertTrue(line.startswith(question))
                self.assertTrue(line.endswith("= "))
            with open(answer_path, encoding="utf-8") as f:
                self.assertEqual(f.read().splitlines(), answers)


if __name__ == "__main__":
    unittest.main()
