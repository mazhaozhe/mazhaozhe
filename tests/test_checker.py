"""checker 的单元测试：表达式解析与判分统计。"""

import os
import tempfile
import unittest
from fractions import Fraction

from myapp.checker import evaluate_text, grade


class EvaluateTextTest(unittest.TestCase):

    def test_spec_example(self):
        """作业示例：1/6 + 1/8 = 7/24。"""
        self.assertEqual(evaluate_text("1/6 + 1/8"), Fraction(7, 24))

    def test_precedence(self):
        self.assertEqual(evaluate_text("1 + 2 × 3"), Fraction(7))
        self.assertEqual(evaluate_text("(1 + 2) × 3"), Fraction(9))

    def test_left_associative(self):
        self.assertEqual(evaluate_text("10 - 2 - 3"), Fraction(5))
        self.assertEqual(evaluate_text("1 ÷ 2 × 4"), Fraction(2))

    def test_mixed_and_fraction_operands(self):
        self.assertEqual(evaluate_text("2'3/8 + 1/8"), Fraction(5, 2))
        self.assertEqual(evaluate_text("3 × 1/2"), Fraction(3, 2))

    def test_equals_sign_ignored(self):
        """行尾的等号与空格应被忽略。"""
        self.assertEqual(evaluate_text("1 + 2 = "), Fraction(3))


class GradeTest(unittest.TestCase):

    def _write(self, tmp, name, lines):
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def test_grade_statistics(self):
        """对错混合时的统计格式。"""
        exercises = ["1 + 2 = ", "3 × 4 = ", "5 - 5 = ", "1 ÷ 2 = "]
        answers = ["3", "11", "1", "1/2"]              # 第 2、3 题错误
        with tempfile.TemporaryDirectory() as tmp:
            old = os.getcwd()
            try:
                os.chdir(tmp)
                exercise = self._write(tmp, "Exercises.txt", exercises)
                answer = self._write(tmp, "Answers.txt", answers)
                correct, wrong = grade(exercise, answer)
                self.assertEqual(correct, [1, 4])
                self.assertEqual(wrong, [2, 3])
                with open("Grade.txt", encoding="utf-8") as f:
                    self.assertEqual(f.read(),
                                     "Correct: 2 (1, 4)\nWrong: 2 (2, 3)\n")
            finally:
                os.chdir(old)

    def test_grade_all_wrong(self):
        exercises = ["1 + 1 = ", "2 × 2 = "]
        answers = ["3", "5"]
        with tempfile.TemporaryDirectory() as tmp:
            old = os.getcwd()
            try:
                os.chdir(tmp)
                exercise = self._write(tmp, "Exercises.txt", exercises)
                answer = self._write(tmp, "Answers.txt", answers)
                correct, wrong = grade(exercise, answer)
                self.assertEqual(correct, [])
                self.assertEqual(wrong, [1, 2])
                with open("Grade.txt", encoding="utf-8") as f:
                    self.assertIn("Correct: 0", f.read())
            finally:
                os.chdir(old)


if __name__ == "__main__":
    unittest.main()
