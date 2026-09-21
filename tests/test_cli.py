"""CLI 功能测试：参数校验、生成模式与判题模式。"""

import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(REPO_ROOT, "Myapp.py")


def run_app(args, cwd):
    return subprocess.run([sys.executable, APP] + args, cwd=cwd,
                          capture_output=True, encoding="utf-8",
                          errors="replace")


class GenerationModeTest(unittest.TestCase):

    def test_generate_with_n_and_r(self):
        """Myapp.py -n 10 -r 10 生成两个输出文件，各 10 行。"""
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_app(["-n", "10", "-r", "10"], tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(os.path.join(tmp, "Exercises.txt"),
                      encoding="utf-8") as f:
                exercises = [s for s in f.read().splitlines() if s.strip()]
            self.assertEqual(len(exercises), 10)
            with open(os.path.join(tmp, "Answers.txt"),
                      encoding="utf-8") as f:
                answers = [s for s in f.read().splitlines() if s.strip()]
            self.assertEqual(len(answers), 10)

    def test_only_r_uses_default_n(self):
        """只给 -r 时按缺省数量 10 生成。"""
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_app(["-r", "20"], tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(os.path.join(tmp, "Exercises.txt"),
                      encoding="utf-8") as f:
                self.assertEqual(len(f.read().splitlines()), 10)

    def test_missing_r_errors_with_help(self):
        """-r 必须给定：缺失时报错并输出帮助信息。"""
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_app(["-n", "10"], tmp)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("usage", proc.stderr.lower())
            self.assertIn("-r", proc.stderr)

    def test_invalid_n(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_app(["-n", "0", "-r", "10"], tmp)
            self.assertEqual(proc.returncode, 2)

    def test_no_args_shows_help(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_app([], tmp)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("usage", proc.stderr.lower())


class GradeModeTest(unittest.TestCase):

    def _prepare(self, tmp):
        with open(os.path.join(tmp, "Exercises.txt"), "w",
                  encoding="utf-8") as f:
            f.write("1 + 2 = \n3 × 4 = \n5 - 5 = \n1 ÷ 2 = \n")
        with open(os.path.join(tmp, "Answers.txt"), "w",
                  encoding="utf-8") as f:
            f.write("3\n11\n1\n1/2\n")

    def test_grade_outputs_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._prepare(tmp)
            proc = run_app(["-e", "Exercises.txt", "-a", "Answers.txt"], tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(os.path.join(tmp, "Grade.txt"), encoding="utf-8") as f:
                self.assertEqual(f.read(),
                                 "Correct: 2 (1, 4)\nWrong: 2 (2, 3)\n")

    def test_grade_requires_both_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._prepare(tmp)
            proc = run_app(["-e", "Exercises.txt"], tmp)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("-a", proc.stderr)


if __name__ == "__main__":
    unittest.main()
