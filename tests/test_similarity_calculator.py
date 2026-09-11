"""
SimilarityCalculator 模块单元测试

测试内容：
- 完全相同的文本（边界值 1.00）
- 完全不同的文本（边界值 0.00）
- 部分相似的文本（核心场景）
- 空文件处理
- 纯标点符号处理
- 单字符文本
- 中英文混合文本
- 超长文本性能测试
"""

import os
import pytest
from similarity_calculator import SimilarityCalculator


def get_data_path(filename):
    """获取 data 目录下测试文件的路径"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    return os.path.join(data_dir, filename)


# ============================================================
# 测试7: 完全相同的文本，相似度应为 1.00
# ============================================================
def test_identical_text():
    """
    边界值测试：两篇完全相同的文本，重复率应为 1.00

    构造思路：使用相同的文件内容，经过相同的分词和过滤后，
    词频向量完全一致，余弦相似度应为 1.0。
    """
    calculator = SimilarityCalculator()
    orig_path = get_data_path('orig.txt')
    same_path = get_data_path('orig_same.txt')

    similarity = calculator.calculate_similarity(orig_path, same_path)

    assert similarity == 1.00


# ============================================================
# 测试8: 完全不同的文本，相似度应为 0.00
# ============================================================
def test_completely_different():
    """
    边界值测试：两篇完全不同的文本，重复率应为 0.00

    构造思路：原文是日常对话，对比文件是计算机科学相关内容，
    两者没有共同词汇，余弦相似度应为 0.0。
    """
    calculator = SimilarityCalculator()
    orig_path = get_data_path('orig.txt')
    diff_path = get_data_path('orig_diff.txt')

    similarity = calculator.calculate_similarity(orig_path, diff_path)

    assert similarity == 0.00


# ============================================================
# 测试9: 部分相似的文本（核心场景：增删改抄袭）
# ============================================================
def test_partial_similarity():
    """
    核心场景测试：经过增删改的抄袭版论文

    构造思路：原文 "今天是星期天，天气晴，今天晚上我要去看电影。"
    抄袭版 "今天是周天，天气晴朗，我晚上要去看电影。"
    两者有部分词汇相同（今天、晚上、看电影），部分不同（星期天 vs 周天），
    相似度应在 0 到 1 之间。
    """
    calculator = SimilarityCalculator()
    orig_path = get_data_path('orig.txt')
    plagia_path = get_data_path('orig_add.txt')

    similarity = calculator.calculate_similarity(orig_path, plagia_path)

    assert 0.00 < similarity < 1.00


# ============================================================
# 测试10: 空文件处理，相似度应为 0.00
# ============================================================
def test_empty_file():
    """
    异常处理测试：空文件

    构造思路：当一方为空文件时，分词结果为空列表，
    词频向量为零向量，余弦相似度应为 0.0。
    """
    calculator = SimilarityCalculator()
    orig_path = get_data_path('orig.txt')
    empty_path = get_data_path('empty.txt')

    similarity = calculator.calculate_similarity(orig_path, empty_path)

    assert similarity == 0.00


# ============================================================
# 测试11: 纯标点符号文件，相似度应为 0.00
# ============================================================
def test_only_punctuation():
    """
    边界情况测试：文件只含标点符号

    构造思路：标点符号在分词后会被去除或标记为停用词，
    剩余有效词为零，相似度应为 0.0。
    """
    calculator = SimilarityCalculator()
    orig_path = get_data_path('orig.txt')
    punct_path = get_data_path('punctuation.txt')

    similarity = calculator.calculate_similarity(orig_path, punct_path)

    assert similarity == 0.00


# ============================================================
# 测试12: 单字符文本处理
# ============================================================
def test_single_character(tmp_path):
    """
    极端输入测试：仅含单个中文字符的文件

    构造思路：两篇文本各只有一个字，相同则相似度为1，
    不同则为0。这里测试两个不同的单字。
    """
    file1 = tmp_path / "char1.txt"
    file1.write_text("山", encoding='utf-8')
    file2 = tmp_path / "char2.txt"
    file2.write_text("水", encoding='utf-8')

    calculator = SimilarityCalculator()
    similarity = calculator.calculate_similarity(str(file1), str(file2))

    assert similarity == 0.00


# ============================================================
# 测试13: 中英文混合文本
# ============================================================
def test_mixed_language(tmp_path):
    """
    混合语言测试：中英文混合的文本

    构造思路：两篇文本包含中文和英文混合内容，
    jieba 应能正确分词，相似度计算正常工作。
    """
    file1 = tmp_path / "mixed1.txt"
    file1.write_text("Python是一种流行的编程语言，适合数据分析。", encoding='utf-8')
    file2 = tmp_path / "mixed2.txt"
    file2.write_text("Python是一种流行的编程语言，适合机器学习。", encoding='utf-8')

    calculator = SimilarityCalculator()
    similarity = calculator.calculate_similarity(str(file1), str(file2))

    # 两段文本大部分相同，相似度应较高
    assert 0.00 < similarity <= 1.00
    assert similarity > 0.5


# ============================================================
# 测试14: 超长文本性能测试（5秒内完成）
# ============================================================
def test_long_text_performance(tmp_path):
    """
    性能测试：超长文本应在5秒内完成计算

    构造思路：生成一篇超长文本（重复多次），确保程序
    在5秒限制内完成处理。
    """
    import time

    base_text = "计算机科学与技术专业是研究计算机软硬件设计与实现的学科。"
    long_text = base_text * 500  # 重复500次，约2万字

    file1 = tmp_path / "long1.txt"
    file1.write_text(long_text, encoding='utf-8')
    file2 = tmp_path / "long2.txt"
    file2.write_text(base_text * 500, encoding='utf-8')

    calculator = SimilarityCalculator()
    start = time.time()
    similarity = calculator.calculate_similarity(str(file1), str(file2))
    elapsed = time.time() - start

    # 应在5秒内完成
    assert elapsed < 5.0
    # 两篇相同的长文本相似度应为 1.00
    assert similarity == 1.00


# ============================================================
# 测试15: 余弦相似度计算（直接测试核心函数）
# ============================================================
def test_cosine_similarity_direct():
    """
    白盒测试：直接测试 cosine_similarity 函数

    构造思路：使用已知词频向量验证余弦相似度计算的正确性。
    """
    from collections import Counter

    calculator = SimilarityCalculator()
    vec1 = Counter({'苹果': 3, '香蕉': 2, '橙子': 1})
    vec2 = Counter({'苹果': 3, '香蕉': 2, '橙子': 1})

    # 相同向量，相似度应为 1.0
    similarity = calculator.cosine_similarity(vec1, vec2)
    assert abs(similarity - 1.0) < 0.001

    vec3 = Counter({'苹果': 1})
    vec4 = Counter({'香蕉': 1})
    # 完全不同，相似度应为 0.0
    similarity = calculator.cosine_similarity(vec3, vec4)
    assert similarity == 0.0

    # 零向量
    vec5 = Counter()
    vec6 = Counter({'苹果': 1})
    similarity = calculator.cosine_similarity(vec5, vec6)
    assert similarity == 0.0
