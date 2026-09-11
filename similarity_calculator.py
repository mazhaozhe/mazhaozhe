"""
相似度计算模块
基于余弦相似度算法计算两篇文本的重复率。
"""

import math
from collections import Counter
from text_processor import TextProcessor


class SimilarityCalculator:
    """
    论文查重相似度计算器。

    使用余弦相似度算法：
    1. 对两篇文本分别进行分词和预处理
    2. 统计词频，构建词频向量
    3. 计算两个向量的余弦夹角，即为重复率
    """

    def __init__(self, stop_words_path=None):
        """
        初始化相似度计算器。

        Args:
            stop_words_path: 停用词文件路径
        """
        self.processor = TextProcessor(stop_words_path)

    def build_vectors(self, tokens1, tokens2):
        """
        将两组 token 转换为词频向量。

        Args:
            tokens1: 第一篇文本的分词结果
            tokens2: 第二篇文本的分词结果

        Returns:
            (Counter, Counter) 两个词频统计字典
        """
        vec1 = Counter(tokens1)
        vec2 = Counter(tokens2)
        return vec1, vec2

    def cosine_similarity(self, vec1, vec2):
        """
        计算两个词频向量的余弦相似度。

        公式: cos(theta) = (A·B) / (||A|| × ||B||)

        Args:
            vec1: 词频向量1 (Counter/dict)
            vec2: 词频向量2 (Counter/dict)

        Returns:
            余弦相似度，范围 [0.0, 1.0]
        """
        # 获取所有词的并集
        all_words = set(vec1.keys()) | set(vec2.keys())

        # 构建数值向量
        v1 = [vec1.get(word, 0) for word in all_words]
        v2 = [vec2.get(word, 0) for word in all_words]

        # 计算点积
        dot_product = sum(a * b for a, b in zip(v1, v2))

        # 计算向量模长
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))

        # 避免除零：如果任一向量为零向量，相似度为0
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def calculate_similarity(self, orig_path, plagia_path):
        """
        计算两个文件的相似度（重复率）。

        完整流程：读取文件 -> 预处理 -> 构建向量 -> 余弦相似度 -> 保留两位小数

        Args:
            orig_path: 原文文件路径
            plagia_path: 抄袭版文件路径

        Returns:
            重复率，浮点数，范围 [0.0, 1.0]，保留两位小数
        """
        # 预处理两篇文本
        tokens1 = self.processor.process(orig_path)
        tokens2 = self.processor.process(plagia_path)

        # 构建词频向量
        vec1, vec2 = self.build_vectors(tokens1, tokens2)

        # 计算余弦相似度
        similarity = self.cosine_similarity(vec1, vec2)

        # 保留两位小数
        return round(similarity, 2)
