"""
文本预处理器模块
负责文件读取、中文分词、停用词过滤等预处理工作。
"""

import os
import re
import jieba

# 匹配纯标点符号、空白字符和特殊符号的正则表达式
_PUNCTUATION_PATTERN = re.compile(r'^[\s\W_]+$')


class TextProcessor:
    """文本预处理器：读取文件 -> jieba分词 -> 去停用词"""

    def __init__(self, stop_words_path=None):
        """
        初始化文本预处理器。

        Args:
            stop_words_path: 停用词文件路径，默认为同目录下的 stop_words.txt
        """
        if stop_words_path is None:
            stop_words_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'stop_words.txt'
            )
        self.stop_words = self._load_stop_words(stop_words_path)

    def _load_stop_words(self, path):
        """
        加载停用词表。

        Args:
            path: 停用词文件路径

        Returns:
            停用词集合 (set)
        """
        stop_words = set()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        stop_words.add(word)
        return stop_words

    def read_file(self, file_path):
        """
        读取文件内容，自动尝试 UTF-8 和 GBK 编码。

        Args:
            file_path: 文件绝对路径

        Returns:
            文件文本内容 (str)

        Raises:
            FileNotFoundError: 文件不存在
            UnicodeDecodeError: 无法用已知编码解码文件
            IOError: 文件读取失败
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 按优先级尝试不同编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
        last_error = None
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError as e:
                last_error = e
                continue

        raise UnicodeDecodeError(
            last_error.encoding,
            last_error.object,
            last_error.start,
            last_error.end,
            f"无法用 UTF-8/GBK 解码文件: {file_path}"
        )

    def tokenize(self, text):
        """
        使用 jieba 进行中文分词。

        Args:
            text: 待分词的文本

        Returns:
            分词结果列表 (list[str])
        """
        return list(jieba.cut(text))

    def _is_punctuation(self, token):
        """
        判断一个 token 是否为纯标点符号或空白字符。

        Args:
            token: 待检查的 token

        Returns:
            True 如果 token 全部由标点符号/空白组成
        """
        return bool(_PUNCTUATION_PATTERN.match(token))

    def remove_stopwords(self, tokens):
        """
        过滤停用词、标点符号和空白字符。

        Args:
            tokens: 分词后的 token 列表

        Returns:
            过滤后的 token 列表 (list[str])
        """
        result = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if token in self.stop_words:
                continue
            if self._is_punctuation(token):
                continue
            result.append(token)
        return result

    def process(self, file_path):
        """
        完整预处理流程：读取文件 -> 分词 -> 去停用词。

        Args:
            file_path: 文件路径

        Returns:
            预处理后的 token 列表 (list[str])
        """
        text = self.read_file(file_path)
        tokens = self.tokenize(text)
        filtered_tokens = self.remove_stopwords(tokens)
        return filtered_tokens
