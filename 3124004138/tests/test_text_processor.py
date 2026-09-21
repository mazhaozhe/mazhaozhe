"""
TextProcessor 模块单元测试

测试内容：
- 文件读取（正常、文件不存在、编码兼容）
- 分词功能
- 停用词过滤
- 完整预处理流程
"""

import os
import pytest
from text_processor import TextProcessor


# ============================================================
# 测试1: 正常读取UTF-8编码文件
# ============================================================
def test_read_file_utf8():
    """测试读取UTF-8编码的文件，应正确返回文件内容"""
    data_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
    file_path = os.path.join(data_dir, 'orig.txt')

    processor = TextProcessor()
    content = processor.read_file(file_path)

    assert '今天' in content
    assert '电影' in content


# ============================================================
# 测试2: 文件不存在时抛出FileNotFoundError
# ============================================================
def test_read_file_not_found():
    """测试读取不存在的文件，应抛出FileNotFoundError"""
    processor = TextProcessor()
    with pytest.raises(FileNotFoundError):
        processor.read_file('C:\\nonexistent\\file.txt')


# ============================================================
# 测试3: jieba分词功能
# ============================================================
def test_tokenize():
    """测试jieba中文分词，应正确切分中文词汇"""
    processor = TextProcessor()
    text = "今天是星期天，天气晴"
    tokens = processor.tokenize(text)

    assert isinstance(tokens, list)
    assert len(tokens) > 0
    # 分词结果中应包含 "今天" 这个词
    assert '今天' in tokens


# ============================================================
# 测试4: 停用词过滤功能
# ============================================================
def test_remove_stopwords():
    """测试停用词过滤，应去除常见停用词和空白字符"""
    processor = TextProcessor()
    # "的" 和 "了" 是停用词，"计算机" 和 "科学" 不是
    tokens = ['计算机', '的', '科学', ' ', '研究', '了']
    filtered = processor.remove_stopwords(tokens)

    assert '的' not in filtered
    assert '了' not in filtered
    assert '' not in filtered
    assert '计算机' in filtered
    assert '科学' in filtered
    assert '研究' in filtered


# ============================================================
# 测试5: 完整预处理流程
# ============================================================
def test_process():
    """测试完整的预处理流程：读取 -> 分词 -> 去停用词"""
    data_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
    file_path = os.path.join(data_dir, 'orig.txt')

    processor = TextProcessor()
    tokens = processor.process(file_path)

    assert isinstance(tokens, list)
    assert len(tokens) > 0
    # 停用词 "的" 不应出现在结果中
    assert '的' not in tokens
    # 有意义的词应保留
    assert any('今' in t for t in tokens)


# ============================================================
# 测试6: 空文件处理
# ============================================================
def test_read_empty_file(tmp_path):
    """测试读取空文件，应返回空字符串"""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding='utf-8')

    processor = TextProcessor()
    content = processor.read_file(str(empty_file))

    assert content == ""
