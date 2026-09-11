# 论文查重程序

基于余弦相似度的中文论文查重工具。

## 算法说明

采用 **余弦相似度** 算法计算两篇文本的重复率：

1. 使用 `jieba` 对文本进行中文分词
2. 过滤停用词和标点符号
3. 统计词频，构建词频向量
4. 计算两个向量的余弦夹角，即为重复率

## 项目结构

```
学号/
├── main.py                    # 程序入口，接收命令行参数
├── text_processor.py          # 文本预处理器（读取、分词、去停用词）
├── similarity_calculator.py   # 相似度计算器（余弦相似度算法）
├── requirements.txt           # Python 依赖
├── stop_words.txt             # 中文停用词表
├── conftest.py                # pytest 配置
├── data/                      # 测试样例数据
│   ├── orig.txt               # 原文样例
│   ├── orig_add.txt           # 抄袭版样例（增删改）
│   ├── orig_same.txt          # 与原文完全相同
│   ├── orig_diff.txt         # 与原文完全不同
│   ├── empty.txt              # 空文件
│   └── punctuation.txt        # 纯标点符号
└── tests/                     # 单元测试
    ├── test_text_processor.py
    ├── test_similarity_calculator.py
    └── test_main.py
```

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行程序

```bash
python main.py [原文文件路径] [抄袭版文件路径] [答案文件路径]
```

示例：

```bash
python main.py C:\tests\orig.txt C:\tests\orig_add.txt C:\tests\ans.txt
```

输出文件 `ans.txt` 中包含重复率（浮点数，精确到小数点后两位，如 `0.85`）。

### 运行单元测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行测试并查看覆盖率
pytest tests/ --cov=. --cov-report=term-missing
```

## 模块说明

### TextProcessor（文本预处理器）

| 方法 | 功能 |
|------|------|
| `read_file(path)` | 读取文件，自动适配 UTF-8/GBK 编码 |
| `tokenize(text)` | 使用 jieba 进行中文分词 |
| `remove_stopwords(tokens)` | 过滤停用词和空白字符 |
| `process(path)` | 完整预处理流程（读取→分词→去停用词） |

### SimilarityCalculator（相似度计算器）

| 方法 | 功能 |
|------|------|
| `build_vectors(tokens1, tokens2)` | 构建词频向量 |
| `cosine_similarity(vec1, vec2)` | 计算余弦相似度 |
| `calculate_similarity(path1, path2)` | 完整查重流程，返回重复率 |

## 异常处理

| 异常类型 | 场景 | 处理方式 |
|---------|------|---------|
| FileNotFoundError | 输入文件路径不存在 | 输出错误信息到 stderr，退出码 1 |
| UnicodeDecodeError | 文件编码无法识别 | 尝试多种编码，全部失败则报错 |
| ZeroDivisionError | 分词后向量为零 | 返回相似度 0.00 |
