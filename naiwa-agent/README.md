# GeneralTaskAgent · 奶蛙 Agent

> 基于 **LangChain 1.x + ReAct** 的通用任务型 Agent——同一套后端，两种形态：
>
> - **CLI 终端版**：`python main.py`，过程框 + 结果面板，token 用量全程可见；
> - **桌面客户端「奶蛙 Agent」**：Electron 桌面应用，开启动画、昼夜双主题、多会话、工作日记。
>
> 能力：**代码开发**（读写文件 / 精确编辑 / 执行命令）、**文档生成**（Word / Excel / PPT）、
> **联网搜索**、**多模型热切换**（DeepSeek / 通义千问 / Kimi / 本地 Ollama，支持自动发现）、
> **长任务上下文自动压缩**、**token 用量与成本统计**、**双层缓存**，全程可观测。
> 125 个单元测试全部通过。

---

## 目录

1. [功能特性](#一功能特性)
2. [快速开始（CLI 版）](#二快速开始cli-版)
3. [桌面客户端「奶蛙 Agent」](#三桌面客户端奶蛙-agent)
4. [配置说明](#四配置说明)
5. [CLI 命令一览](#五cli-命令一览)
6. [架构设计](#六架构设计)
7. [模型系统：多模型热切换 + 自动发现](#七模型系统多模型热切换--自动发现)
8. [Token 用量：事前预估 + 事后实耗](#八token-用量事前预估--事后实耗)
9. [长任务不断线：上下文自动压缩](#九长任务不断线上下文自动压缩)
10. [缓存 / RAG 知识库 / MCP 外部工具](#十缓存--rag-知识库--mcp-外部工具)
11. [示例：Agent 自己写的贪吃蛇](#十一示例agent-自己写的贪吃蛇)
12. [运行测试](#十二运行测试)
13. [常见问题 FAQ](#十三常见问题-faq)
14. [项目结构](#十四项目结构)

---

## 一、功能特性

| 能力 | 说明 |
|------|------|
| ReAct 自主循环 | 思考 → 行动 → 观察，自主读文件、改代码、跑命令、出文档 |
| 工具防幻觉 | 工具注册表严格匹配，调错名字时用编辑距离给出"你是否想调用 xxx"建议 |
| 多模型热切换 | 对话中 `/use` 秒换模型，**无需重启、记忆保留**；一家 Key 自动拉全系模型 |
| 长任务不中断 | 历史接近模型窗口时自动凝练为任务摘要，在原会话内继续跑 |
| token 全程可见 | 每轮事前预估 + 事后实耗（API 真实返回值）+ 成本折算（¥） |
| 双层缓存 | 相同模型请求零 token 复现；幂等工具结果 TTL 缓存 |
| 安全阀 | 模型/工具调用次数上限、限流（频率+并发）、危险命令黑名单、凭据自动掩码 |
| 可观测 | 每个 ReAct 步骤结构化落盘（`data/agent.log`），`/trace` 一键回看全过程 |
| 桌面客户端 | Electron 桌面端「奶蛙 Agent」：动画开场、昼夜主题、多会话、工作日记、Key 加密存储 |

---

## 二、快速开始（CLI 版）

### 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows / macOS / Linux |
| Python | **3.10 ~ 3.13**（⚠️ 暂不支持 3.14，部分依赖没有对应 wheel） |
| 网络 | 可访问 `api.deepseek.com` |
| DeepSeek API Key | 必需，[申请地址](https://platform.deepseek.com)，新用户有免费额度 |

### 安装

```powershell
cd GeneralTaskAgent

# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活（PowerShell 示例；CMD 用 activate.bat，macOS/Linux 用 source .venv/bin/activate）
.venv\Scripts\Activate.ps1

# 3. 安装依赖（约 1~3 分钟；国内网络慢可加 -i https://mirrors.aliyun.com/pypi/simple/）
pip install -r requirements.txt
```

### 配置

```powershell
copy .env.example .env
notepad .env        # 只需填 DEEPSEEK_API_KEY，其余都有安全默认值
```

### 运行

**方式 1（最省事）**：双击根目录 `启动.bat`。
**方式 2（命令行）**：

```powershell
.venv\Scripts\python.exe main.py
```

启动后直接打字对话：

```
你> 帮我在桌面创建一个 hello.py，写一个猜数字游戏，然后跑一下测试
┌─ Agent 思考与行动过程 ──────────────────┐
│ 需要先创建文件...                        │   ← 固定高度过程框，
│ ⚙ 调用工具 write_file {...}             │     只显示最近 N 行，
│ ✔ write_file: 已写入 hello.py           │     完整过程不丢
│ ...                                     │
└─────────────────────────────────────────┘
┌─ 结果 ──────────────────────────────────┐
│ 文件已创建并测试通过，共 3 个用例全绿。   │   ← 最终答案紧贴框下
└─────────────────────────────────────────┘
⚡ 预计消耗 ~3,157 tokens · ¥0.0028        ← 事前预估
💰 本次实耗 3,003 tokens · ¥0.0026        ← 事后实耗（API 真实返回）
```

---

## 三、桌面客户端「奶蛙 Agent」

桌面端与 CLI **共用同一套后端**（`agent/` 包），通过 `client/bridge.py` 以 JSON-Lines 协议通信，API Key 只经进程内存传递、用系统 `safeStorage` 加密存储，不写入任何仓库文件。

```powershell
# 前置：先按上面的 CLI 步骤装好根目录 .venv（客户端复用同一环境）
cd client
npm install
npm start          # 或双击 client\start.bat（会自动检查依赖）
```

客户端功能：

- **开启动画**：奶龙与奶蛙的名场面（可跳过）；
- **对话工作台**：多会话管理（新建 / 切换 / 删除）、历史对话侧栏；
- **思考与行动过程框**：思考、推理、工具调用与结果折叠展示，默认收起，点击展开；答案与过程分离，支持 Markdown 渲染（表格 / 代码块 / 列表）；
- **模型与设置**：三家供应商（DeepSeek / 通义千问 / Kimi）Key 分别保存，模型列表实时切换；
- **昼夜双主题**：白天 / 夜间壁纸与界面配色一键切换；
- **工作日记**：随手记录工作内容，本地保存；
- **打包分发**：`npm run dist` 用 electron-builder 打 Windows 安装包（产物在 `client/dist/`，已默认不入库）。

> 客户端默认调用仓库根目录的 `.venv`（见 `client/main.js` 的 `PYTHON` 路径）——所以请先完成 CLI 版的环境安装。

---

## 四、配置说明

`.env` 复制自 `.env.example`。**必须修改的只有 `DEEPSEEK_API_KEY`**，常用可调项：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAIN_MODEL` / `REASONER_MODEL` | `deepseek-chat` / `deepseek-reasoner` | 对话 / 推理模型 |
| `TEMPERATURE` | `0.2` | 工具场景建议低温，减少幻觉 |
| `MODEL_RUN_LIMIT` | `60` | 单次任务最大模型调用数（安全阀） |
| `TOOL_RUN_LIMIT` | `100` | 单次任务最大工具调用数（安全阀） |
| `TOOL_RETRIES` | `3` | 工具失败自动重试次数（指数退避） |
| `CALLS_PER_MINUTE` / `MAX_CONCURRENCY` | `60` / `4` | 限流：频率 / 并发 |
| `SUMMARIZE_TRIGGER_TOKENS` | `16000` | 窗口未知模型的压缩兜底阈值 |
| `SUMMARIZE_KEEP_MESSAGES` | `20` | 压缩时保留最近多少条原文 |
| `RAG_ENABLED` | `false` | 知识库开关（需装 `requirements-rag.txt`） |

模型目录在 `config/models.json` 维护：窗口大小、输出上限、价格（元/百万 token）、密钥环境变量名都在里面，**加新模型只改 JSON，不用改代码**，`/reload-models` 热重载生效。

---

## 五、CLI 命令一览

| 命令 | 作用 |
|------|------|
| `/help` | 帮助 |
| `/tools` | 列出所有已注册工具 |
| `/models` | 列出可切换模型（标注当前 ● / 是否已配密钥） |
| `/use <模型id>` | 切换模型（立即生效、无需重启、记忆保留） |
| `/reload-models` | 改完 `config/models.json` 后热重载模型目录 |
| `/discover` | 从各已配置 Key 的供应商拉取全部可用模型 |
| `/trace` | 用记事本打开完整过程日志 `data/trace.log` |
| `/usage` | 本会话累计 token 用量与成本 |
| `/context` | 当前上下文占用、压缩触发点与剩余空间 |
| `/new` | 开启新会话（清空上下文与统计） |
| `/exit` | 退出 |

---

## 六、架构设计

```
┌──────────────────────────── 两种入口，一套后端 ───────────────────────────┐
│                                                                          │
│   main.py（CLI）              client/（Electron 桌面客户端）              │
│        │                            │  main.js ── JSON-Lines ── bridge.py│
│        └──────────┬─────────────────┘                                    │
│                   ▼                                                      │
│            agent/api.py（门面：list/switch/reload/discover）             │
│                   │                                                      │
│            agent/runtime.py（AgentRuntime：切换模型时重建 Agent，        │
│             │                     但复用同一 checkpointer → 记忆不丢）   │
│             ▼                                                           │
│            agent/core.py（build_agent：项目心脏）                        │
│     ┌─────────┼──────────────┬───────────────┬──────────────┐          │
│     ▼         ▼              ▼               ▼              ▼          │
│  模型层     工具层        中间件链        会话记忆        缓存层         │
│ models.py  tools/       （顺序即优先级）  AsyncSqlite    memory/       │
│ （多供应商  registry    ModelCallLimit   Saver（SQLite） cache.py     │
│  +自动发现） coding      ToolCallLimit   跨进程持久化     LLM缓存+TTL   │
│            documents    RateLimit                      rag.py(可选)   │
│            search       ToolError→Retry→ModelRetry                     │
│            mcp(可选)    Summarization→TodoList                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**中间件链**（`agent/core.py`，越靠前越先拦截）：

1. `ModelCallLimit` / `ToolCallLimit` —— 安全阀，未执行就省 token；
2. `RateLimit`（自定义）—— 令牌间隔 + 并发信号量；
3. `ToolError` —— 把异常转成"模型能看懂并修正"的提示（含工具名纠错建议）；
4. `ToolRetry` / `ModelRetry` —— 指数退避自动重试；
5. `Summarization` —— 长对话上下文压缩（详见第九节）；
6. `TodoList` —— 复杂任务自动规划。

**安全设计**：

- 危险命令黑名单（`rm -rf /`、`format c:` 等直接拦截）；
- 读文件时 `key/secret/token/password` 赋值自动掩码；
- System Prompt 内置凭据卫生准则（不转录凭据、不推断真伪）、防幻觉准则（数据不足直说，严禁编造实时信息）；
- 无判定标准的模糊指令（"整理一下""优化一下"）先反问给方案，再动手。

---

## 七、模型系统：多模型热切换 + 自动发现

- **目录驱动**：`config/models.json` 维护全部模型，默认提供 DeepSeek（对话/推理）、通义千问、Kimi、本地 Ollama 五条；
- **热切换**：`/use deepseek-reasoner` 立即生效，**对话记忆保留**（切换失败自动回滚，原模型原封不动）；
- **自动发现**：启动时调各家 `GET /models` 接口，把"你的 Key 能用的全部模型"注册进列表——厂商上新你不用改配置；断网时用上次发现结果（`data/discovered_models.json`，不含密钥）；
- **扩展**：已有供应商加新模型只改 JSON；全新厂商用 `register_provider()` 注册一个构造器。

---

## 八、Token 用量：事前预估 + 事后实耗

每轮对话自动显示两行：

```
⚡ 预计消耗 ~3,157 tokens (输入 ~2,901 / 输出 ~256，其中历史对话 ~0) · ¥0.0028
💰 本次实耗 3,003 tokens (输入 2,878 / 输出 125，模型调用 1 次) · ¥0.0026 · 会话累计 …
```

- **事前预估**：系统提示 + 工具定义 + 当前历史 + 本次输入，按 DeepSeek 官方口径（英文 0.3 token/字符、中文 0.6）启发式估算，经真实调用标定，误差约 ±10~20%；
- **事后实耗**：从 API 返回的 `usage_metadata` 汇总真实用量（含 ReAct 中间步的多次调用）；
- **成本折算**：按 `config/models.json` 的 `pricing` 字段（元/百万 token，输入按缓存未命中保守口径），可自行修改；本地 Ollama 显示免费。

---

## 九、长任务不断线：上下文自动压缩

复杂任务会把历史撑到模型窗口上限。本 Agent 的应对不是"开新对话"，而是**在原会话内把历史凝练成任务摘要，腾出空间继续跑**，用户无感知、任务不中断：

```
历史消息接近触发点
      ↓
SummarizationMiddleware 把旧历史总结为结构化摘要：
   ① 会话目标（原文保留）  ② 已完成的工作与关键决策
   ③ 产物清单（文件路径）  ④ 下一步计划
      ↓
新上下文 = 系统提示 + 任务摘要 + 最近 20 条原文消息 → 循环继续
```

**触发点按模型窗口预算自动计算**（不是写死的数字）：

```
触发点 = 模型窗口 - 单次最大输出 - 静态开销 - 安全余量
```

换更大窗口的模型时触发点自动抬高；窗口未知的模型（如自动发现）用 `.env` 绝对阈值兜底。以 deepseek-chat（128K 窗口）标定，触发点约 11 万 token，请求永远落在窗口内。

**怎么确认它在工作**：每轮压缩会打印 `📦 上下文已自动凝练 …`；`/context` 随时查看占用与触发点；`/usage` 显示本会话压缩次数。

---

## 十、缓存 / RAG 知识库 / MCP 外部工具

### 双层缓存（`memory/cache.py`）

- **LLM 缓存**：完全相同的模型请求直接命中，零 token 消耗（SQLite 落盘，跨进程可用）；
- **工具 TTL 缓存**：幂等工具（搜索、读文件等）10 分钟内复用结果，只缓存无副作用调用。

### RAG 知识库（可选）

```powershell
pip install -r requirements-rag.txt      # .env 中 RAG_ENABLED=true
python scripts/ingest.py 笔记.txt 论文.md  # 或整个目录：python scripts/ingest.py ./docs
```

对话时 Agent 自动调用 `rag_search` 检索后回答；未装依赖时自动降级，其余功能不受影响。

### MCP 外部工具（可选）

```powershell
pip install mcp langchain-mcp-adapters
```

```python
from tools.mcp_client import load_mcp_tools
tools = await load_mcp_tools(
    ["npx", "-y", "@modelcontextprotocol/server-filesystem", "D:/workspace"]
)
agent = build_agent(extra_tools=tools)
```

---

## 十一、示例：Agent 自己写的贪吃蛇

`examples/snake/` 里的三个贪吃蛇（tkinter 版 / HTML 经典版 / HTML 增强版）**全部由本 Agent 自己写代码生成**，是对"代码开发 + 自己跑测试"能力的一次真实记录——`_test_snake_plus.js` 是它为自己写的浏览器环境 mock 测试。

```powershell
python examples/snake/snake_game.py     # tkinter 版，零第三方依赖
# 或用浏览器打开 examples/snake/snake_plus.html
```

---

## 十二、运行测试

```powershell
# 项目根目录、虚拟环境激活（或 .venv\Scripts\python.exe -m pytest）
pytest tests/ -v
```

当前 **125 个测试全部通过**，覆盖：工具注册与模糊匹配（防幻觉）、文件读写与危险命令拦截、文档生成、缓存命中与过期、配置校验、模型目录与自动发现、token 预估/实耗/偏差、上下文压缩预算、过程框 UI 等。

---

## 十三、常见问题 FAQ

**Q1：启动报错 `未配置 DEEPSEEK_API_KEY`**
`.env` 没建或没填。`copy .env.example .env` 后填入真实 Key，`.env` 必须放在项目根目录（与 `main.py` 同级）。

**Q2：`pip install` 报 `No matching distribution found`**
九成是 Python 版本问题：本项目要求 3.10~3.13。确认版本后用 3.13 重建环境：`py -3.13 -m venv .venv`。

**Q3：`pip install` 很慢**
换国内镜像：`pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/`（部分镜像快照不及时会误报"找不到 langchain 1.x"，优先阿里云源）。

**Q4：模型调用报 `401 Unauthorized`**
Key 错误或过期，去 DeepSeek 控制台确认（注意复制时别带空格）。

**Q5：报 `429 Rate limit`**
调小 `CALLS_PER_MINUTE`（如 10）、`TOOL_RETRIES` 调到 5。

**Q6：任务做到一半提示"达到调用上限"**
安全阀生效（默认 60/100）。确属正常长任务时调大 `MODEL_RUN_LIMIT` / `TOOL_RUN_LIMIT`。

**Q7：生成的 Word/Excel 打不开**
文件正被 Office/WPS 打开着，关掉后重试。

**Q8：中文/图标乱码**
Windows 老终端编码问题，用 Windows Terminal 或先 `chcp 65001`。

**Q9：想看 Agent 到底做了什么**
`data/agent.log` 是结构化 JSON 行日志；输入 `/trace` 打开全过程流水。

**Q10：会话历史存在哪？如何重置**
`data/checkpoint.sqlite`。删掉整个 `data/` 目录即可完全重置（缓存、日志、向量库都在里面，删前确认无重要数据）。注意 `data/` 是运行时数据，默认不入库。

**Q11：桌面客户端启动报"未找到项目 .venv"**
客户端复用根目录虚拟环境，先完成 CLI 版安装（`python -m venv .venv && pip install -r requirements.txt`），再进 `client/` 执行 `npm install && npm start`。

---

## 十四、项目结构

```
GeneralTaskAgent/
├── main.py                   # CLI 入口：配置校验 → 组装 Agent → 对话循环
├── agent/                    # 后端核心
│   ├── api.py                #   对外门面（模型查询/切换/重载/发现）
│   ├── runtime.py            #   Agent 运行时（热切换时重建 Agent、复用记忆）
│   ├── core.py               #   Agent 组装（中间件链，项目心脏）
│   ├── models.py             #   模型管理器（多供应商 + 自动发现 + 语义化异常）
│   ├── config.py             #   配置中心（.env 加载 + 启动校验）
│   ├── token_tracker.py      #   token 事前预估 + 事后实耗 + 成本折算
│   ├── observability.py      #   结构化日志 + 流式输出编排
│   ├── ui.py                 #   过程框 UI（Rich Live）+ /trace
│   └── prompts.py            #   System Prompt（行为准则）
├── client/                   # Electron 桌面客户端「奶蛙 Agent」
│   ├── main.js               #   主进程（窗口 / IPC / 子进程管理 / Key 加密）
│   ├── preload.js            #   预加载（contextBridge 安全暴露 API）
│   ├── bridge.py             #   JSON-Lines 桥接服务（Electron ↔ Python 后端）
│   ├── renderer/             #   渲染层（界面 / 主题 / Markdown / 过程框）
│   ├── assets/               #   客户端素材（角色图 / 壁纸 / 动画）
│   ├── package.json          #   electron + electron-builder 配置
│   └── start.bat             #   一键启动（自动检查依赖）
├── tools/                    # 工具层（@register 注册表 + 模糊匹配防幻觉）
│   ├── coding.py             #   读写文件 / 精确编辑 / 目录 / Shell（危险命令拦截）
│   ├── documents.py          #   Word / Excel / PPT 生成
│   ├── search.py             #   联网搜索（TTL 缓存 + 失败熔断）
│   ├── mcp_client.py         #   MCP 外部工具（可选）
│   └── registry.py           #   注册表 + 编辑距离纠错
├── memory/
│   ├── cache.py              # LLM 缓存 + 工具 TTL 缓存
│   └── rag.py                # 向量知识库（可选，自动降级）
├── middleware/
│   └── rate_limit.py         # 自定义限流（频率 + 并发）
├── config/models.json        # 模型目录（窗口 / 价格 / 密钥变量名）
├── scripts/ingest.py         # RAG 入库脚本
├── tests/                    # pytest，125 个用例
├── examples/snake/           # Agent 自己写的贪吃蛇 demo
├── requirements.txt          # 基础依赖
├── requirements-rag.txt      # RAG 扩展依赖（可选）
├── .env.example              # 配置模板（复制为 .env 填写）
└── 启动.bat                  # CLI 一键启动
```

> 运行时数据（`.env`、`.venv/`、`data/`、`client/node_modules/`、`client/dist/`）均已被 `.gitignore` 排除，不会入库。

---

## License

MIT © 2026 马肇哲
