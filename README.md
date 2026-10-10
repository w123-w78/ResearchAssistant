# 🔬 智能科研助理 —— 基于 RAG + Agent 的论文知识库问答系统

面向科研人员的私有文献问答系统：上传论文 → 自动构建混合检索索引 →
Agent 自主规划工具调用 → 生成带引用溯源、可核查的学术回答。



---

## 一、技术栈

| 层次 | 技术选型 | 说明 |
|------|---------|------|
| 大模型 | **阿里云 DashScope（通义千问 qwen-plus）** | OpenAI 兼容接口，中文学术问答效果好 |
| 向量模型 | **DashScope text-embedding-v3**（1024 维） | 与 LLM 共用同一 API Key |
| 重排序 | **gte-rerank-v2**（在线交叉编码） | 失败自动降级为本地启发式精排 |
| 存储 | **SQLite** | 文档元数据 / 分块 / 对话记忆 / 语义缓存 / 请求日志 |
| 检索 | BM25 + 向量双路召回 + RRF 融合 + 精排 | 三层递进架构 |
| Agent | 手写 ReAct 循环（约 200 行核心逻辑） | 9 个工具、并行调用、错误恢复、死循环检测 |
| 前端 | Streamlit | 三栏布局 + 推理过程可视化 |

**单一接入方式**：系统只使用 DashScope 一种模型接入，无本地/云端多套分支，
配置与代码路径都保持最简。

---

## 二、快速开始

### 1. 安装依赖

```bash
conda create -n research_assistant python=3.10
conda activate research_assistant
pip install -r requirements.txt
```

### 2. 配置 API Key（二选一）

```bash
# 推荐：环境变量（不会误提交到仓库）
set DASHSCOPE_API_KEY=sk-xxxxxxxxxxxx        # Windows
export DASHSCOPE_API_KEY=sk-xxxxxxxxxxxx     # Linux / macOS

# 或者：填写 config.yaml 的 llm.api_key
```

> Key 获取：阿里云百炼控制台 → API-KEY 管理。复制 `.env.example` 为 `.env` 亦可。

### 3. 启动

```bash
# Web 界面（推荐）
streamlit run src/frontend/app.py

# 或命令行演示（自动生成示例论文并跑通全链路）
python scripts/cli_demo.py

# 离线冒烟测试（不需要 API Key）
python scripts/smoke_test.py
```

启动后访问 `http://localhost:8501`，在左侧上传论文即可开始问答。

---

## 三、目录结构

```
ResearchAssistant/
├── config.yaml                 # 全局配置（模型/检索/Agent/存储参数）
├── requirements.txt
├── src/
│   ├── data_loader/            # 模块一：多格式文档加载（PDF/Word/TXT/MD）
│   ├── chunking/               # 模块一：4 种分块策略 + 对比实验工具
│   ├── retrieval/              # 模块一：向量库 / BM25 / 混合检索 / 重排 / 评测指标
│   ├── generation/             # 模块二：Prompt / RAG流水线 / 流式 / 语义缓存
│   ├── agent/                  # 模块三：ReAct循环 / 工具集 / 路由 / 记忆管理
│   ├── storage/                # SQLite 持久化（6 张表）
│   ├── frontend/               # 模块四：Streamlit 界面 + 组件
│   ├── system.py               # 模块四：系统集成门面（对外唯一入口）
│   └── utils/                  # 配置 / 日志 / LLM / Embedding / 文本工具
├── scripts/                    # CLI 演示、示例语料、冒烟测试
├── tests/                      # pytest 用例
├── docs/                       # 技术设计文档、用户手册
├── reports/                    # 评测集、Bad Case 分析
└── docker/                     # Dockerfile + docker-compose
```

---

## 四、各模块实现要点

### 模块一：文档处理与检索层
- **加载**：PDF 优先 PyMuPDF（`sort=True` 解决双栏乱序），表格转 Markdown 防跨页切碎；
  Word 用 python-docx（未安装时降级为内置 XML 解析）；TXT/MD 自动探测编码
- **分块**：`fixed` / `recursive` / `semantic` / `academic` 四种策略，
  `compare_chunkers()` 一键产出《分块策略对比实验报告》数据
- **检索**：向量（余弦）+ BM25（k1=1.5, b=0.75）双路召回 → RRF(k=60) 融合 →
  gte-rerank-v2 精排 Top-5；关闭重排可做消融对照
- **评测**：`src/retrieval/evaluate.py` 提供 Hit@5 / MRR / Recall@K / Precision@K

### 模块二：RAG 生成层
- **Prompt**：强制引用标注 + 允许"不确定就直说"，显著降低幻觉
- **流式**：`StreamMerger` 边出答案边发引用事件，前端打字机 + 引用动态浮现
- **缓存**：语义缓存（余弦 ≥ 0.93 命中）存 SQLite，知识库变更自动失效
- **降级**：无结果 → 纯模型+提示；低相关性 → 带警告回答；API 失败 → 友好提示

### 模块三：Agent 决策层
- **ReAct**：手写 `Thought → Action → Observation` 循环，支持并行工具调用
- **工具（9 个）**：知识库检索、论文元信息、论文对比、关键词提取、结构化摘要、
  当前时间、安全计算器（AST 解析，非 eval）、联网搜索、知识库统计
- **路由**：规则路由命中时跳过 ReAct，省约 80% Token
- **容错**：工具重试、死循环检测、超迭代强制收敛
- **记忆**：SQLite 存完整历史；内存工作集做滑动窗口 + 摘要压缩（只影响送模型的上下文）

### 模块四：系统集成与前端
- `ResearchAssistantSystem` 是唯一门面，CLI / Web / 测试共用
- 三栏界面：知识库管理 / 对话（流式+引用）/ Agent 推理轨迹可视化
- `health_check()` 检查 LLM、向量库、Embedding、数据库、工具集状态
- `request_logs` 表记录每次问答，模块五评测可直接 SQL 聚合

---

## 五、常用配置（config.yaml）

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `llm.model` | `qwen-plus` | 可换 `qwen-turbo`（快）/ `qwen-max`（强） |
| `embedding.model` | `text-embedding-v3` | 维度 1024（可改 768/512） |
| `storage.db_path` | `./data/assistant.db` | SQLite 路径 |
| `chunking.strategy` | `recursive` | 论文场景建议改 `academic` |
| `retrieval.top_k` / `final_top_k` | 20 / 5 | 召回候选数 / 进 Prompt 片段数 |
| `retrieval.reranker.provider` | `dashscope` | 改 `heuristic` 可离线运行 |
| `agent.max_iterations` | 6 | ReAct 最大轮次 |
| `cache.similarity_threshold` | 0.93 | 语义缓存命中阈值 |

所有参数支持环境变量覆盖，如 `APP_LLM__MODEL=qwen-max`。

---

## 六、常见问题

**Q：没有 API Key 能跑吗？**
A：`python scripts/smoke_test.py` 可验证存储/分块/检索/路由（34 项，无需 Key）；
完整问答需要配置 DashScope Key。

**Q：重排一定要联网吗？**
A：默认调用在线 gte-rerank-v2；把 `retrieval.reranker.provider` 改为 `heuristic`
即可用本地启发式精排，完全离线。

**Q：换模型要改代码吗？**
A：不用，只改 `config.yaml` 的 `llm.model` 即可。

**Q：知识库数据存在哪？**
A：文档元数据与分块在 SQLite；向量索引在 `data/index/`（pickle），
两者都由 `system.save_index()` 持久化。
