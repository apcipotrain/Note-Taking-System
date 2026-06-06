# Note-taking System: 基于多模态 Agent 与 RAG 的手写笔记智能转换系统

## 项目简介

**Note-taking System** 旨在解决手写笔记数字化过程中的"信息丢失"与"格式凌乱"问题。不同于传统 OCR 仅进行文字识别，本项目利用 **Qwen3-vl-plus 多模态视觉能力**提取手写笔记中的语义颜色（如红笔重点、马克笔高亮），并结合 **RAG（检索增强生成）** 技术，自动检索专业知识库以修正识别偏差并补充背景知识，最终生成结构清晰、美观的 Markdown 电子文档。

2.0 版本引入了 **Agent 质量校验闭环**，能够针对识别结果进行语义审校。如果系统发现识别内容遗漏了红笔批注、公式或信息密度不足，会通过**回流（Looping）机制**携带错误建议进行二次重试，直至产出覆盖全面且表达简洁的 Markdown 电子档。超过重试上限后触发熔断兜底，保证用户始终能得到输出。

---

## 核心亮点

- **多模态语义提取**：通过 Prompt Engineering 实现了对颜色元数据的捕获，能够自动将红笔标注转换为 Markdown 高亮语法（`==text==`）或加粗（`**text**`），蓝色补充说明转为斜体。
- **Agentic Workflow**：基于 **LangGraph** 构建有向无环图（DAG），将视觉感知（Vision）、质量审校（Validator）、知识检索（RAG）、文档排版（Formatter）解耦为独立的智能体节点，支持条件路由、自回流和熔断兜底。
- **RAG 知识对齐**：自建 **76 条专业领域知识库**（覆盖 ML/AI、数学、计算机网络、数据库、操作系统等），采用多查询变体融合 + 关键词重排序策略，**Recall@5 达到 96.6%**。检索到的背景知识用于术语修正和文末知识补充。
- **端到端交互**：基于 **Streamlit** 开发了可视化 Web 界面，支持实时上传图片、处理进度监控及 Markdown 在线预览与下载。
- **可量化的检索评估**：`tests/eval_recall.py` 提供 22 个测试用例的 Recall@5 / Precision@5 / MRR 自动化评估框架。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| Agent 编排 | LangGraph |
| 视觉识别 | Qwen3-vl-plus（阿里灵积） |
| 文本生成 | Qwen-plus |
| 向量嵌入 | Text-embedding-v2（1536 维） |
| 向量数据库 | ChromaDB（HNSW 索引） |
| 前端交互 | Streamlit |
| 评估框架 | 自建 eval_recall（22 测试用例） |

---

## 项目结构

```
Note-taking-System/
├── app/                      # UI 交互层
│   └── main.py               # Streamlit 启动程序
├── core/                     # 核心逻辑层
│   ├── agents/
│   │   └── formatter.py      # Markdown 排版 Agent
│   ├── workflow.py           # LangGraph 工作流定义（DAG + 路由 + 回流）
│   └── multimodal.py         # 多模态视觉识别（Qwen3-vl-plus）
├── engine/                   # RAG 支撑层
│   ├── retriever.py          # 多查询融合检索器
│   └── vector_db.py          # ChromaDB 向量库封装
├── scripts/                  # 工具脚本
│   └── seed_knowledge_base.py  # 知识库初始化（76 条）
├── data/                     # 数据层
│   ├── input/                # 待处理手写图片
│   └── vector_db/            # ChromaDB 持久化存储
├── tests/                    # 测试
│   ├── eval_recall.py        # RAG 召回率评估（Recall@5 / Precision@5 / MRR）
│   ├── eval_rag.py           # 早期 RAG 测试
│   ├── test3.py              # Agent 迭代回流单元测试
│   └── debug_api.py          # 端到端工作流调试
├── requirements.txt          # Python 依赖
├── key.env                   # API 密钥配置（不纳入版本控制）
└── README.md
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置密钥

在项目根目录创建 `key.env` 文件：

```
DASHSCOPE_API_KEY=your_aliyun_sk_xxx
```

### 3. 初始化知识库（首次运行）

```bash
python scripts/seed_knowledge_base.py
```

### 4. 启动应用

```bash
python -m streamlit run app/main.py
```

### 5. 运行召回率评估

```bash
python tests/eval_recall.py
```

---

## 系统工作流

```
[上传图片]
     │
     ▼
┌─────────────────┐
│  Vision Node     │  Qwen3-vl-plus 识别文字 + 颜色 + 公式 → 结构化 JSON
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Validator Node  │  检查覆盖度（颜色/公式）+ 简洁度（平均长度）
└────────┬────────┘
         │
    ┌────┼────┐
    │    │    │
   通过  未通过  未通过
    │   <3次   ≥3次
    │    │    │
    │    ▼    ▼
    │  带回馈 有数据→兜底通过
    │  重跑   无数据→报错
    │
    ▼
┌─────────────────┐
│  RAG Node        │  多查询变体 → ChromaDB 检索 → 关键词重排序 → Top-5
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Formatter Node  │  融合识别结果 + 参考知识 → Markdown 排版输出
└────────┬────────┘
         │
         ▼
   [Markdown 文件]
```

### 节点说明

| 节点 | 功能 | 输入 | 输出 |
|------|------|------|------|
| Vision | 多模态识别 | 图片路径 + 反馈信息 | 结构化 JSON（type/content/style） |
| Validator | 质量审校 | raw_json | is_valid + error_msg（反馈指令） |
| RAG | 知识检索 | raw_json | 5 条最相关参考知识 |
| Formatter | Markdown 排版 | raw_json + context | final_markdown |

### 回流机制

- 最多重试 **3 次**
- 每次重试时，Validator 生成的改进建议（如"未发现颜色细节，请检查是否有红笔/荧光笔标记被忽略"）会注入到 Vision Node 的 Prompt 开头
- 3 次后仍未通过：有数据则触发**熔断兜底**，强制进入排版流程；无数据则终止并报错

---

## RAG 召回率评估

当前评估结果（22 个测试用例，Top-K=5）：

| 指标 | 数值 | 说明 |
|------|------|------|
| Recall@5 | **96.59%** | 预期关键词出现在前 5 条结果中的比例 |
| Precision@5 | **54.70%** | 前 5 条结果中相关结果的比例 |
| MRR | **100%** | 首个相关结果排名的倒数均值 |

---

## 未来计划

- [ ] 引入 LLM-as-a-Judge 替代启发式 Validator，实现语义级别的质量审校
- [ ] 收集 50-100 张标注笔记构建 ground truth 测试基准
- [ ] 支持本地部署多模态模型（Qwen2.5-VL / LLaVA）以保护数据隐私
- [ ] 将 Streamlit 替换为 FastAPI + 现代前端框架，支持并发访问
- [ ] 引入模型路由：简单笔记走轻量模型，复杂笔记走强模型，降低 API 成本
- [ ] 支持增量识别：仅重跑 Validator 标记的不合格部分

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE) 文件。
