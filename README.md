# Note-taking System: 基于多模态 Agent 与 RAG 的手写笔记智能转换系统

## 📌 项目简介

**Note-taking System** 旨在解决手写笔记数字化过程中的“信息丢失”与“格式凌乱”问题。不同于传统的 OCR 仅进行文字识别，本项目利用 **Qwen3-vl-plus 多模态视觉能力** 提取手写笔记中的语义颜色（如红笔重点、马克笔高亮），并结合 **RAG (检索增强生成)** 技术，自动检索专业知识库以修正识别偏差并补充背景知识，最终生成结构清晰、美观的 Markdown 电子文档。

2.0版本引入了 **Agent 质量校验闭环**，能够针对识别结果进行“语义审校”。如果系统发现识别内容遗漏了红笔批注、公式或信息密度不足，会通过**回流（Looping）机制**携带错误建议进行二次重试，直至产出“全内容覆盖且表达最简洁”的 Markdown 电子档。

------

## 🚀 核心亮点

- **多模态语义提取**：通过 Prompt Engineering 实现了对颜色元数据的捕获，能够自动将红笔标注转换为 Markdown 高亮语法。
- **Agentic Workflow**：基于 **LangGraph** 构建有向无环图 (DAG)，将视觉感知、知识检索、文档排版解耦为独立的智能体节点。
- **RAG 知识对齐**：引入 **ChromaDB** 向量数据库，利用语义搜索修正手写稿中的模糊概念，并自动在笔记末尾生成“知识补充”模块。
- **端到端交互**：基于 **Streamlit** 开发了可视化 Web 界面，支持实时上传、处理进度监控及 Markdown 在线预览。

- **🔄 语义纠错回流 (Self-Reflexion Loop)**：基于 **LangGraph** 构建非线性工作流。`Validator` 节点会根据“信息覆盖度”和“简洁度”对初稿进行评分。若不达标，将强制触发 `Vision Node` 重跑。
- **🎯 反馈驱动的 Prompt 进化**：在 `multimodal.py` 中实现了反馈注入机制。第二次识别时，模型会接收到类似 *“未检测到颜色标注，请重点分析红笔部分”* 的具体指令，实现动态优化。
- **📐 全内容 + 最简洁审校标准**：
- - **全内容**：通过特征探测算法，确保颜色元数据（红笔重点）、复杂公式（Latex 意图）不遗漏。
  - **最简洁**：通过 `Formatter Agent` 进行语义重构，合并冗余信息，将凌乱的手写草稿转化为逻辑清晰的排版。
- **📚 知识对齐优化**：结合 **ChromaDB** RAG 链路，利用检索到的专业上下文对 OCR 可能出现的笔误进行术语级修正。

------

## 🛠️ 技术栈

- **核心框架**: LangGraph, LangChain
- **大模型**: Qwen3-vl-plus, Text-embedding-v2
- **向量数据库**: ChromaDB
- **前端交互**: Streamlit (Web UI)
- **数据处理**: Pillow, Dotenv, Pathlib

------

## 📂 项目结构

Plaintext

```
Note-taking-System/
├── app/                # UI 交互层 (Streamlit 界面)
│   └── main.py         # 启动程序
├── core/               # 核心逻辑层 (Agent 决策与视觉识别)
│   ├── agents/         # 各角色 Agent 定义
│   │   ├── formatter.py
│   └── workflow.py     # LangGraph 工作流定义
│   └── multimodal.py   # 封装视觉识别调用 (Qwen3-vl-plus)
├── engine/             # 支撑层 (RAG 检索与向量库管理)
│   ├── retriever.py    # 向量检索逻辑
│   ├── vector_db.py    # 数据库初始化与索引管理
├── data/               # 数据层 (输入图片、输出 MD、参考文档)
│   ├── input/          # 存放待处理的手写图片
│   ├── output/         # 生成的 Markdown 结果
├── tests/              # 测试脚本
├── requirements.txt    # 依赖项
└── key.env             # 环境变量
```

------

## 📦 快速开始

1. **安装依赖**

   ```Bash
   pip install -r requirements.txt
   ```

2. **配置密钥** 在项目根目录创建 `key.env` 文件：

   ```Plaintext
   DASHSCOPE_API_KEY=your_aliyun_sk_xxx
   ```

3. **运行应用** 请在项目根目录下执行：

   ```Bash
   python -m streamlit run app/main.py
   ```



------

## 🧠 系统工作流

1. **Vision Node**: 接收手写图片，输出包含内容、颜色、样式的结构化 JSON。
2. **RAG Node**: 提取关键词，在 `data/reference` 库中进行向量检索，获取背景知识。
3. **Formatter Agent**: 融合视觉数据与背景知识，生成符合 Markdown 标准的电子笔记。

------

## 📝 未来计划

- [ ] 支持手写数学公式的 LaTeX 自动转换。
- [ ] 集成 Notion / Obsidian API，实现一键同步。
- [ ] 引入本地部署的多模态模型（如 LLaVA / Qwen-VL）以保护数据隐私。