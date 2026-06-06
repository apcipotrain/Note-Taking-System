"""Seed the ChromaDB knowledge base with domain reference entries.

Run once: python scripts/seed_knowledge_base.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.vector_db import VectorEngine

KNOWLEDGE_ENTRIES = [
    # === 深度学习 / 机器学习 ===
    ("卷积神经网络（CNN）是一种前馈神经网络，核心组件包括卷积层（提取局部特征）、池化层（降采样）和全连接层（分类）。广泛应用于图像识别、目标检测。",
     "cnn_definition"),
    ("循环神经网络（RNN）处理序列数据，通过隐藏状态传递时序信息。LSTM 和 GRU 是其改进变体，解决了长序列中的梯度消失问题。",
     "rnn_definition"),
    ("Transformer 架构基于自注意力机制（Self-Attention），完全摒弃了循环结构。核心公式：Attention(Q,K,V) = softmax(QK^T/√d_k)V。代表模型有 BERT、GPT 系列。",
     "transformer_definition"),
    ("反向传播（Backpropagation）通过链式法则计算损失函数对各层参数的梯度，是训练神经网络的核心算法。",
     "backprop_definition"),
    ("激活函数（Activation Function）引入非线性。常见的有：Sigmoid（输出 0~1）、ReLU（f(x)=max(0,x)）、GELU（GPT 常用）、Softmax（多分类输出概率分布）。",
     "activation_function"),
    ("梯度下降（Gradient Descent）通过沿负梯度方向迭代更新参数来最小化损失函数。变体：SGD、Momentum、Adam、AdamW。",
     "gradient_descent"),
    ("损失函数（Loss Function）衡量模型预测与真实值的差距。分类常用交叉熵（Cross Entropy），回归常用均方误差（MSE）。",
     "loss_function"),
    ("过拟合（Overfitting）指模型在训练集上表现好但在测试集上泛化差。常用对策：正则化（L1/L2）、Dropout、早停（Early Stopping）、数据增强。",
     "overfitting"),

    # === 计算机视觉 ===
    ("图像分类任务：给定一张图片，判断其所属类别。经典数据集：ImageNet（1000 类）、CIFAR-10/100、MNIST。",
     "image_classification"),
    ("目标检测（Object Detection）同时定位和分类图像中的物体。经典算法：R-CNN 系列、YOLO 系列、SSD。",
     "object_detection"),
    ("图像分割分为语义分割（Semantic Segmentation，逐像素分类）和实例分割（Instance Segmentation，区分同类的不同个体）。代表模型：FCN、U-Net、Mask R-CNN。",
     "image_segmentation"),

    # === 自然语言处理 ===
    ("词嵌入（Word Embedding）将词语映射为稠密向量，捕获语义关系。经典方法：Word2Vec（CBOW/Skip-gram）、GloVe。",
     "word_embedding"),
    ("BERT（Bidirectional Encoder Representations from Transformers）使用掩码语言模型（MLM）和下一句预测（NSP）进行预训练，在 11 项 NLP 任务上刷新了记录。",
     "bert"),
    ("GPT（Generative Pre-trained Transformer）是自回归语言模型，通过预测下一个 token 进行预训练。GPT-3 展示了强大的少样本学习能力。",
     "gpt"),
    ("注意力机制（Attention Mechanism）允许模型在处理序列时动态聚焦于不同位置的信息，是 Transformer 的核心创新。",
     "attention_mechanism"),

    # === 数学基础 ===
    ("矩阵乘法：若 A 为 m×n 矩阵，B 为 n×p 矩阵，则 C=AB 为 m×p 矩阵，其中 C_{ij} = Σ_k A_{ik}·B_{kj}。",
     "matrix_multiplication"),
    ("链式法则（Chain Rule）：若 y=f(u), u=g(x)，则 dy/dx = dy/du · du/dx。反向传播的理论基础。",
     "chain_rule"),
    ("概率分布：正态分布 N(μ,σ²) 的密度函数 f(x) = (1/√(2πσ²))·exp(-(x-μ)²/(2σ²))。中心极限定理表明大量独立随机变量之和近似服从正态分布。",
     "normal_distribution"),
    ("贝叶斯定理：P(A|B) = P(B|A)·P(A) / P(B)。在机器学习中广泛用于朴素贝叶斯分类器、贝叶斯优化等。",
     "bayes_theorem"),
    ("信息熵 H(X) = -Σ P(x_i)·log P(x_i) 度量随机变量的不确定性。交叉熵 H(p,q) = -Σ p(x)·log q(x) 是分类任务常用损失。",
     "information_entropy"),
    ("PCA（主成分分析）通过特征值分解将高维数据投影到方差最大的方向，实现降维的同时保留最大信息量。",
     "pca"),
    ("SVD（奇异值分解）将矩阵 A 分解为 UΣV^T，在推荐系统（协同过滤）和图像压缩中有重要应用。",
     "svd"),

    # === 数据结构与算法 ===
    ("时间复杂度用大 O 表示法描述算法运行时间随输入规模增长的趋势。常见：O(1)、O(log n)、O(n)、O(n log n)、O(n²)、O(2^n)。",
     "time_complexity"),
    ("二分查找（Binary Search）在有序数组中每次折半缩小搜索范围，时间复杂度 O(log n)。前提是数据必须有序。",
     "binary_search"),
    ("动态规划（Dynamic Programming）将复杂问题分解为重叠子问题，通过记忆化（Memoization）避免重复计算。关键要素：最优子结构、状态转移方程。",
     "dynamic_programming"),
    ("BFS（广度优先搜索）使用队列逐层遍历图/树，适合求最短路径（无权图）。DFS（深度优先搜索）使用栈（或递归）沿路径深入，适合拓扑排序、连通分量。",
     "bfs_dfs"),
    ("哈希表（Hash Table）通过哈希函数将键映射到桶，平均 O(1) 的查找/插入。冲突解决：链地址法、开放寻址法。Python 的 dict 使用开放寻址。",
     "hash_table"),

    # === 操作系统 ===
    ("进程（Process）是程序的执行实例，拥有独立的地址空间。线程（Thread）是进程内的执行单元，共享进程资源。协程（Coroutine）是用户态的轻量级并发单元。",
     "process_thread"),
    ("死锁（Deadlock）四个必要条件：互斥、持有并等待、不可抢占、循环等待。解决策略：鸵鸟算法、死锁预防、死锁避免（银行家算法）、死锁检测与恢复。",
     "deadlock"),
    ("虚拟内存（Virtual Memory）将磁盘空间映射为地址空间，使程序可以使用超过物理内存的地址。核心机制：分页（Paging）、分段（Segmentation）、页表。",
     "virtual_memory"),

    # === 数据库 ===
    ("ACID 特性：原子性（Atomicity）、一致性（Consistency）、隔离性（Isolation）、持久性（Durability）。是关系型数据库事务的核心保障。",
     "acid"),
    ("索引（Index）通过 B+ 树或哈希结构加速数据检索。优点：加速查询；代价：增删改变慢、占用存储。常见类型：聚簇索引、非聚簇索引、联合索引。",
     "database_index"),
    ("SQL JOIN 类型：INNER JOIN（交集）、LEFT JOIN（保留左表所有行）、RIGHT JOIN、FULL OUTER JOIN、CROSS JOIN（笛卡尔积）。",
     "sql_join"),

    # === 计算机网络 ===
    ("OSI 七层模型：物理层、数据链路层、网络层、传输层、会话层、表示层、应用层。TCP/IP 四层模型更为实用：网络接口层、网络层、传输层、应用层。",
     "osi_model"),
    ("TCP 三次握手：SYN → SYN-ACK → ACK。四次挥手：FIN → ACK → FIN → ACK。TCP 通过序列号、确认应答、超时重传保证可靠传输。",
     "tcp_handshake"),
    ("HTTP 状态码：1xx 信息、2xx 成功（200 OK、201 Created）、3xx 重定向（301 永久、302 临时）、4xx 客户端错误（400 Bad Request、401 Unauthorized、404 Not Found）、5xx 服务器错误（500 Internal Server Error）。",
     "http_status"),
    ("RESTful API 设计原则：资源用 URL 标识、使用标准 HTTP 方法（GET/POST/PUT/DELETE）、无状态、JSON 格式、版本控制（/v1/）。",
     "restful_api"),

    # === 向量检索 / RAG ===
    ("RAG（检索增强生成）将信息检索与语言模型生成结合：先将用户查询检索出相关文档片段，再将片段注入到 LLM 的上下文中进行生成，减少幻觉。",
     "rag_definition"),
    ("ChromaDB 是一个开源向量数据库，使用 HNSW（Hierarchical Navigable Small World）图索引实现近似最近邻搜索（ANN），支持余弦相似度等多种距离度量。",
     "chromadb"),
    ("嵌入（Embedding）将文本/图像等非结构化数据映射为固定维度的稠密向量，使语义相似的输入在向量空间中距离更近。",
     "embedding"),
    ("余弦相似度（Cosine Similarity）：cos(θ) = (A·B) / (||A||·||B||)，取值 [-1, 1]。用于衡量向量方向的一致性，在文本语义检索中最常用。",
     "cosine_similarity"),

    # === Prompt Engineering ===
    ("Few-shot Prompting：在 Prompt 中提供少量示例，引导模型按示例格式和风格输出。Chain-of-Thought（CoT）：在示例中加入推理步骤，提升复杂推理任务的表现。",
     "prompt_engineering"),
    ("System Prompt 用于设定模型的角色、行为边界和输出格式。User Prompt 是具体的任务指令。两者的分工是 Prompt Engineering 的基础。",
     "system_prompt"),

    # === Python 基础 ===
    ("Python 列表推导式：[expr for item in iterable if condition]。比传统 for 循环更简洁高效，底层由 C 实现。",
     "list_comprehension"),
    ("Python 装饰器（Decorator）本质上是一个接受函数作为参数并返回新函数的高阶函数，用于在不修改原函数代码的情况下增加额外功能。语法：@decorator_name。",
     "python_decorator"),
    ("生成器（Generator）使用 yield 关键字，惰性产生值，节省内存。适合处理大数据流。表达式：(x**2 for x in range(10))。",
     "python_generator"),

    # === 通用学术 ===
    ("P 值与假设检验：P 值表示在原假设成立的前提下观察到当前结果或更极端结果的概率。P < 0.05 通常认为结果统计显著，但这不是绝对的。",
     "p_value"),
    ("F1 Score 是精确率（Precision）和召回率（Recall）的调和平均：F1 = 2·P·R / (P+R)。适用于类别不平衡的场景。",
     "f1_score"),
    ("ROC 曲线以假阳性率（FPR）为横轴、真阳性率（TPR）为纵轴。AUC（Area Under Curve）值越接近 1 表示模型分类性能越好。",
     "roc_auc"),
    ("正则化（Regularization）：L1（Lasso）使部分权重为 0，起到特征选择作用；L2（Ridge）使权重趋近于 0 但不为 0，防止过拟合。",
     "regularization"),
    ("强化学习（Reinforcement Learning）中 Agent 通过与环境交互、接收奖励信号来学习最优策略。核心概念：状态、动作、奖励、策略、Q 值。",
     "reinforcement_learning"),
    ("Markdown 语法：使用 # 表示标题层级、**加粗**、*斜体*、==高亮==（部分引擎支持）、$LaTeX$ 公式、```代码块```、> 引用块。",
     "markdown_syntax"),

    # === 第二批：提升知识库覆盖密度 ===
    ("YOLO (You Only Look Once) 是单阶段目标检测算法的代表，将检测问题转化为回归问题，直接预测边界框和类别，速度极快。YOLOv8 是最新主流版本。",
     "yolo"),
    ("梯度消失（Vanishing Gradient）是指深层网络中梯度在反向传播时逐层衰减，导致浅层参数几乎不更新。解决方法：ReLU、BatchNorm、残差连接。",
     "vanishing_gradient"),
    ("先验概率（Prior Probability）是在观测数据之前对事件发生概率的估计。后验概率（Posterior Probability）是在获得新证据后对先验的更新。贝叶斯定理连接了先验和后验。",
     "prior_posterior"),
    ("条件概率 P(A|B) 表示在事件 B 已发生的条件下事件 A 发生的概率。若 A、B 独立，则 P(A|B) = P(A)。",
     "conditional_probability"),
    ("高斯分布（Gaussian Distribution）即正态分布，其概率密度函数呈钟形。均值 μ 决定位置，标准差 σ 决定宽度。68-95-99.7 规则描述了数据在 ±1σ、±2σ、±3σ 内的比例。",
     "gaussian_distribution"),
    ("标准差（Standard Deviation）是方差的算术平方根，衡量数据的离散程度。样本标准差公式 s = sqrt(Σ(x_i - x̄)² / (n-1))。",
     "standard_deviation"),
    ("BFS（广度优先搜索）按层遍历，使用队列实现，适合求无权图最短路径。DFS（深度优先搜索）沿一条路径走到底再回溯，适合检测环、拓扑排序。",
     "bfs"),
    ("哈希函数将任意大小的数据映射为固定长度的摘要。好的哈希函数应均匀分布、抗碰撞。常见算法：MD5（已不安全）、SHA-256、MurmurHash。",
     "hash_function"),
    ("SYN Flood 攻击利用 TCP 三次握手的机制，发送大量 SYN 包但不完成握手，耗尽服务器半连接队列资源。防御：SYN Cookie、连接队列调优。",
     "syn_flood"),
    ("数据库事务（Transaction）是一组原子性的数据库操作序列。事务必须满足 ACID 特性。隔离级别：读未提交、读已提交、可重复读、串行化。",
     "database_transaction"),
    ("精确率（Precision）衡量预测为正例的样本中真正例的比例。召回率（Recall）衡量实际正例中被正确预测的比例。两者通常存在权衡（Precision-Recall Tradeoff）。",
     "precision_recall"),
    ("向量数据库（Vector Database）专门存储和检索高维嵌入向量。通过 ANN（近似最近邻）算法实现高效相似搜索。代表产品：ChromaDB、Milvus、Pinecone、Weaviate。",
     "vector_database"),
    ("ANN（Approximate Nearest Neighbor）近似最近邻搜索，牺牲少量精度换取检索速度。主要算法：HNSW（分层可导航小世界图）、IVF-PQ（倒排索引+乘积量化）、LSH（局部敏感哈希）。",
     "ann"),
    ("HNSW（Hierarchical Navigable Small World）是一种图索引结构，通过多层跳表结构实现高效近似最近邻搜索，在速度和精度之间取得良好平衡。大多数现代向量数据库默认使用 HNSW。",
     "hnsw"),
    ("LLM（Large Language Model）大型语言模型，如 GPT-4、Claude 3.5、Qwen 系列。通过海量文本预训练获得语言理解和生成能力。",
     "llm"),
    ("推理（Inference）是 LLM 在实际使用中的前向计算过程。自回归生成每次预测一个 token，拼接到输入后再预测下一个。温度参数（Temperature，温度参数）控制输出的随机性。",
     "inference"),
    ("Token 是大语言模型处理文本的基本单位，一个中文汉字约等于 1.5-2 个 token。API 按输入和输出的 token 总数计费。",
     "token"),
    ("多模态模型（Multimodal Model）能同时处理文本、图像、音频等多种输入模态。代表：Qwen-VL、GPT-4o、Claude 3.5 Sonnet。用于视觉问答、OCR、视频理解等场景。",
     "multimodal_model"),
    ("OCR（Optical Character Recognition）光学字符识别，将图片中的文字转化为可编辑的文本。传统方法：Tesseract + 版面分析；现代方法：多模态大模型端到端识别。",
     "ocr"),
    ("版面分析（Layout Analysis）将文档图片分割为文本、表格、图片等区域，是传统 OCR 管道的关键步骤。深度学习版面分析模型如 LayoutLM、DiT。",
     "layout_analysis"),
    ("LangGraph 是 LangChain 团队开发的 Agent 编排框架。通过 StateGraph 定义节点（Node）和边（Edge）构建有状态的工作流，支持条件路由和循环（Loop）。",
     "langgraph"),
    ("LoRA（Low-Rank Adaptation）低秩适应是一种参数高效微调技术。通过在预训练权重旁添加低秩矩阵来适配下游任务，大幅减少可训练参数（通常 < 1%）。",
     "lora"),
    ("Qwen（通义千问）是阿里云自研的大语言模型系列。Qwen3-vl-plus 是多模态版本，支持图像理解。Text-embedding-v2 是文本嵌入模型，输出 1536 维向量。",
     "qwen"),
    ("Streamlit 是一个 Python Web 框架，用于快速构建数据科学和 ML 应用的交互式界面。通过 st.session_state 管理会话状态，不支持原生 WebSocket 长连接。",
     "streamlit"),
]

def main():
    db = VectorEngine()

    existing = db.collection.count()
    if existing >= len(KNOWLEDGE_ENTRIES):
        print(f"知识库已有 {existing} 条记录，无需重复录入。")
        return

    # 清空旧数据，重新录入
    if existing > 0:
        all_ids = db.collection.get()["ids"]
        db.collection.delete(ids=all_ids)
        print(f"已清空旧数据 {existing} 条。")

    texts = [entry[0] for entry in KNOWLEDGE_ENTRIES]
    ids = [entry[1] for entry in KNOWLEDGE_ENTRIES]

    db.add_documents(texts, ids)
    print(f"知识库初始化完成：共 {len(texts)} 条专业领域参考知识。")

    # 验证
    count = db.collection.count()
    assert count == len(KNOWLEDGE_ENTRIES), f"预期 {len(KNOWLEDGE_ENTRIES)} 条，实际 {count} 条"
    print("验证通过。")

if __name__ == "__main__":
    main()
