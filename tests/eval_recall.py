"""RAG 召回率评估框架

评估指标:
- Recall@K: 预期术语至少出现在前 K 个检索结果中的比例
- Precision@K: 前 K 个结果中相关结果的比例
- MRR (Mean Reciprocal Rank): 第一个相关结果排名的倒数均值

用法: python tests/eval_recall.py
"""
import sys
import io
from pathlib import Path

# 强制 UTF-8 输出，解决 Windows GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.retriever import NoteRetriever

# 测试集: (模拟识别 JSON, 预期应召回的关键词列表)
# 模拟真实场景的手写笔记片段
TEST_CASES = [
    # === ML / AI 领域 ===
    {
        "name": "CNN 基础概念",
        "input": [
            {"type": "header", "level": 1, "content": "卷积神经网络"},
            {"type": "text", "content": "卷积层提取特征，池化层降维"},
            {"type": "text", "content": "最后接全连接层做分类"},
        ],
        "expected": ["卷积神经网络", "CNN", "特征提取", "池化"],
    },
    {
        "name": "Transformer 与注意力",
        "input": [
            {"type": "header", "level": 1, "content": "Transformer 架构"},
            {"type": "text", "content": "Self-Attention 机制是核心"},
            {"type": "text", "content": "公式: softmax(QK^T/sqrt(d_k))V"},
        ],
        "expected": ["Transformer", "自注意力", "Attention", "QKV"],
    },
    {
        "name": "反向传播与梯度",
        "input": [
            {"type": "text", "content": "反向传播算法，链式法则求梯度"},
            {"type": "text", "content": "梯度下降优化损失函数"},
        ],
        "expected": ["反向传播", "梯度", "链式法则", "损失函数"],
    },
    {
        "name": "BERT 预训练",
        "input": [
            {"type": "header", "level": 2, "content": "BERT 模型"},
            {"type": "text", "content": "Masked Language Model 预训练"},
            {"type": "text", "content": "Next Sentence Prediction"},
        ],
        "expected": ["BERT", "MLM", "预训练", "NSP"],
    },
    {
        "name": "GPT 自回归生成",
        "input": [
            {"type": "text", "content": "GPT 自回归语言模型"},
            {"type": "text", "content": "预测下一个 token，生成文本"},
        ],
        "expected": ["GPT", "自回归", "生成式", "token"],
    },
    {
        "name": "过拟合与正则化",
        "input": [
            {"type": "text", "content": "训练集准确率高但测试集差"},
            {"type": "text", "content": "过拟合问题，用 L1 L2 正则化"},
        ],
        "expected": ["过拟合", "正则化", "L1", "L2", "泛化"],
    },
    {
        "name": "图像目标检测",
        "input": [
            {"type": "text", "content": "YOLO 目标检测算法"},
            {"type": "text", "content": "同时定位和分类物体"},
        ],
        "expected": ["目标检测", "YOLO", "检测"],
    },
    {
        "name": "激活函数 ReLU",
        "input": [
            {"type": "text", "content": "ReLU 激活函数 f(x)=max(0,x)"},
            {"type": "text", "content": "解决梯度消失问题"},
        ],
        "expected": ["ReLU", "激活函数", "梯度消失"],
    },

    # === 数学领域 ===
    {
        "name": "贝叶斯定理",
        "input": [
            {"type": "text", "content": "贝叶斯公式 P(A|B)=P(B|A)P(A)/P(B)"},
            {"type": "text", "content": "先验概率更新为后验概率"},
        ],
        "expected": ["贝叶斯", "先验", "后验", "条件概率"],
    },
    {
        "name": "正态分布",
        "input": [
            {"type": "text", "content": "高斯分布也叫正态分布"},
            {"type": "text", "content": "钟形曲线，μ 均值 σ 标准差"},
        ],
        "expected": ["正态分布", "高斯", "均值", "标准差"],
    },
    {
        "name": "PCA 降维",
        "input": [
            {"type": "text", "content": "主成分分析 PCA 降维"},
            {"type": "text", "content": "特征值分解找最大方差方向"},
        ],
        "expected": ["PCA", "主成分分析", "降维", "特征值"],
    },
    {
        "name": "信息熵",
        "input": [
            {"type": "text", "content": "信息熵衡量不确定性"},
            {"type": "text", "content": "交叉熵用于分类损失"},
        ],
        "expected": ["信息熵", "交叉熵", "不确定性"],
    },

    # === CS 基础 ===
    {
        "name": "动态规划",
        "input": [
            {"type": "text", "content": "动态规划最优子结构"},
            {"type": "text", "content": "重叠子问题，记忆化搜索"},
        ],
        "expected": ["动态规划", "最优子结构", "记忆化"],
    },
    {
        "name": "BFS 与 DFS",
        "input": [
            {"type": "text", "content": "图的遍历：BFS 用队列"},
            {"type": "text", "content": "DFS 用栈或递归"},
        ],
        "expected": ["BFS", "DFS", "广度优先", "深度优先"],
    },
    {
        "name": "哈希表",
        "input": [
            {"type": "text", "content": "哈希表 O(1) 查找"},
            {"type": "text", "content": "哈希函数映射，解决冲突"},
        ],
        "expected": ["哈希表", "哈希函数", "O(1)"],
    },
    {
        "name": "TCP 三次握手",
        "input": [
            {"type": "text", "content": "TCP 建立连接 SYN, SYN-ACK, ACK"},
            {"type": "text", "content": "可靠传输，序列号确认"},
        ],
        "expected": ["TCP", "三次握手", "SYN", "ACK"],
    },
    {
        "name": "死锁条件",
        "input": [
            {"type": "text", "content": "死锁：互斥、持有等待、不可抢占、循环等待"},
        ],
        "expected": ["死锁", "互斥", "循环等待"],
    },
    {
        "name": "ACID 事务",
        "input": [
            {"type": "text", "content": "数据库事务 ACID 特性"},
            {"type": "text", "content": "原子性、一致性、隔离性、持久性"},
        ],
        "expected": ["ACID", "事务", "原子性", "持久性"],
    },
    {
        "name": "F1 Score",
        "input": [
            {"type": "text", "content": "F1 是精确率和召回率的调和平均"},
            {"type": "text", "content": "不均衡分类问题的评估"},
        ],
        "expected": ["F1", "精确率", "召回率", "调和平均"],
    },

    # === RAG / 向量检索 ===
    {
        "name": "RAG 检索增强",
        "input": [
            {"type": "text", "content": "检索增强生成 RAG"},
            {"type": "text", "content": "先检索再生成，减少幻觉"},
        ],
        "expected": ["RAG", "检索增强", "生成", "幻觉"],
    },
    {
        "name": "余弦相似度",
        "input": [
            {"type": "text", "content": "向量相似度用余弦"},
            {"type": "text", "content": "cos(θ)=A·B/||A||·||B||"},
        ],
        "expected": ["余弦", "相似度", "向量", "cos"],
    },
    {
        "name": "ChromaDB 向量库",
        "input": [
            {"type": "text", "content": "ChromaDB 向量数据库"},
            {"type": "text", "content": "HNSW 索引，ANN 检索"},
        ],
        "expected": ["ChromaDB", "HNSW", "ANN", "向量数据库"],
    },
]


def tokenize(text: str) -> set:
    """简单分词，便于召回匹配"""
    text = text.lower()
    # 保留中文字符、英文单词、数字
    tokens = set()
    # 英文单词
    for m in __import__("re").finditer(r"[a-z0-9]{2,}", text):
        tokens.add(m.group())
    # 中文 2-grams
    import re
    cn_chars = re.findall(r"[一-鿿]+", text)
    for cn_word in cn_chars:
        tokens.add(cn_word)
        for i in range(len(cn_word) - 1):
            tokens.add(cn_word[i : i + 2])
    return tokens


def evaluate():
    retriever = NoteRetriever()
    k = 5

    total = len(TEST_CASES)
    recall_sum = 0.0
    precision_sum = 0.0
    mrr_sum = 0.0
    per_case = []

    for case in TEST_CASES:
        results, queries = retriever.retrieve_with_scores(case["input"])
        retrieved_text = " ".join(results).lower()

        # 计算每个预期词的排名
        expected_lower = [e.lower() for e in case["expected"]]
        hits = 0
        first_rank = float("inf")  # 第一个命中的排名

        for expected in expected_lower:
            found = False
            for rank_idx, doc in enumerate(results[:k]):
                if expected in doc.lower():
                    hits += 1
                    found = True
                    first_rank = min(first_rank, rank_idx + 1)
                    break

        # Recall@K
        recall = hits / len(expected_lower) if expected_lower else 0.0
        recall_sum += recall

        # Precision@K
        # 判断前 K 个结果中每个结果是否与预期相关
        relevant_count = 0
        for doc in results[:k]:
            doc_lower = doc.lower()
            if any(e in doc_lower for e in expected_lower):
                relevant_count += 1
        precision = relevant_count / min(len(results[:k]), k) if results[:k] else 0.0
        precision_sum += precision

        # MRR
        rr = 1.0 / first_rank if first_rank != float("inf") else 0.0
        mrr_sum += rr

        per_case.append(
            {
                "name": case["name"],
                "recall": recall,
                "precision": precision,
                "rr": rr,
                "hits": hits,
                "total_expected": len(expected_lower),
            }
        )

    avg_recall = recall_sum / total
    avg_precision = precision_sum / total
    avg_mrr = mrr_sum / total

    print("=" * 70)
    print("RAG 召回率评估报告")
    print("=" * 70)
    print(f"测试用例数: {total}")
    print(f"Top-K: {k}")
    print()

    for c in per_case:
        bar = "█" * int(c["recall"] * 20) + "░" * (20 - int(c["recall"] * 20))
        status = "✅" if c["recall"] >= 0.5 else "⚠️" if c["recall"] > 0 else "❌"
        print(f"  {status} {c['name']:<20s}  Recall@{k}: {c['recall']:.2f}  "
              f"Precision@{k}: {c['precision']:.2f}  MRR: {c['rr']:.2f}  "
              f"({c['hits']}/{c['total_expected']})")

    print()
    print("-" * 70)
    print(f"  📊 平均 Recall@{k}:    {avg_recall:.2%}")
    print(f"  📊 平均 Precision@{k}:  {avg_precision:.2%}")
    print(f"  📊 平均 MRR:           {avg_mrr:.2%}")
    print("-" * 70)

    # 判定是否达标
    if avg_recall >= 0.6 and avg_precision >= 0.4:
        print("✅ 召回率和精确率均达标！")
    else:
        print("⚠️ 指标未达标，建议排查：")
        if avg_recall < 0.6:
            print("   - 召回率偏低 → 知识库覆盖不够或查询策略不佳")
        if avg_precision < 0.4:
            print("   - 精确率偏低 → 检索结果中噪声过多，需增加重排序")

    return avg_recall, avg_precision, avg_mrr


if __name__ == "__main__":
    evaluate()
