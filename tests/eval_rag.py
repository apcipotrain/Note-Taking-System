from engine.retriever import NoteRetriever
import pandas as pd


def run_recall_test():
    retriever = NoteRetriever()

    # 测试集：输入手写片段 -> 预期召回的关键词
    test_cases = [
        {"input": [{"content": "卷积层和池化层"}], "expected": "神经网络"},
        {"input": [{"content": "Redox reaction"}], "expected": "oxidation"},
    ]

    results = []
    for case in test_cases:
        context = retriever.get_context_for_note(case["input"])
        # 简单计算关键词是否在召回的 context 中
        hit = 1 if case["expected"].lower() in context.lower() else 0
        results.append(hit)

    recall = sum(results) / len(results)
    print(f"RAG 召回率测试完成: {recall:.2%}")


if __name__ == "__main__":
    run_recall_test()