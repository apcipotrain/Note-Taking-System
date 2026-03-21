from engine.vector_db import VectorEngine


def init_test_data():
    db = VectorEngine()
    # 模拟一些你笔记里可能出现的专业知识
    knowledge_texts = [
        "卷积神经网络(CNN)是一种深度学习模型，常用于图像识别，核心层包括卷积层、池化层和全连接层。",
        "ReLU激活函数的公式是 f(x) = max(0, x)，它能解决梯度消失问题。",
        "Markdown 语法中，可以使用 ==文本== 来表示高亮，使用 **文本** 表示加粗。"
    ]
    ids = ["id1", "id2", "id3"]

    print("正在往向量库注入初始知识...")
    db.add_documents(knowledge_texts, ids)

    # 立即测试检索
    test_query = "什么是CNN？"
    res = db.query(test_query)
    print(f"✅ RAG 检索验证成功！查询 '{test_query}' 的匹配结果：\n{res}")


if __name__ == "__main__":
    init_test_data()