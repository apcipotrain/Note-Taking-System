from core.agents.formatter import FormatterAgent
from core.multimodal import client  # 复用阿里的 client


def test_format():
    agent = FormatterAgent()
    # 模拟一段带红色的识别结果
    mock_json = [
        {"type": "header", "level": 1, "content": "神经网络笔记"},
        {"type": "text", "content": "这是红笔写的重点：激活函数很重要", "style": {"color": "red"}}
    ]

    print("正在验证 Agent 排版能力...")
    md = agent.format(mock_json, client)
    print("✅ Markdown 生成结果：")
    print("-" * 20)
    print(md)


if __name__ == "__main__":
    test_format()