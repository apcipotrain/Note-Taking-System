import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from unittest.mock import patch

# 1. 环境准备
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
load_dotenv(BASE_DIR / "key.env")

# 确保在导入 app 前环境已就绪
from core.workflow import app


def test_agent_iteration_logic():
    print("🚀 开始测试：Agent 语义迭代与多次回流逻辑")
    print("=" * 60)

    # 模拟输入状态
    initial_state = {
        "image_path": "data/input/test.png",
        "retry_count": 0,
        "error_msg": "",
        "is_valid": False
    }

    # 2. 使用 patch 模拟 Vision 节点的多次返回
    # 第一次：返回一个没有颜色的简陋数据（触发你新写的质量检查）
    # 第二次：返回一个带有颜色标注的高质量数据（通过检查）
    mock_responses = [
        [{"type": "text", "content": "这是第一次识别，我故意没写颜色信息", "style": {"color": "black"}}],
        [
            {"type": "header", "content": "第二次迭代", "style": {"color": "black"}},
            {"type": "text", "content": "我收到了反馈，现在加上了红色标注", "style": {"color": "red"}}
        ]
    ]

    with patch('core.workflow.analyze_handwriting') as mock_vision:
        mock_vision.side_effect = mock_responses

        print("[运行] 启动 LangGraph 工作流...")

        step = 0
        # 3. 执行流并观察节点
        for event in app.stream(initial_state):
            for node_name, output in event.items():
                step += 1
                print(f"\n📍 步骤 {step} | 节点: {node_name}")

                if node_name == "vision_processor":
                    rc = output.get("retry_count", 1)
                    print(f"   [Vision] 正在进行第 {rc} 次尝试")

                if node_name == "quality_validator":
                    is_valid = output.get("is_valid")
                    err = output.get("error_msg")
                    print(f"   [Validator] 质量达标: {is_valid}")
                    if not is_valid:
                        print(f"   [Feedback] 产生的改进建议: {err}")

                if node_name == "note_formatter":
                    print("   [Success] 流程最终走向了排版节点！")

    print("\n" + "=" * 60)
    print("🚩 如何判断测试通过：")
    print("1. 观察是否出现了两次 'vision_processor'。")
    print("2. 观察第一次 Validator 是否输出了你新写的语义改进建议（如“未检测到颜色”）。")
    print("3. 如果流程在第二次 Vision 后走向了 RAG 和 Formatter，说明迭代成功。")


if __name__ == "__main__":
    test_agent_iteration_logic()