import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 1. 环境准备：确保能导入项目根目录下的 core 模块
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
load_dotenv(BASE_DIR / "key.env")

# 2. 导入你重构后的 workflow
try:
    from core.workflow import app

    print("✅ 成功加载 Workflow 图")
except Exception as e:
    print(f"❌ 加载 Workflow 失败: {e}")
    sys.exit(1)


def run_step_by_step_test(image_name):
    # 构造测试输入
    input_path = str(BASE_DIR / "data" / "input" / image_name)
    if not os.path.exists(input_path):
        print(f"❌ 找不到测试图片: {input_path}")
        return

    # 初始化 LangGraph 状态
    initial_state = {
        "image_path": input_path,
        "retry_count": 0,
        "error_msg": "",
        "is_valid": False
    }

    print(f"\n🚀 开始执行工作流测试 (图片: {image_name})")
    print("=" * 50)

    # 3. 使用 stream 模式逐个节点观察
    current_state = initial_state
    try:
        # event 包含了每个节点执行后的输出
        for event in app.stream(initial_state):
            for node_name, output in event.items():
                print(f"\n📍 [节点完成]: {node_name}")

                # 打印该节点返回的内容
                for key, value in output.items():
                    if key == "raw_json":
                        print(f"   - {key}: 收到数据条数 = {len(value) if isinstance(value, list) else '非列表'}")
                        if value:
                            print(f"     预览首条内容: {str(value[0])[:100]}...")
                    else:
                        print(f"   - {key}: {value}")

                # 更新模拟的全局状态
                current_state.update(output)

        print("\n" + "=" * 50)
        if "final_markdown" in current_state:
            print("🎉 测试成功！最终生成了 Markdown 内容。")
        else:
            print("⚠️ 测试结束，但未生成最终 Markdown。请检查 Router 是否走向了 END。")

    except Exception as e:
        print(f"\n💥 运行中崩溃: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 使用你刚才测试成功的图片名
    run_step_by_step_test("test.png")