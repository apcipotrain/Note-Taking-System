from core.multimodal import analyze_handwriting
import json

try:
    print("正在启动 Qwen3-VL-Plus 视觉验证...")
    # 请确保路径下有一张真实图片
    result = analyze_handwriting("data/input/test.png")
    print("✅ API 通讯成功！识别结果如下：")
    print(json.dumps(result, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"❌ 验证失败。错误原因: {e}")
    print("检查建议：1. key.env 路径是否正确 2. 密钥是否有效 3. 网络是否能访问阿里服务器")