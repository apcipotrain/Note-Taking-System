import base64
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件中的变量
load_dotenv("key.env") # 如果你的文件名叫 key.env，请加上参数

# 初始化客户端，指向阿里灵积的地址
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def encode_image(image_path):
    """将图片转换为 base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_handwriting(image_path):
    # 调用多模态模型识别手写笔记，并提取颜色和样式元数据
    base64_image = encode_image(image_path)

    prompt = """
    你是一个专业的手写笔记分析专家。请识别图片中的内容并输出为 JSON 格式。
    要求：
    1. 识别所有文字内容，保持原始层级（标题、段落、列表）。
    2. 特别标注颜色信息：如果使用了红笔、马克笔高亮，请在 style 字段中注明。
    3. 如果有手写绘图或公式，请在 type 字段标注为 'diagram' 或 'formula' 并描述其内容。
    输出格式示例：
    [
      {"type": "header", "level": 1, "content": "卷积神经网络", "style": {"color": "black"}},
      {"type": "text", "content": "核心概念：特征提取", "style": {"color": "red", "highlight": true}},
      {"type": "list_item", "content": "卷积层", "style": {"color": "blue"}}
    ]
    """

    response = client.chat.completions.create(
        model="qwen3-vl-plus",  # 或者使用 qwen3-vl-max 等
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                ],
            }
        ],
        response_format={"type": "json_object"}  # 强制返回 JSON
    )

    content = response.choices[0].message.content
    # 清理掉可能存在的 Markdown 标签 ```json
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    return json.loads(content)

# 测试代码
if __name__ == "__main__":
    result = analyze_handwriting("data/input/test_note.jpg")
    print(json.dumps(result, indent=2, ensure_ascii=False))