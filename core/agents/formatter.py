from typing import List, Dict

class FormatterAgent:
    def __init__(self):
        self.role = "Markdown Architect"

    def generate_prompt(self, raw_json: List[Dict]):
        return f"""
        你是一个精通 Obsidian 和 Notion 语法的笔记架构师。
        任务：将下方 JSON 格式的手写识别内容转化为美观、结构清晰的 Markdown 文档。
        转换规则：
        1. 颜色处理：
           - 红色 (red) 内容：使用 ==高亮== 或 **加粗**。如果是核心定义，请使用 Blockquote (> )。
           - 蓝色 (blue) 内容：通常是补充说明，使用斜体 *内容*。
        2. 结构处理：
           - 识别 level 为 1 的作为 # 一级标题，以此类推。
           - 将 type 为 'list_item' 的内容整合为无序列表 (- )。
        3. 智能优化：
           - 修正 OCR 可能存在的标点错误。
           - 如果发现有 type 为 'formula' 的内容，请确保使用 $...$ 或 $$...$$ 的 LaTeX 格式。
        4. 尊重笔记：
           - 尽可能保留原始笔记的内容以及方法。
           - 其他地方不去过多添加补充修改，重在调整。
        待处理数据：
        {raw_json}
        """

    def format(self, raw_json: List[Dict], llm_client):
        # 调用 LLM 进行排版转换
        prompt = self.generate_prompt(raw_json)
        # --- 关键修改点：将模型改为 qwen3-vl-plus ---
        response = llm_client.chat.completions.create(
            model="qwen3-vl-plus",  # 阿里灵积的高性价比文本模型
            messages=[
                {"role": "system", "content": "你是一个专业的 Markdown 排版助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # 降低随机性，保证排版稳定
        )
        return response.choices[0].message.content