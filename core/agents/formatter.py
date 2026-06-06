from typing import Dict

class FormatterAgent:
    def __init__(self):
        self.role = "Markdown Architect"

    def generate_prompt(self, combined_input: Dict):
        notes = combined_input.get("notes", [])
        reference = combined_input.get("reference_knowledge", "")

        ref_section = ""
        if reference:
            ref_section = f"""
参考知识（来自知识库检索，可用于修正术语、补充定义，但不要改变原作者的表达意图）：
{reference}
"""

        return f"""你是一个精通 Obsidian 和 Notion 语法的笔记架构师。
任务：将下方 JSON 格式的手写识别内容转化为美观、结构清晰的 Markdown 文档。
{ref_section}
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
4. 尊重原笔记：
   - 尽可能保留原始笔记的内容和方法，不要过度修改。
   - 参考知识仅用于术语修正和背景补充，不要改变原作者的表达意图。
   - 如果有参考知识可补充的背景信息，请在文末添加 "📚 知识补充" 模块。

待处理数据：
{notes}
"""

    def format(self, combined_input: Dict, llm_client):
        prompt = self.generate_prompt(combined_input)
        response = llm_client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是一个专业的 Markdown 排版助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content