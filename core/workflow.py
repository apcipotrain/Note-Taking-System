from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, END
from core.multimodal import analyze_handwriting
from engine.retriever import NoteRetriever
from core.agents.formatter import FormatterAgent
from openai import OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv

# 1. 定义状态对象 (State)
# 它是各个 Agent 之间传递的“接力棒”
class AgentState(TypedDict):
    image_path: str                 # 输入：图片路径
    raw_json: List[Dict]            # 视觉识别后的结构化数据
    context: str                    # RAG 检索回来的背景知识
    final_markdown: str             # 最终生成的 MD 内容

# 2. 实例化组件
# 统一使用阿里灵积的 client
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
retriever = NoteRetriever()
formatter = FormatterAgent()

# 3. 定义节点 (Nodes)
def vision_node(state: AgentState):
    """视觉识别节点"""
    print("--- 正在识别手写笔记内容 ---")
    raw_data = analyze_handwriting(state["image_path"])
    return {"raw_json": raw_data}

def rag_node(state: AgentState):
    """检索增强节点"""
    print("--- 正在检索相关背景知识 ---")
    # 从识别到的 JSON 中提取关键词去检索
    context = retriever.get_context_for_note(state["raw_json"])
    return {"context": context}

def formatter_node(state: AgentState):
    """格式化整理节点"""
    print("--- 正在生成最终 Markdown ---")
    # 将视觉数据 + RAG 背景知识交给排版 Agent
    combined_input = {
        "notes": state["raw_json"],
        "reference_knowledge": state["context"]
    }
    markdown = formatter.format(combined_input, client)
    return {"final_markdown": markdown}

# 4. 构建图 (Graph)
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("vision_processor", vision_node)
workflow.add_node("rag_retriever", rag_node)
workflow.add_node("note_formatter", formatter_node)

# 建立连线 (Edge)
workflow.set_entry_point("vision_processor")
workflow.add_edge("vision_processor", "rag_retriever")
workflow.add_edge("rag_retriever", "note_formatter")
workflow.add_edge("note_formatter", END)

# 编译成可执行应用
app = workflow.compile()

# --- 测试运行 ---
if __name__ == "__main__":
    initial_state = {"image_path": "data/input/my_physics_note.jpg"}
    result = app.invoke(initial_state)
    print("\n生成结果：\n", result["final_markdown"])