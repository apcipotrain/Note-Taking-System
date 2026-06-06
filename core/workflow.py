import os
from typing import TypedDict, List, Dict, Literal
from langgraph.graph import StateGraph, END
from openai import OpenAI
from dotenv import load_dotenv

# 导入你原有的组件
from core.multimodal import analyze_handwriting
from engine.retriever import NoteRetriever
from core.agents.formatter import FormatterAgent

# 加载环境
load_dotenv("key.env")


# --- 1. 定义增强型状态 (AgentState) ---
class AgentState(TypedDict):
    image_path: str
    raw_json: List[Dict]
    context: str
    final_markdown: str
    # 新增控制字段
    retry_count: int  # 记录重试次数
    error_msg: str  # 记录报错信息，用于反馈给模型
    is_valid: bool  # 质量达标位


# 实例化组件
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
retriever = NoteRetriever()
formatter = FormatterAgent()


# --- 2. 定义节点逻辑 (Nodes) ---

def vision_node(state: AgentState):
    print("--- [Node: Vision] 开始识别 ---")
    # 增加 retry_msg 处理逻辑
    retry_msg = state.get("error_msg", "")

    # 拿到数据
    data = analyze_handwriting(state["image_path"], extra_feedback=retry_msg)

    # 关键：强制打印，确保在这里 data 不是空的
    print(f"--- [Node: Vision] 识别完成，条数: {len(data) if isinstance(data, list) else '非列表'}")

    return {
        "raw_json": data,
        "retry_count": state.get("retry_count", 0) + 1
    }

# --- core/workflow.py ---

# core/workflow.py

def validator_node(state: AgentState):
    print("--- [Node: Validator] 正在执行【全内容+最简洁】审校 ---")
    data = state.get("raw_json", [])
    retry_count = state.get("retry_count", 0)

    if not data:
        return {"is_valid": False, "error_msg": "内容为空，请重新识别。", "retry_count": retry_count}

    # 评判标准 1：覆盖度检查（颜色和公式）
    has_color = any(
        item.get("style", {}).get("color") not in ("black", None, "")
        or item.get("style", {}).get("highlight")
        for item in data
    )
    has_formula = any(item.get("type") in ["formula", "diagram"] for item in data)

    # 评判标准 2：信息紧凑度
    total_chars = sum(len(str(item.get("content", ""))) for item in data)
    avg_len = total_chars / len(data) if data else 0

    issues = []

    if not has_color and retry_count < 2:
        issues.append("未发现颜色细节，请检查是否有红笔/荧光笔标记被忽略")
    if not has_formula and retry_count < 2:
        issues.append("请检查是否遗漏了数学公式或手绘图表")
    if avg_len > 120:
        issues.append("识别内容过于冗长，请重新提炼，用最简洁的语言覆盖所有细节")

    if issues:
        feedback = " | ".join(issues)
        return {
            "is_valid": False,
            "error_msg": feedback,
            "retry_count": retry_count,
        }

    return {"is_valid": True, "retry_count": retry_count}

def rag_node(state: AgentState):
    """检索增强节点"""
    print("--- [Node: RAG] 检索专业背景知识 ---")
    context = retriever.get_context_for_note(state["raw_json"])
    return {"context": context}


def formatter_node(state: AgentState):
    """格式化节点"""
    print("--- [Node: Formatter] 构建 Markdown 文档 ---")
    combined_input = {
        "notes": state["raw_json"],
        "reference_knowledge": state["context"]
    }
    markdown = formatter.format(combined_input, client)
    return {"final_markdown": markdown}


def graceful_fallback_node(state: AgentState):
    """当重试耗尽时，直接使用最后一次识别结果继续流程，避免用户一无所获。"""
    print("--- [Node: Fallback] 重试耗尽，使用当前结果继续 ---")
    return {"is_valid": True}


# --- 3. 定义路由决策 (Conditional Edges) ---

def router(state: AgentState) -> Literal["to_rag", "to_retry", "to_fallback", "to_end"]:
    if state.get("is_valid"):
        return "to_rag"

    if state.get("retry_count", 0) < 3:
        return "to_retry"

    # 有数据就走 fallback 继续，完全没数据才 end
    if state.get("raw_json"):
        return "to_fallback"

    return "to_end"


# --- 4. 构建图 (The Graph) ---

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("vision_processor", vision_node)
workflow.add_node("quality_validator", validator_node)
workflow.add_node("rag_retriever", rag_node)
workflow.add_node("note_formatter", formatter_node)
workflow.add_node("graceful_fallback", graceful_fallback_node)

# 连线
workflow.set_entry_point("vision_processor")
workflow.add_edge("vision_processor", "quality_validator")

# 核心：条件路由
workflow.add_conditional_edges(
    "quality_validator",
    router,
    {
        "to_rag": "rag_retriever",
        "to_retry": "vision_processor",
        "to_fallback": "rag_retriever",
        "to_end": END,
    }
)

workflow.add_edge("rag_retriever", "note_formatter")
workflow.add_edge("note_formatter", END)

# 编译应用
app = workflow.compile()