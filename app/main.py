import streamlit as st
import os
from PIL import Image
from dotenv import load_dotenv
from pathlib import Path

# --- 【关键点 1】必须在 import 自定义模块之前加载环境变量 ---
base_dir = Path(__file__).resolve().parent.parent # 找到 key.env 所在的根目录
load_dotenv(base_dir / "key.env")

# --- 【关键点 2】加载完环境后再导入 app ---
try:
    from core.workflow import app
except Exception as e:
    st.error(f"工作流加载失败，请检查 core/ 目录下的文件。错误: {e}")

# 1. 页面配置
st.set_page_config(page_title="Note-taking System", layout="wide")

st.title("📝 Note-taking System: 手写笔记智能转换器")
st.markdown("""
本系统基于 **Multi-Agent + RAG** 架构。上传一张你的手写笔记图片，AI 将自动识别、检索专业背景知识并整理成标准的 Markdown。
""")

# 2. 侧边栏：配置与上传
with st.sidebar:
    st.header("配置中心")
    api_key = os.getenv("DASHSCOPE_API_KEY")

    if api_key:
        st.success("✅ 阿里灵积密钥已从系统加载")
        st.caption(f"当前密钥: {api_key[:6]}****{api_key[-4:]}")
    else:
        st.error("❌ 未找到 DASHSCOPE_API_KEY")
        st.info("请检查项目根目录下的 key.env 文件")

    st.divider()
    uploaded_file = st.file_uploader("上传手写笔记图片", type=["jpg", "jpeg", "png"])

# 3. 主界面逻辑
if uploaded_file is not None:
    # 保存临时文件
    input_path = f"data/input/{uploaded_file.name}"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 展示原图
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("原始手写稿")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("电子档预览")

        if st.button("开始转换 ✨"):
            if not os.environ.get("DASHSCOPE_API_KEY"):
                st.error("请先在侧边栏输入 API Key")
            else:
                with st.spinner("Agent 正在协作（识别 -> 检索 -> 排版）..."):
                    try:
                        # 执行 LangGraph 工作流
                        inputs = {"image_path": input_path}
                        result = app.invoke(inputs)

                        # 获取最终生成的 Markdown
                        md_content = result.get("final_markdown", "生成失败")

                        # 渲染 Markdown
                        st.markdown(md_content)

                        # 提供下载按钮
                        st.download_button(
                            label="下载 Markdown 文件",
                            data=md_content,
                            file_name=f"{uploaded_file.name.split('.')[0]}.md",
                            mime="text/markdown"
                        )
                        st.success("处理完成！")
                    except Exception as e:
                        st.error(f"处理过程中出错: {e}")
else:
    st.info("请在左侧上传一张手写笔记图片开始。")

# 4. 底部补充说明（简历加分项）
st.divider()
with st.expander("查看 Agent 协作逻辑"):
    st.write("""
    1. **Vision Node**: 利用 GPT-4o 识别文字并提取颜色（如红笔标注）的元数据。
    2. **RAG Node**: 将关键词输入向量数据库 (ChromaDB)，检索相关背景知识，修正识别偏差。
    3. **Formatter Agent**: 将结构化 JSON 与检索到的 Context 结合，生成符合审美标准的 Markdown。
    """)

    # 添加一张说明架构图或流程图
    # 建议图片路径：assets/framework.png
    try:
        st.image(
            "app/licensed-image.jpeg",
            caption="系统架构：Multi-Agent + RAG 协作流程",
            use_container_width=True
        )
    except:
        # 如果图片还没准备好，先显示一个占位提示，防止页面报错
        st.info("💡 此处可放置一张系统架构图 (assets/framework.png) 展示 Agent 交互逻辑。")