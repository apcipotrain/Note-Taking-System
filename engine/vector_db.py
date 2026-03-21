import os
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 1. 强制使用绝对路径加载环境变量
# 无论你在哪个文件夹下运行 python，它都会去项目的根目录找 key.env
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / "key.env"
load_dotenv(dotenv_path=ENV_PATH)

class VectorEngine:
    def __init__(self, db_path=None):
        # 默认数据库路径设在根目录的 data 文件夹下
        if db_path is None:
            db_path = str(BASE_DIR / "data" / "vector_db")

        # 确保数据文件夹存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.client = chromadb.PersistentClient(path=db_path)

        # 2. 获取 API KEY 并检查
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(f"无法从 {ENV_PATH} 读取 DASHSCOPE_API_KEY，请检查文件内容")

        self.ali_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # 3. 创建集合
        self.collection = self.client.get_or_create_collection(name="knowledge_base")

    def get_embedding(self, text):
        """将文本转化为 1536 维向量"""
        # 阿里 text-embedding-v2 默认维度是 1536
        response = self.ali_client.embeddings.create(
            model="text-embedding-v2",
            input=text
        )
        return response.data[0].embedding

    def add_documents(self, texts: list, ids: list):
        """批量存入参考知识"""
        embeddings = [self.get_embedding(t) for t in texts]
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids
        )
        print(f"成功存入 {len(texts)} 条知识到向量库")

    def query(self, query_text: str, n_results: int = 2):
        """根据输入检索最相关的背景知识"""
        query_embedding = self.get_embedding(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        if results and results['documents']:
            return results['documents'][0]
        return []