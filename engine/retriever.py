from engine.vector_db import VectorEngine

class NoteRetriever:
    def __init__(self):
        self.db = VectorEngine()

    def get_context_for_note(self, note_json: list):
        # 1. 提取关键词（增加健壮性检查）
        query_fragments = [str(item.get('content', '')) for item in note_json if 'content' in item]
        full_query = " ".join(query_fragments[:5])

        if not full_query.strip():
            return ""

        # 2. 执行检索
        context_list = self.db.query(full_query)

        # 3. 【核心修复】确保 context_list 是纯字符串列表
        # 如果 self.db.query 返回的是 [['doc1', 'doc2']]，需要展平它
        if context_list and isinstance(context_list[0], list):
            context_list = context_list[0]

        # 4. 再次检查，确保 join 的全是字符串
        clean_context = [str(i) for i in context_list if i]

        return "\n".join(clean_context)