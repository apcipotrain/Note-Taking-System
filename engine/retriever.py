import re
from engine.vector_db import VectorEngine


class NoteRetriever:
    def __init__(self):
        self.db = VectorEngine()

    def _extract_key_terms(self, note_json: list) -> list:
        """从识别结果中提取关键查询词"""
        terms = set()
        for item in note_json:
            content = str(item.get("content", ""))
            item_type = item.get("type", "")

            # 标题是天然的关键词
            if item_type == "header":
                terms.add(content)

            # 提取大写缩写词 (CNN, RNN, LSTM, etc.)
            acronyms = re.findall(r"\b[A-Z]{2,6}\b", content)
            terms.update(acronyms)

            # 提取中文专有名词 (XX网络, XX定理, XX算法)
            cn_terms = re.findall(r"[一-龥]{2,8}(?:网络|定理|算法|模型|函数|公式|结构|机制|方法|系统)", content)
            terms.update(cn_terms)

            # 提取英文专业术语
            en_terms = re.findall(r"\b[a-zA-Z]+(?:\s+[a-zA-Z]+){0,3}\b", content)
            for t in en_terms:
                if len(t) > 6 and t.lower() not in {"this", "that", "these", "those", "there", "their", "which", "where", "would", "could", "should", "about", "after", "before"}:
                    terms.add(t.lower())

        return list(terms)[:10]

    def _build_query_variants(self, note_json: list) -> list:
        """构建多个查询变体以覆盖不同语义角度"""
        variants = []

        # 变体 1: 拼接所有标题
        headers = [
            str(item.get("content", ""))
            for item in note_json
            if item.get("type") == "header"
        ]
        if headers:
            variants.append(" ".join(headers))

        # 变体 2: 拼接前 5 条正文内容
        body = [
            str(item.get("content", ""))
            for item in note_json
            if item.get("type") not in ("header",)
        ][:5]
        if body:
            variants.append(" ".join(body))

        # 变体 3: 只取关键术语
        key_terms = self._extract_key_terms(note_json)
        if key_terms:
            variants.append(" ".join(key_terms))

        # 变体 4: 全量拼接（语义全景）
        full = [str(item.get("content", "")) for item in note_json]
        variants.append(" ".join(full[:10]))

        # 过滤空变体和过短变体
        return [v for v in variants if len(v.strip()) > 3]

    def _rerank_by_keyword_overlap(self, docs: list, queries: list) -> list:
        """简单关键词重叠重排序：提升包含原始查询词的文档排名"""
        if not docs:
            return docs

        all_query_text = " ".join(queries).lower()

        scored = []
        for doc in docs:
            doc_lower = doc.lower()
            # 计算原始查询词在文档中的命中数
            overlap = sum(1 for word in all_query_text.split() if word in doc_lower)
            scored.append((doc, overlap))

        # 稳定排序：保持原向量排名的同时提升关键词命中多的文档
        # 使用原始位置作为 tie-breaker
        for i, (doc, score) in enumerate(scored):
            scored[i] = (doc, score, i)

        scored.sort(key=lambda x: (-x[1], x[2]))
        return [doc for doc, _, _ in scored]

    def get_context_for_note(self, note_json: list) -> str:
        if not note_json:
            return ""

        queries = self._build_query_variants(note_json)
        if not queries:
            return ""

        # 多查询检索 + 自动去重排序 + 关键词重排序
        results = self.db.batch_query(queries, n_results=5, distance_threshold=1.5)
        results = self._rerank_by_keyword_overlap(results, queries)

        # 返回前 5 条最相关的结果
        return "\n\n".join(results[:5])

    def retrieve_with_scores(self, note_json: list):
        """离线评估用：返回 (文档列表, 查询变体列表)"""
        queries = self._build_query_variants(note_json)
        results = self.db.batch_query(queries, n_results=5, distance_threshold=1.5)
        results = self._rerank_by_keyword_overlap(results, queries)
        return results[:5], queries
