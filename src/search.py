"""Semantic search with FAISS and Cross-Encoder re-ranking (Lab 5)."""
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class ComplaintSearch:
    def __init__(self, texts):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.texts = list(texts)
        self.model = SentenceTransformer(EMBED_MODEL)
        embeddings = self.model.encode(self.texts)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self._reranker = None

    def search(self, query, k=3):
        """Return the k closest complaints as (text, L2 distance); smaller is closer."""
        k = min(k, len(self.texts))
        distances, indices = self.index.search(self.model.encode([query]), k)
        return [(self.texts[i], float(d)) for d, i in zip(distances[0], indices[0])]

    def rerank(self, query, k=5):
        """Retrieve k candidates, then re-order them with a Cross-Encoder."""
        from sentence_transformers import CrossEncoder

        if self._reranker is None:
            self._reranker = CrossEncoder(RERANK_MODEL)
        candidates = [t for t, _ in self.search(query, k)]
        scores = self._reranker.predict([[query, t] for t in candidates])
        return sorted(zip(candidates, map(float, scores)), key=lambda x: x[1], reverse=True)


def recall_mrr(searcher, eval_set, k=5):
    """Recall@k and MRR@k over (query, gold_text) pairs."""
    ranks = []
    for query, gold in eval_set:
        found = [t for t, _ in searcher.search(query, k)]
        ranks.append(found.index(gold) + 1 if gold in found else None)
    n = len(eval_set)
    recall = sum(r is not None for r in ranks) / n
    mrr = sum(1 / r for r in ranks if r) / n
    return recall, mrr, ranks
