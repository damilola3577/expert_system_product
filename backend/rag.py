from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import os
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

class KnowledgeIndex:
    def __init__(self, docs_dir="knowledge"):
        self.model = SentenceTransformer(MODEL_NAME)
        self.docs = []
        self.embeddings = None
        self._load_docs(docs_dir)

    def _load_docs(self, docs_dir):
        docs = []
        if not os.path.exists(docs_dir):
            return
        for fname in os.listdir(docs_dir):
            if fname.endswith(".txt"):
                with open(os.path.join(docs_dir, fname), "r", encoding="utf8") as f:
                    docs.append(f.read())
        self.docs = docs
        if docs:
            self.embeddings = self.model.encode(docs, convert_to_numpy=True)

    def retrieve(self, query, k=3):
        if self.embeddings is None or len(self.docs) == 0:
            return []
        qvec = self.model.encode([query], convert_to_numpy=True)
        nbrs = NearestNeighbors(n_neighbors=min(k, len(self.docs)), metric="cosine").fit(self.embeddings)
        dists, idxs = nbrs.kneighbors(qvec)
        results = []
        for idx in idxs[0]:
            results.append(self.docs[idx])
        return results

def llm_answer(prompt, api_client=None):
    return "LLM answer (demo): Based on retrieved documents and the schedule, it's recommended to raise the setpoint during certain hours to reduce costs while keeping comfort near target."

def rag_query(query, index: KnowledgeIndex, k=3):
    contexts = index.retrieve(query, k=k)
    joined = "\n\n---\n\n".join(contexts)
    prompt = f"User question:\n{query}\n\nContext documents:\n{joined}\n\nPlease answer concisely, reference any specific doc passages if relevant."
    return llm_answer(prompt)
