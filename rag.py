# backend/rag.py
"""
Robust RAG helper.
- Tries to use sentence-transformers if available (better semantic retrieval).
- Falls back to sklearn.TfidfVectorizer + cosine similarity if sentence-transformers is not available
  or fails to import (avoids crashing the whole server on startup).
- Provides KnowledgeIndex.retrieve(query, k) and rag_query(query, index, k).
- The llm_answer() is still a demo stub — replace with your fine-tuned LLM call when ready.
"""

import os
import numpy as np
from typing import List

# Try to import SentenceTransformer lazily. If unavailable, we'll use TF-IDF fallback.
_HAS_ST = False
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import NearestNeighbors
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# TF-IDF fallback imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"


class KnowledgeIndex:
    def __init__(self, docs_dir="knowledge"):
        self.docs_dir = docs_dir
        self.docs: List[str] = []
        self.embeddings = None
        self.model = None
        self.tfidf_vect = None
        self.tfidf_matrix = None

        self._load_docs()
        if len(self.docs) == 0:
            # nothing to do
            return

        if _HAS_ST:
            try:
                # create ST model lazily
                self.model = SentenceTransformer(MODEL_NAME)
                self.embeddings = self.model.encode(self.docs, convert_to_numpy=True)
            except Exception:
                # fall back to TF-IDF if SentenceTransformer fails at runtime
                self.model = None
                self._build_tfidf()
        else:
            # fallback to tf-idf
            self._build_tfidf()

    def _load_docs(self):
        docs = []
        if not os.path.exists(self.docs_dir):
            return
        for fname in os.listdir(self.docs_dir):
            if fname.endswith(".txt"):
                path = os.path.join(self.docs_dir, fname)
                try:
                    with open(path, "r", encoding="utf8") as f:
                        docs.append(f.read())
                except Exception:
                    # skip unreadable files
                    continue
        self.docs = docs

    def _build_tfidf(self):
        # Build TF-IDF matrix for the corpus (fallback)
        try:
            self.tfidf_vect = TfidfVectorizer(stop_words="english")
            self.tfidf_matrix = self.tfidf_vect.fit_transform(self.docs)
        except Exception:
            self.tfidf_vect = None
            self.tfidf_matrix = None

    def retrieve(self, query: str, k=3) -> List[str]:
        """
        Return up to k most relevant document texts for the query.
        Uses SentenceTransformer if available, otherwise TF-IDF cosine similarity.
        """
        if len(self.docs) == 0:
            return []

        # Use semantic vectors if available
        if self.embeddings is not None and self.model is not None:
            try:
                qvec = self.model.encode([query], convert_to_numpy=True)
                # cosine similarity via dot product
                sims = (self.embeddings @ qvec[0]) / (
                    (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(qvec[0]) + 1e-12)
                )
                top_idx = np.argsort(-sims)[:k]
                return [self.docs[i] for i in top_idx]
            except Exception:
                # fallback to TF-IDF if anything fails
                pass

        # TF-IDF fallback
        if self.tfidf_matrix is not None and self.tfidf_vect is not None:
            try:
                qtf = self.tfidf_vect.transform([query])
                sims = cosine_similarity(self.tfidf_matrix, qtf).reshape(-1)
                top_idx = np.argsort(-sims)[:k]
                return [self.docs[i] for i in top_idx]
            except Exception:
                return []

        return []


# Demo LLM function stub - replace with real fine-tuned LLM call
def llm_answer(prompt: str, api_client=None) -> str:
    """
    Replace this stub with a call to your fine-tuned LLM (OpenAI, HF inference, etc.)
    Example: openai.ChatCompletion.create(...) or local model inference.
    """
    # Provide a conservative, deterministic demo response when no LLM connected.
    return (
        "LLM answer (demo): Based on the retrieved documents and the schedule, "
        "it is recommended to raise the setpoint during high-tariff hours to reduce cost while keeping comfort near target."
    )


def rag_query(query: str, index: KnowledgeIndex, k=3) -> str:
    contexts = index.retrieve(query, k=k)
    if contexts:
        joined = "\n\n---\n\n".join(contexts)
    else:
        joined = "(no supporting documents found in the knowledge base)"
    prompt = f"User question:\n{query}\n\nContext documents:\n{joined}\n\nPlease answer concisely and reference documents when appropriate."
    # If you have a fine-tuned LLM, call it here and return its response.
    return llm_answer(prompt)

