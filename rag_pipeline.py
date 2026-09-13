"""
rag_pipeline.py
Retrieval-Augmented Generation grounding layer for MedTriage AI.

Loads guidelines.txt, embeds each snippet with a free sentence-transformer,
stores in an in-memory ChromaDB collection, and retrieves the most relevant
snippets for a given query (ECG finding + symptoms) so the Groq LLM has
real evidence to reason over instead of hallucinating.
"""

import chromadb
from sentence_transformers import SentenceTransformer

GUIDELINES_FILE = "guidelines.txt"
_embedder = None
_collection = None


def _load_snippets(path=GUIDELINES_FILE):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    # split on the "### SOURCE:" markers, keep the header with each chunk
    chunks = [c.strip() for c in raw.split("###") if c.strip()]
    return ["### " + c for c in chunks]


def init_rag():
    """Call once at app startup. Builds the in-memory vector store."""
    global _embedder, _collection

    if _collection is not None:
        return  # already initialized

    _embedder = SentenceTransformer("all-MiniLM-L6-v2")  # free, small, fast

    client = chromadb.Client()  # in-memory, no server needed
    _collection = client.get_or_create_collection(name="clinical_guidelines")

    snippets = _load_snippets()
    embeddings = _embedder.encode(snippets).tolist()
    ids = [f"doc_{i}" for i in range(len(snippets))]

    _collection.add(
        documents=snippets,
        embeddings=embeddings,
        ids=ids,
    )


def retrieve(query: str, k: int = 3):
    """Return top-k most relevant guideline snippets for the query."""
    if _collection is None:
        init_rag()

    query_emb = _embedder.encode([query]).tolist()
    results = _collection.query(query_embeddings=query_emb, n_results=k)
    docs = results.get("documents", [[]])[0]
    return docs
