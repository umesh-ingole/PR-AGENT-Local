"""RAG Retrieval stage (green box) + Vector DB (orange cylinder).

Chroma by default. To use FAISS or pgvector, replace _collection() and query();
the rest of the app only calls retrieve_context().
"""
import glob
import os

KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chroma")

_collection_cache = None


def _chunks(text: str) -> list[str]:
    """Split a markdown guide into one chunk per bullet, keeping the heading."""
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    title = lines[0] if lines and lines[0].startswith("#") else ""
    return [f"{title}\n{l}" for l in lines if l.startswith("- ")]


def _sync_collection(col) -> None:
    docs, ids, metas = [], [], []
    for path in sorted(glob.glob(os.path.join(KNOWLEDGE_DIR, "*.md"))):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as guide_file:
            for i, chunk in enumerate(_chunks(guide_file.read())):
                docs.append(chunk)
                ids.append(f"{name}-{i}")
                metas.append({"source": name})

    if not docs:
        return

    existing = col.get(include=["documents", "metadatas"])
    existing_docs = dict(zip(existing["ids"], existing.get("documents") or []))
    existing_metas = dict(zip(existing["ids"], existing.get("metadatas") or []))
    changed = [
        i for i, chunk_id in enumerate(ids)
        if (
            existing_docs.get(chunk_id) != docs[i]
            or existing_metas.get(chunk_id) != metas[i]
        )
    ]
    if changed:
        col.upsert(
            ids=[ids[i] for i in changed],
            documents=[docs[i] for i in changed],
            metadatas=[metas[i] for i in changed],
        )

    stale_ids = sorted(set(existing["ids"]) - set(ids))
    if stale_ids:
        col.delete(ids=stale_ids)


def _collection():
    global _collection_cache
    if _collection_cache is None:
        import chromadb

        client = chromadb.PersistentClient(path=DB_DIR)
        _collection_cache = client.get_or_create_collection("style_guides")
    _sync_collection(_collection_cache)
    return _collection_cache


def retrieve_context(language: str, code: str, static_report: str, k: int = 2) -> str:
    """Return a short context block. Small k on purpose (see Meng et al. 2025)."""
    try:
        col = _collection()
        query = f"{language} code review. {static_report[:500]}\n{code[:1500]}"
        res = col.query(query_texts=[query], n_results=max(k * 3, 6))
    except Exception as exc:
        return f"[RAG unavailable: {type(exc).__name__}]"

    # Keep language-specific guides plus the shared security/bug guides.
    prefer = {
        "python": "python_style.md", "c": "c_cpp_style.md", "cpp": "c_cpp_style.md",
        "java": "java_style.md", "ruby": "ruby_style.md",
    }[language]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    picked = [d for d, m in zip(docs, metas) if m["source"] == prefer][:k]
    picked += [d for d, m in zip(docs, metas) if m["source"] != prefer][:1]
    return "\n\n".join(picked) if picked else "[no relevant context found]"
