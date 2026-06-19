"""
core/vector_store.py
Thin adapter over Pinecone (production) and Chroma (local dev).

Switch backend via VECTOR_STORE env var:
  VECTOR_STORE=pinecone  (default) — hybrid dense+BM25, requires PINECONE_* env vars
  VECTOR_STORE=chroma    — dense-only, fully local, requires: uv sync --extra dev
"""
import os


class PineconeStore:
    """Hybrid dense+sparse search via Pinecone. Production backend."""

    _ALPHA = 0.7  # dense weight; (1 - alpha) applied to sparse

    def __init__(self) -> None:
        from pinecone import Pinecone
        from pinecone_text.sparse import BM25Encoder

        self._index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(
            os.environ["PINECONE_INDEX_NAME"]
        )
        self._bm25 = BM25Encoder.default()
        self._namespace = os.environ.get("PINECONE_NAMESPACE", "")

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        texts: list[str],
    ) -> None:
        vectors = []
        for vec_id, embedding, metadata, text in zip(ids, embeddings, metadatas, texts):
            sparse = self._bm25.encode_documents([text])[0]
            if not sparse.get("indices"):
                continue  # too short / all stopwords — skip
            vectors.append({
                "id": vec_id,
                "values": embedding,
                "sparse_values": sparse,
                "metadata": {**metadata, "text_content": text[:1000]},
            })
        for i in range(0, len(vectors), 100):
            self._index.upsert(vectors=vectors[i : i + 100], namespace=self._namespace)

    def query(
        self,
        embedding: list[float],
        n_results: int = 20,
        where: dict | None = None,
        query_text: str = "",
    ) -> list[dict]:
        """Return list of {metadata: {...}} dicts, normalised from Pinecone matches."""
        pinecone_filter = None
        if where and "primary_topic" in where:
            # Old vectors use topics[] array; new ones also have primary_topic scalar.
            # $in on the array field provides backward compat with both.
            pinecone_filter = {"topics": {"$in": [where["primary_topic"]]}}

        dense = [v * self._ALPHA for v in embedding]
        sparse_raw = self._bm25.encode_queries([query_text or " "])[0]
        sparse = {
            "indices": sparse_raw["indices"],
            "values": [v * (1 - self._ALPHA) for v in sparse_raw["values"]],
        }

        results = self._index.query(
            vector=dense,
            sparse_vector=sparse,
            top_k=n_results,
            filter=pinecone_filter,
            include_metadata=True,
            namespace=self._namespace,
        )
        return [{"metadata": m["metadata"]} for m in results.get("matches", [])]

    def delete(self, ids: list[str]) -> None:
        self._index.delete(ids=ids, namespace=self._namespace)


class ChromaStore:
    """Dense-only search via Chroma. Local dev backend (zero cost, zero cloud deps)."""

    _COLLECTION = "rag"

    def __init__(self, path: str = "/tmp/chroma") -> None:
        import chromadb  # optional dep: uv sync --extra dev

        self._client = chromadb.PersistentClient(path=path)
        self._col = self._client.get_or_create_collection(self._COLLECTION)

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        texts: list[str],
    ) -> None:
        if not ids:
            return
        # Chroma requires metadata values to be str/int/float/bool — convert lists to str
        safe_metas = [_safe_chroma_meta(m) for m in metadatas]
        self._col.upsert(ids=ids, embeddings=embeddings, metadatas=safe_metas, documents=texts)

    def query(
        self,
        embedding: list[float],
        n_results: int = 20,
        where: dict | None = None,
        query_text: str = "",
    ) -> list[dict]:
        """Return list of {metadata: {...}} dicts, normalised from Chroma results."""
        chroma_where = None
        if where and "primary_topic" in where:
            chroma_where = {"primary_topic": {"$eq": where["primary_topic"]}}

        kwargs: dict = {"include": ["metadatas", "documents"]}
        if chroma_where:
            kwargs["where"] = chroma_where

        # Guard: Chroma errors if n_results > collection size
        count = self._col.count()
        actual_n = min(n_results, count) if count > 0 else 0
        if actual_n == 0:
            return []

        results = self._col.query(
            query_embeddings=[embedding],
            n_results=actual_n,
            **kwargs,
        )
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]

        return [
            {"metadata": {**meta, "text_content": doc}}
            for meta, doc in zip(metadatas, documents)
        ]

    def delete(self, ids: list[str]) -> None:
        self._col.delete(ids=ids)


# Type alias for use in function signatures across ingestion and retrieval modules
VectorStore = PineconeStore | ChromaStore


def get_vector_store() -> VectorStore:
    """Factory: read VECTOR_STORE env var and return the appropriate backend."""
    backend = os.environ.get("VECTOR_STORE", "pinecone").lower()
    if backend == "chroma":
        path = os.environ.get("CHROMA_PATH", "/tmp/chroma")
        return ChromaStore(path=path)
    return PineconeStore()


def _safe_chroma_meta(meta: dict) -> dict:
    """Chroma metadata values must be scalar (str/int/float/bool). Convert lists to JSON strings."""
    import json
    return {
        k: json.dumps(v) if isinstance(v, (list, dict)) else v
        for k, v in meta.items()
        if v is not None
    }
