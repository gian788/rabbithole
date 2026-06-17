"""
retrieval/main.py
FastAPI application wrapped in Mangum for AWS Lambda + HTTP API Gateway.

Endpoints
---------
GET  /v1/health
POST /v1/conversations
GET  /v1/conversations/{conversation_id}/messages
POST /v1/chat   {"query": str, "conversation_id"?: str, "session_id"?: str}
"""
import json
import os
import re
import sys
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, Iterator, Optional

import psycopg2
import psycopg2.extras
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from jose import JWTError, jwt
from mangum import Mangum
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from core.db import (
    create_conversation,
    get_connection,
    get_conversation_messages,
    get_conversation_owner,
    get_user_conversations,
    save_message,
    update_conversation,
)
from core.gateway import ModelGateway

# ---------------------------------------------------------------------------
# Module-level singletons — loaded once per Lambda container lifecycle.
# Both models are CPU-compatible and cache under SENTENCE_TRANSFORMERS_HOME.
# ---------------------------------------------------------------------------
_bm25     = BM25Encoder.default()
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Topic vectors: populated at startup by embedding each topic description.
# Used for zero-latency classification via dot-product against the query vector.
_topic_vectors: dict[str, list[float]] = {}
_PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "")
_WIDGET_SECRET      = os.environ.get("WIDGET_SECRET", "")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def get_current_user(authorization: str = Header(...)) -> str:
    """Verify HS256 JWT from the widget and return the userId (sub claim). Requires auth."""
    if not _WIDGET_SECRET:
        raise HTTPException(status_code=500, detail="WIDGET_SECRET not configured")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = jwt.decode(token, _WIDGET_SECRET, algorithms=["HS256"])
        user_id: str = payload.get("sub", "")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing sub")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def optional_current_user(authorization: Optional[str] = Header(default=None)) -> Optional[str]:
    """Like get_current_user but returns None when no Authorization header is sent."""
    if not authorization or not _WIDGET_SECRET:
        return None
    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = jwt.decode(token, _WIDGET_SECRET, algorithms=["HS256"])
        return payload.get("sub") or None
    except JWTError:
        return None


# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db      = get_connection()
    app.state.index   = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(
        os.environ["PINECONE_INDEX_NAME"]
    )
    app.state.gateway = ModelGateway(db_conn=app.state.db)

    # Pre-embed topic descriptions for fast query-time classification
    import psycopg2.extras
    with app.state.db.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT name, description FROM topics")
        topics_rows = cur.fetchall()
    for row in topics_rows:
        resp = app.state.gateway.get_embedding(
            f"{row['name']}: {row['description']}", associated_id="topic_centroid"
        )
        _topic_vectors[row["name"]] = resp.embedding_vector

    yield
    app.state.db.close()


app     = FastAPI(title="YouTube Topic RAG API", version="1.0.0", lifespan=lifespan)
handler = Mangum(app)   # AWS Lambda entry point

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ConversationSummary(BaseModel):
    id:              str
    title:           str
    topic:           Optional[str] = None
    last_message_at: Optional[str] = None
    preview:         str


class ConversationRequest(BaseModel):
    session_id: Optional[str] = None


class MessageOut(BaseModel):
    role:       str
    content:    str
    citations:  Optional[list] = None
    created_at: str


class ConversationResponse(BaseModel):
    conversation_id: str
    messages:        list[MessageOut]


class ChatRequest(BaseModel):
    query:           str
    conversation_id: Optional[str] = None
    session_id:      Optional[str] = None
    stream:          bool          = False


class Clip(BaseModel):
    chapter:       str
    url:           str
    start_seconds: int


class Source(BaseModel):
    video_id: str
    title:    str
    channel:  str
    speaker:  str        # parsed from "Episode Title | Guest Name"
    clips:    list[Clip] # max 2 per source video


class ChatResponse(BaseModel):
    answer:          str
    topic:           str
    sources:         list[Source]
    conversation_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nearest_topic(query_vector: list[float]) -> str:
    """Return the topic whose embedding has the highest dot-product with the query vector."""
    best, best_score = "", float("-inf")
    for topic, tvec in _topic_vectors.items():
        score = sum(q * t for q, t in zip(query_vector, tvec))
        if score > best_score:
            best, best_score = topic, score
    return best


def _extract_speaker(title: str) -> str:
    """Parse guest name from 'Episode Title | Guest Name' podcast title format."""
    m = re.search(r"\|\s*(.+)$", title)
    return m.group(1).strip() if m else ""


def _ensure_db(state: Any) -> None:
    """Reconnect if the psycopg2 connection was closed (Lambda idle or Neon SSL drop)."""
    try:
        with state.db.cursor() as cur:
            cur.execute("SELECT 1")
    except (psycopg2.InterfaceError, psycopg2.OperationalError):
        state.db = get_connection()
        state.gateway.db_conn = state.db



def _merge_adjacent_chunks(matches: list[dict]) -> list[dict]:
    """
    Merge consecutive chunks from the same video that are within 30 seconds of each
    other into a single pseudo-match. The merged match inherits the first chunk's
    metadata and concatenates text_content so the cross-encoder sees full context.
    """
    if not matches:
        return []
    # Sort by video + start_seconds so neighbours are adjacent
    sorted_m = sorted(
        matches,
        key=lambda m: (m["metadata"]["video_id"], m["metadata"].get("start_seconds", 0)),
    )
    groups: list[list[dict]] = [[sorted_m[0]]]
    for m in sorted_m[1:]:
        prev = groups[-1][-1]
        same_video = m["metadata"]["video_id"] == prev["metadata"]["video_id"]
        close_in_time = (
            m["metadata"].get("start_seconds", 0) - prev["metadata"].get("start_seconds", 0)
        ) <= 30
        if same_video and close_in_time:
            groups[-1].append(m)
        else:
            groups.append([m])

    merged = []
    for group in groups:
        if len(group) == 1:
            merged.append({"metadata": dict(group[0]["metadata"])})
        else:
            meta = dict(group[0]["metadata"])
            meta["text_content"] = " ".join(
                g["metadata"].get("text_content", "") for g in group
            )
            merged.append({"metadata": meta})
    return merged


def _rerank(query: str, matches: list[dict], top_n: int = 5) -> list[dict]:
    if not matches:
        return []
    merged = _merge_adjacent_chunks(matches)
    pairs  = [(query, m["metadata"].get("text_content", "")) for m in merged]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(merged, scores), key=lambda x: x[1], reverse=True)
    return [m for m, _ in ranked[:top_n]]


def _fetch_video_meta(db, video_ids: list[str]) -> dict[str, dict]:
    if not video_ids:
        return {}
    with db.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT v.id, v.title, c.name AS channel_name
            FROM videos v
            JOIN channels c ON c.id = v.channel_id
            WHERE v.id = ANY(%s)
            """,
            (video_ids,),
        )
        return {row["id"]: dict(row) for row in cur.fetchall()}


def _format_source_block(m: dict, video_lookup: dict[str, dict]) -> str:
    meta    = video_lookup.get(m["metadata"]["video_id"], {})
    title   = meta.get("title", "Unknown")
    speaker = _extract_speaker(title)
    label   = f'"{title}"' + (f" — {speaker}" if speaker else "")
    chapter = m["metadata"].get("chapter", "General")
    url     = m["metadata"].get("deep_link", "")
    text    = m["metadata"].get("text_content", "")
    return f"[Source: {label} | Chapter: {chapter} | {url}]\n{text}"


def _sse(event_type: str, data: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


def _log_query(db, query: str, topic: str, video_ids: list[str], cost: float) -> None:
    try:
        with db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_queries (user_query, queried_topic, video_ids, retrieval_cost)
                VALUES (%s, %s, %s, %s)
                """,
                (query, topic, video_ids, cost),
            )
        db.commit()
    except Exception as exc:
        print(f"[chat] analytics log failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/v1/conversations", response_model=list[ConversationSummary])
def list_conversations(user_id: str = Depends(get_current_user)):
    _ensure_db(app.state)
    rows = get_user_conversations(app.state.db, user_id)
    return [ConversationSummary(**r) for r in rows]


@app.post("/v1/conversations", response_model=ConversationResponse)
def create_conv(request: ConversationRequest, user_id: str = Depends(get_current_user)):
    _ensure_db(app.state)
    db = app.state.db
    session_id = request.session_id or "anonymous"
    conversation_id = create_conversation(db, session_id=session_id, user_id=user_id)
    return ConversationResponse(conversation_id=conversation_id, messages=[])


@app.get("/v1/conversations/{conversation_id}/messages", response_model=ConversationResponse)
def get_messages(conversation_id: str, user_id: str = Depends(get_current_user)):
    _ensure_db(app.state)
    db = app.state.db
    owner = get_conversation_owner(db, conversation_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if owner != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    rows = get_conversation_messages(db, conversation_id, limit=50)
    messages = [
        MessageOut(
            role=r["role"],
            content=r["content"],
            citations=r["citations"],
            created_at=r["created_at"].isoformat(),
        )
        for r in rows
    ]
    return ConversationResponse(conversation_id=conversation_id, messages=messages)


@app.post("/v1/chat")
def chat(request: ChatRequest, user_id: Optional[str] = Depends(optional_current_user)):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    _ensure_db(app.state)

    db      = app.state.db
    index   = app.state.index
    gateway = app.state.gateway

    # Resolve or create conversation
    conversation_id = request.conversation_id
    session_id = request.session_id or "anonymous"
    if not conversation_id:
        conversation_id = create_conversation(db, session_id=session_id, user_id=user_id, title=query[:100])

    # Load prior history (last 6 messages = 3 turns)
    history = get_conversation_messages(db, conversation_id, limit=6)

    # Step A — embed query, classify topic via dot-product (~0ms, no extra API call)
    _alpha     = 0.7
    embed_resp = gateway.get_embedding(query, associated_id="query_embed")

    predicted_topic = _nearest_topic(embed_resp.embedding_vector)

    dense_vector = [v * _alpha for v in embed_resp.embedding_vector]
    raw_sparse   = _bm25.encode_queries([query])[0]
    sparse_vector = {
        "indices": raw_sparse["indices"],
        "values":  [v * (1 - _alpha) for v in raw_sparse["values"]],
    }

    # Step B — hybrid search filtered by nearest topic
    results = index.query(
        vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=20,
        filter={"topics": {"$in": [predicted_topic]}},
        include_metadata=True,
        namespace=_PINECONE_NAMESPACE,
    )
    matches = results.get("matches", [])

    if not matches:
        save_message(db, conversation_id, "user", query)
        save_message(db, conversation_id, "assistant", "I could not find relevant content for that query in the indexed videos.")
        if request.stream:
            def _empty_stream():
                yield _sse("done", {
                    "answer": "I could not find relevant content for that query in the indexed videos.",
                    "topic": "",
                    "sources": [],
                    "conversation_id": conversation_id,
                })
            return StreamingResponse(_empty_stream(), media_type="text/event-stream")
        return ChatResponse(
            answer="I could not find relevant content for that query in the indexed videos.",
            topic="",
            sources=[],
            conversation_id=conversation_id,
        )

    top5 = _rerank(query, matches, top_n=5)

    # Step C — fetch video metadata first so speaker names are available for context labels
    video_ids    = list({m["metadata"]["video_id"] for m in top5})
    video_lookup = _fetch_video_meta(db, video_ids)

    context = "\n\n---\n\n".join(
        _format_source_block(m, video_lookup)
        for m in top5
    )

    # Build prompt: prepend conversation history if present
    if history:
        history_lines = []
        for msg in history:
            prefix = "Human" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{prefix}: {msg['content']}")
        history_text = "\n".join(history_lines)
        prompt = f"{history_text}\nHuman: {query}"
    else:
        prompt = query

    system_prompt = (
        "You are an expert research assistant specialising in consciousness, spirituality, "
        "alternative history, and biohacking. Your audience is curious, intelligent people "
        "who want to deeply understand the concepts they ask about.\n\n"
        "Answer using ONLY the provided source excerpts. "
        "For complex or esoteric concepts, give a thorough explanation — define the term, "
        "explain its significance, describe how it is used in context, and include any "
        "relevant nuance from the sources. Aim for 3–5 sentences minimum; use more when "
        "the concept warrants it. "
        "Each source is labelled with its video title and speaker name. "
        "When citing a specific claim, attribute it inline — e.g. 'According to [Speaker]...' "
        "or '[Speaker] describes this as...'. Never use generic phrases like 'the text' or "
        "'the source'. "
        "Quote or closely paraphrase the most illuminating passage when it strengthens the answer. "
        "If multiple sources address different facets, synthesise them into a coherent explanation. "
        "Do not fabricate information not present in the sources.\n\n"
        "Sources:\n" + context
    )

    # Build grouped sources — max 2 clips per video, preserving re-rank order
    seen: dict[str, int] = defaultdict(int)
    groups: dict[str, list] = defaultdict(list)
    for m in top5:
        vid = m["metadata"]["video_id"]
        if seen[vid] < 2:
            groups[vid].append(m)
            seen[vid] += 1

    sources = []
    for vid, chunks in groups.items():
        meta    = video_lookup.get(vid, {})
        title   = meta.get("title", "Unknown")
        sources.append(Source(
            video_id = vid,
            title    = title,
            channel  = meta.get("channel_name", "Unknown"),
            speaker  = _extract_speaker(title),
            clips    = [
                Clip(
                    chapter       = c["metadata"].get("chapter", "General"),
                    url           = c["metadata"]["deep_link"],
                    start_seconds = int(c["metadata"].get("start_seconds", 0)),
                )
                for c in chunks
            ],
        ))

    sources_data = [s.model_dump() for s in sources]

    # Step D — synthesis: branch on stream flag
    if request.stream:
        def event_stream() -> Iterator[str]:
            collected: list[str] = []
            for token in gateway.stream_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                model="gpt-4o-mini",
                provider="openai",
                associated_id="synthesis",
            ):
                collected.append(token)
                yield _sse("token", {"content": token})

            full_answer = "".join(collected)
            save_message(db, conversation_id, "user", query)
            save_message(db, conversation_id, "assistant", full_answer, citations=sources_data)
            update_conversation(db, conversation_id, topic=predicted_topic)
            _log_query(db, query, predicted_topic, video_ids, embed_resp.cost)

            yield _sse("done", {
                "answer": full_answer,
                "topic": predicted_topic,
                "sources": sources_data,
                "conversation_id": conversation_id,
            })

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming path (default — safe for Postman / curl testing)
    synthesis_resp = gateway.get_completion(
        prompt=prompt,
        system_prompt=system_prompt,
        model="gpt-4o-mini",
        provider="openai",
        associated_id="synthesis",
    )

    save_message(db, conversation_id, "user", query)
    save_message(db, conversation_id, "assistant", synthesis_resp.text_content, citations=sources_data)
    update_conversation(db, conversation_id, topic=predicted_topic)

    total_cost = embed_resp.cost + synthesis_resp.cost
    _log_query(db, query, predicted_topic, video_ids, total_cost)

    return ChatResponse(
        answer=synthesis_resp.text_content,
        topic=predicted_topic,
        sources=sources,
        conversation_id=conversation_id,
    )
