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
import json as _json
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
from core.vector_store import get_vector_store

# ---------------------------------------------------------------------------
# Module-level singletons — loaded once per Lambda container lifecycle.
# ---------------------------------------------------------------------------
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Topic vectors: populated at startup by embedding each topic description.
# Used for zero-latency classification via dot-product against the query vector.
_topic_vectors: dict[str, list[float]] = {}
_WIDGET_SECRET = os.environ.get("WIDGET_SECRET", "")
ENTITY_WEIGHT = float(os.environ.get("ENTITY_WEIGHT", "0.3"))


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
    app.state.store   = get_vector_store()
    app.state.gateway = ModelGateway(db_conn=app.state.db)

    # Pre-embed topic descriptions for fast query-time classification
    from core.db import get_topic_names
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
    start_seconds: Optional[int] = None   # None for article sections


class Source(BaseModel):
    source_type: str = "youtube_video"
    title:       str
    clips:       list[Clip]
    # YouTube-only
    video_id:    Optional[str] = None
    channel:     Optional[str] = None
    speaker:     Optional[str] = None
    # Article-only
    article_id:  Optional[str] = None
    author:      Optional[str] = None
    website:     Optional[str] = None


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
    Merge consecutive chunks from the same source that are within 30 seconds of each
    other into a single pseudo-match. For articles, merging is skipped (no start_seconds).
    """
    if not matches:
        return []

    def _sort_key(m: dict) -> tuple:
        meta = m["metadata"]
        source_id = meta.get("video_id") or meta.get("article_id", "")
        return (source_id, meta.get("start_seconds") or 0)

    sorted_m = sorted(matches, key=_sort_key)
    groups: list[list[dict]] = [[sorted_m[0]]]
    for m in sorted_m[1:]:
        prev = groups[-1][-1]
        prev_meta = prev["metadata"]
        curr_meta = m["metadata"]
        same_source = (
            curr_meta.get("video_id") == prev_meta.get("video_id")
            and curr_meta.get("video_id") is not None
        )
        close_in_time = (
            (curr_meta.get("start_seconds") or 0) - (prev_meta.get("start_seconds") or 0)
        ) <= 30
        if same_source and close_in_time:
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


def _parse_list_meta(value) -> list:
    """Coerce a metadata field that is either a native list (Pinecone) or a JSON string (Chroma)."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            result = _json.loads(value)
            return result if isinstance(result, list) else []
        except Exception:
            return []
    return []


def _entity_overlap_score(query_lower: str, meta: dict) -> float:
    """Fraction of a chunk's entities that appear as substrings in the query."""
    entities = _parse_list_meta(meta.get("entities"))
    if not entities:
        return 0.0
    matched = sum(1 for e in entities if str(e).lower() in query_lower)
    return matched / len(entities)


def _people_bonus(query_lower: str, meta: dict) -> float:
    """Return 0.15 if any of host/guests/author appears in the query, else 0.0."""
    people: list[str] = []
    if meta.get("host"):
        people.append(str(meta["host"]))
    people.extend(str(g) for g in _parse_list_meta(meta.get("guests")) if g)
    if meta.get("author"):
        people.append(str(meta["author"]))
    return 0.15 if any(p.lower() in query_lower for p in people if p) else 0.0


def _rerank(query: str, matches: list[dict], top_n: int = 5) -> list[dict]:
    if not matches:
        return []
    merged    = _merge_adjacent_chunks(matches)
    pairs     = [(query, m["metadata"].get("text_content", "")) for m in merged]
    ce_scores = _reranker.predict(pairs)
    query_lower = query.lower()
    final_scores = [
        ce
        + ENTITY_WEIGHT * _entity_overlap_score(query_lower, m["metadata"])
        + _people_bonus(query_lower, m["metadata"])
        for m, ce in zip(merged, ce_scores)
    ]
    ranked = sorted(zip(merged, final_scores), key=lambda x: x[1], reverse=True)
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


def _fetch_article_meta(db, article_ids: list[str]) -> dict[str, dict]:
    if not article_ids:
        return {}
    with db.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT a.id, a.title, a.author, w.name AS website_name
            FROM articles a
            LEFT JOIN websites w ON w.id = a.website_id
            WHERE a.id = ANY(%s)
            """,
            (article_ids,),
        )
        return {row["id"]: dict(row) for row in cur.fetchall()}


def _format_source_block(m: dict, video_lookup: dict, article_lookup: dict) -> str:
    meta = m["metadata"]
    source_type = meta.get("source_type", "youtube_video")
    chapter = meta.get("chapter", "General")
    url     = meta.get("deep_link", "")
    text    = meta.get("text_content", "")

    if source_type == "article":
        info    = article_lookup.get(meta.get("article_id", ""), {})
        t       = info.get("title", "Unknown")
        label   = f'"{t}"'
        author  = info.get("author", "")
        website = info.get("website_name", "")
        if author:
            label += f" by {author}"
        if website:
            label += f" ({website})"
    else:
        info    = video_lookup.get(meta.get("video_id", ""), {})
        t       = info.get("title", "Unknown")
        speaker = _extract_speaker(t)
        label   = f'"{t}"' + (f" — {speaker}" if speaker else "")

    return f"[Source: {label} | Section: {chapter} | {url}]\n{text}"


def _sse(event_type: str, data: dict) -> str:
    return f"data: {_json.dumps({'type': event_type, **data})}\n\n"


def _log_query(
    db, query: str, topic: str, video_ids: list[str], article_ids: list[str], cost: float
) -> None:
    try:
        with db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_queries (user_query, queried_topic, video_ids, article_ids, retrieval_cost)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (query, topic, video_ids, article_ids, cost),
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
    store   = app.state.store
    gateway = app.state.gateway

    # Resolve or create conversation
    conversation_id = request.conversation_id
    session_id = request.session_id or "anonymous"
    if not conversation_id:
        conversation_id = create_conversation(db, session_id=session_id, user_id=user_id, title=query[:100])

    # Load prior history (last 6 messages = 3 turns)
    history = get_conversation_messages(db, conversation_id, limit=6)

    # Step A — embed query, classify topic via dot-product (~0ms, no extra API call)
    embed_resp = gateway.get_embedding(query, associated_id="query_embed")
    predicted_topic = _nearest_topic(embed_resp.embedding_vector)

    # Step B — hybrid/dense search filtered by nearest topic
    matches = store.query(
        embedding=embed_resp.embedding_vector,
        n_results=20,
        where={"primary_topic": predicted_topic},
        query_text=query,
    )

    _no_results_msg = "I could not find relevant content for that query in the indexed sources."

    if not matches:
        save_message(db, conversation_id, "user", query)
        save_message(db, conversation_id, "assistant", _no_results_msg)
        if request.stream:
            def _empty_stream():
                yield _sse("done", {
                    "answer": _no_results_msg,
                    "topic": "",
                    "sources": [],
                    "conversation_id": conversation_id,
                })
            return StreamingResponse(_empty_stream(), media_type="text/event-stream")
        return ChatResponse(
            answer=_no_results_msg,
            topic="",
            sources=[],
            conversation_id=conversation_id,
        )

    top5 = _rerank(query, matches, top_n=5)

    # Step C — fetch source metadata for context labels and response building
    video_ids   = list({m["metadata"]["video_id"]   for m in top5 if m["metadata"].get("video_id")})
    article_ids = list({m["metadata"]["article_id"] for m in top5 if m["metadata"].get("article_id")})
    video_lookup   = _fetch_video_meta(db, video_ids)
    article_lookup = _fetch_article_meta(db, article_ids)

    context = "\n\n---\n\n".join(
        _format_source_block(m, video_lookup, article_lookup)
        for m in top5
    )

    # Build prompt with optional conversation history
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
        "Each source is labelled with its title and author/speaker name. "
        "When citing a specific claim, attribute it inline — e.g. 'According to [Author]...' "
        "or '[Speaker] describes this as...'. Never use generic phrases like 'the text' or "
        "'the source'. "
        "Quote or closely paraphrase the most illuminating passage when it strengthens the answer. "
        "If multiple sources address different facets, synthesise them into a coherent explanation. "
        "Do not fabricate information not present in the sources.\n\n"
        "Sources:\n" + context
    )

    # Build grouped sources — max 2 clips per source, preserving re-rank order
    seen: dict[str, int] = defaultdict(int)
    groups: dict[str, list] = defaultdict(list)
    for m in top5:
        meta = m["metadata"]
        source_type = meta.get("source_type", "youtube_video")
        source_key = meta.get("video_id") if source_type == "youtube_video" else meta.get("article_id")
        if source_key and seen[source_key] < 2:
            groups[source_key].append(m)
            seen[source_key] += 1

    sources = []
    for source_key, chunks in groups.items():
        meta = chunks[0]["metadata"]
        source_type = meta.get("source_type", "youtube_video")

        if source_type == "article":
            info = article_lookup.get(source_key, {})
            sources.append(Source(
                source_type = "article",
                title       = info.get("title", "Unknown"),
                author      = info.get("author") or None,
                website     = info.get("website_name") or None,
                article_id  = source_key,
                clips=[
                    Clip(
                        chapter = c["metadata"].get("chapter", "General"),
                        url     = c["metadata"].get("deep_link", ""),
                    )
                    for c in chunks
                ],
            ))
        else:
            info = video_lookup.get(source_key, {})
            title = info.get("title", "Unknown")
            sources.append(Source(
                source_type = "youtube_video",
                title       = title,
                channel     = info.get("channel_name", "Unknown"),
                speaker     = _extract_speaker(title) or None,
                video_id    = source_key,
                clips=[
                    Clip(
                        chapter       = c["metadata"].get("chapter", "General"),
                        url           = c["metadata"].get("deep_link", ""),
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
            _log_query(db, query, predicted_topic, video_ids, article_ids, embed_resp.cost)

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
    _log_query(db, query, predicted_topic, video_ids, article_ids, total_cost)

    return ChatResponse(
        answer=synthesis_resp.text_content,
        topic=predicted_topic,
        sources=sources,
        conversation_id=conversation_id,
    )
