import json
import os
import pathlib

import pytest
from dotenv import load_dotenv

load_dotenv()

FIXTURE_FILE = pathlib.Path(__file__).parent / "fixtures" / "seed_chunks.json"
SCHEMA_FILE  = pathlib.Path(__file__).parent.parent.parent / "schema.sql"

TOPICS_SEED = [
    ("consciousness",      "Human consciousness, mind, awareness, and perception"),
    ("alternative_history","Alternative history, ancient Egypt, Atlantis, lost civilizations"),
    ("biohacking",         "Biohacking, longevity, nootropics, and performance optimization"),
    ("spirituality",       "Spirituality, metaphysics, meditation, and esoteric knowledge"),
]


def _llm_judge(query: str, answer: str, sources: list[dict]) -> dict:
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    source_titles = [s["title"] for s in sources]
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "You are evaluating a RAG system answer. Score on three axes (1–10):\n"
                "- relevance: does the answer address the question?\n"
                "- grounding: are claims traceable to the provided sources?\n"
                "- attribution: are speakers/authors cited by name (not 'the text')?\n"
                "Return only valid JSON: "
                '{"relevance": N, "grounding": N, "attribution": N, "reason": "..."}'
            )},
            {"role": "user", "content": (
                f"Question: {query}\n\n"
                f"Sources available: {source_titles}\n\n"
                f"Answer: {answer}"
            )},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


@pytest.fixture(scope="session")
def real_db():
    import psycopg2
    dsn = os.environ["DATABASE_URL"]
    # Inject search_path=test into DSN so all queries use the test schema
    sep = "&" if "?" in dsn else "?"
    test_dsn = dsn + sep + "options=-c%20search_path%3Dtest"
    conn = psycopg2.connect(test_dsn)
    return conn


@pytest.fixture(scope="session", autouse=True)
def test_schema(real_db):
    """Create isolated test schema, seed topics, tear down after session."""
    if not FIXTURE_FILE.exists():
        pytest.skip(
            f"Seed data not found: {FIXTURE_FILE}\n"
            f"Copy {FIXTURE_FILE.parent / 'seed_chunks.example.json'} to seed_chunks.json "
            "and fill in real transcript excerpts."
        )

    with real_db.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS test")
        cur.execute("SET search_path TO test")
        cur.execute(SCHEMA_FILE.read_text())
        cur.executemany(
            "INSERT INTO topics (name, description) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
            TOPICS_SEED,
        )
    real_db.commit()

    yield

    with real_db.cursor() as cur:
        cur.execute("DROP SCHEMA test CASCADE")
    real_db.commit()
    real_db.close()


@pytest.fixture(scope="session", autouse=True)
def seed_pinecone(test_schema):
    """Embed seed chunks and upsert into the 'test' Pinecone namespace. Clean up after session."""
    from pinecone import Pinecone
    from pinecone_text.sparse import BM25Encoder

    from core.gateway import ModelGateway

    chunks = json.loads(FIXTURE_FILE.read_text())
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(
        os.environ["PINECONE_INDEX_NAME"]
    )
    gateway = ModelGateway()
    bm25 = BM25Encoder.default()

    vectors = []
    for c in chunks:
        emb = gateway.get_embedding(c["text_content"])
        sparse = bm25.encode_documents([c["text_content"]])[0]
        if not sparse.get("indices"):
            continue
        vectors.append({
            "id": f"test_{c['video_id']}_{c['start_seconds']}",
            "values": emb.embedding_vector,
            "sparse_values": sparse,
            "metadata": {k: c[k] for k in (
                "video_id", "channel_id", "topics",
                "chapter", "start_seconds", "deep_link", "text_content"
            )},
        })

    if vectors:
        index.upsert(vectors=vectors, namespace="test")

    yield

    index.delete(delete_all=True, namespace="test")


@pytest.fixture(scope="session")
def real_client(real_db, seed_pinecone):
    """Live TestClient pointing at the test schema and test Pinecone namespace."""
    import os

    from fastapi.testclient import TestClient

    os.environ["PINECONE_NAMESPACE"] = "test"
    dsn = os.environ["DATABASE_URL"]
    sep = "&" if "?" in dsn else "?"
    os.environ["DATABASE_URL"] = dsn + sep + "options=-c%20search_path%3Dtest"

    from retrieval.main import app
    with TestClient(app) as client:
        yield client


# Export judge for use in test file
pytest.llm_judge = _llm_judge
