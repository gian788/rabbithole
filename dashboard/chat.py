"""
dashboard/chat.py
Public-facing Streamlit chat UI for the YouTube Topic RAG system.

Run:
    streamlit run dashboard/chat.py
"""
import uuid

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Topic RAG Chat", page_icon="🔍", layout="centered")
st.title("Ask the Archive")
st.caption("Multi-turn Q&A over indexed YouTube content.")


# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, citations}


def _start_new_conversation():
    try:
        resp = requests.post(
            f"{API_BASE}/v1/conversations",
            json={"session_id": st.session_state.session_id},
            timeout=10,
        )
        resp.raise_for_status()
        st.session_state.conversation_id = resp.json()["conversation_id"]
        st.session_state.messages = []
    except Exception as e:
        st.error(f"Could not start conversation: {e}")


def _restore_conversation(conversation_id: str):
    try:
        resp = requests.get(
            f"{API_BASE}/v1/conversations/{conversation_id}/messages",
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()["messages"]
        st.session_state.messages = [
            {
                "role":      m["role"],
                "content":   m["content"],
                "citations": m.get("citations") or [],
            }
            for m in raw
        ]
    except Exception as e:
        st.warning(f"Could not restore history: {e}")
        st.session_state.messages = []


# On first load, create a conversation and restore any history
if st.session_state.conversation_id is None:
    _start_new_conversation()
elif not st.session_state.messages:
    _restore_conversation(st.session_state.conversation_id)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Conversation")
    if st.button("New Chat", use_container_width=True):
        _start_new_conversation()
        st.rerun()

    if st.session_state.conversation_id:
        st.caption(f"ID: `{st.session_state.conversation_id[:8]}…`")


# ---------------------------------------------------------------------------
# Render message history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        citations = msg.get("citations") or []
        if citations and msg["role"] == "assistant":
            with st.expander(f"Sources ({len(citations)})", expanded=False):
                for c in citations:
                    st.markdown(
                        f"**[{c['title']}]({c['url']})** — {c['channel']}\n\n"
                        f"_{c['chapter']}_ · {c['start_seconds']}s"
                    )


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask something…"):
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "citations": []})

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/v1/chat",
                    json={
                        "query":           prompt,
                        "conversation_id": st.session_state.conversation_id,
                        "session_id":      st.session_state.session_id,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()

                answer    = data["answer"]
                citations = data.get("citations", [])
                # Update conversation_id in case it was created server-side
                st.session_state.conversation_id = data["conversation_id"]

                st.markdown(answer)
                if citations:
                    with st.expander(f"Sources ({len(citations)})", expanded=False):
                        for c in citations:
                            st.markdown(
                                f"**[{c['title']}]({c['url']})** — {c['channel']}\n\n"
                                f"_{c['chapter']}_ · {c['start_seconds']}s"
                            )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "citations": citations}
                )

            except requests.HTTPError as e:
                err = f"API error {e.response.status_code}: {e.response.text}"
                st.error(err)
            except Exception as e:
                st.error(f"Request failed: {e}")
