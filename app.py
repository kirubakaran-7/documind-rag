"""DocuMind - ask questions about your own documents.

Run:  streamlit run app.py
"""
import streamlit as st

from rag.chain import answer_question, build_qa_chain
from rag.config import OLLAMA, OPENAI
from rag.ingest import ingest

st.set_page_config(page_title="DocuMind - RAG Q&A", page_icon="📄")
st.title("📄 DocuMind")
st.caption("Ask questions about your own documents. Answers come from the files you upload.")

# Sidebar: pick a backend and upload files
with st.sidebar:
    st.header("Setup")
    backend = st.radio("LLM backend", [OPENAI, OLLAMA], index=0)
    st.markdown(
        "- **OpenAI**: needs `OPENAI_API_KEY` in `.env`\n"
        "- **Ollama**: needs a local Ollama server running"
    )
    uploaded = st.file_uploader(
        "Upload PDF / TXT / MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Build knowledge base", type="primary"):
        if not uploaded:
            st.warning("Please upload at least one document first.")
        else:
            with st.spinner("Reading, chunking and embedding your documents..."):
                try:
                    vs = ingest(uploaded, backend)
                    st.session_state.qa_chain = build_qa_chain(vs, backend)
                    st.session_state.messages = []
                    st.success("Knowledge base ready. Ask a question below!")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

# Chat area
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something about your documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if "qa_chain" not in st.session_state:
        reply = "Build a knowledge base first (upload files in the sidebar)."
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                try:
                    result = answer_question(st.session_state.qa_chain, question)
                    sources = ", ".join(result["sources"]) or "n/a"
                    reply = f"{result['answer']}\n\n*Sources: {sources}*"
                except Exception as exc:
                    reply = f"Error while answering: {exc}"
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
