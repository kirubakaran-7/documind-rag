"""Turn uploaded files into a searchable Chroma vector store."""
import os
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma

from .config import get_embeddings

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _load_file(path):
    lower = path.lower()
    if lower.endswith(".pdf"):
        return PyPDFLoader(path).load()
    if lower.endswith((".txt", ".md")):
        return TextLoader(path, encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {path}")


def load_uploaded_files(uploaded_files):
    # Streamlit gives us in-memory files, so write them to temp files first.
    docs = []
    for uploaded in uploaded_files:
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        try:
            for doc in _load_file(tmp_path):
                doc.metadata["source"] = uploaded.name
                docs.append(doc)
        finally:
            os.unlink(tmp_path)
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vector_store(chunks, backend):
    embeddings = get_embeddings(backend)
    return Chroma.from_documents(documents=chunks, embedding=embeddings)


def ingest(uploaded_files, backend):
    docs = load_uploaded_files(uploaded_files)
    if not docs:
        raise ValueError("No readable content found in the uploaded files.")
    chunks = split_documents(docs)
    return build_vector_store(chunks, backend)
