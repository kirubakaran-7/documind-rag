"""Build the chat + embedding models for whichever backend we're using."""
import os

from dotenv import load_dotenv

load_dotenv()

OPENAI = "OpenAI"
OLLAMA = "Ollama"


def get_chat_model(backend, temperature=0.0):
    if backend == OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )
    if backend == OLLAMA:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.1"),
            temperature=temperature,
        )
    raise ValueError(f"Unknown backend: {backend}")


def get_embeddings(backend):
    if backend == OPENAI:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        )
    if backend == OLLAMA:
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        )
    raise ValueError(f"Unknown backend: {backend}")
