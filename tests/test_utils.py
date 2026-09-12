"""Tests for the source helpers and backend checks. No API key needed."""
from dataclasses import dataclass, field

import pytest

from rag.config import get_chat_model, get_embeddings
from rag.utils import format_sources, unique_sources


@dataclass
class FakeDoc:
    metadata: dict = field(default_factory=dict)


def test_unique_sources_dedupes_and_sorts():
    docs = [
        FakeDoc({"source": "b.pdf"}),
        FakeDoc({"source": "a.pdf"}),
        FakeDoc({"source": "b.pdf"}),
    ]
    assert unique_sources(docs) == ["a.pdf", "b.pdf"]


def test_unique_sources_handles_missing_metadata():
    docs = [FakeDoc({}), FakeDoc({"source": "a.txt"})]
    assert unique_sources(docs) == ["a.txt", "unknown"]


def test_format_sources_empty():
    assert format_sources([]) == "n/a"


def test_format_sources_joins():
    docs = [FakeDoc({"source": "a.pdf"}), FakeDoc({"source": "b.pdf"})]
    assert format_sources(docs) == "a.pdf, b.pdf"


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_chat_model("Gemini")
    with pytest.raises(ValueError):
        get_embeddings("Gemini")
