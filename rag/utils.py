"""Small helpers for handling document sources."""


def unique_sources(documents):
    # Collect the source file names, no duplicates, sorted for a stable order.
    sources = set()
    for doc in documents:
        metadata = getattr(doc, "metadata", {}) or {}
        sources.add(metadata.get("source", "unknown"))
    return sorted(sources)


def format_sources(documents):
    sources = unique_sources(documents)
    return ", ".join(sources) if sources else "n/a"
