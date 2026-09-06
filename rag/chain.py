"""Take a question, pull the relevant chunks, and let the LLM answer from them."""
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from .config import get_chat_model
from .utils import unique_sources

# Telling the model to stick to the context is what stops it making things up.
SYSTEM_PROMPT = (
    "You are DocuMind, a helpful assistant that answers questions using only "
    "the provided context from the user's documents.\n"
    "Rules:\n"
    "1. Answer only from the context below. Do not use outside knowledge.\n"
    "2. If the answer is not in the context, say: "
    '"I could not find that in the provided documents."\n'
    "3. Be concise and mention the source file name(s) you used.\n\n"
    "Context:\n{context}"
)


def build_qa_chain(vector_store, backend, k=4):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{input}")]
    )
    llm = get_chat_model(backend)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)


def answer_question(qa_chain, question):
    result = qa_chain.invoke({"input": question})
    sources = unique_sources(result.get("context", []))
    return {"answer": result["answer"], "sources": sources}
