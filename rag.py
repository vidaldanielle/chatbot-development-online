from retriever import hybrid_search
from reranker import rerank

from langchain_community.llms import (
    Ollama
)

# =========================
# LLM
# =========================

llm = Ollama(
    model="qwen3:8b",
    temperature=0
)


# =========================
# MAIN RAG FUNCTION
# =========================

def ask_question(
    question,
    chat_history=""
):

    # =========================
    # RETRIEVAL
    # =========================

    retrieved = hybrid_search(
        question,
        top_k=10
    )

    # =========================
    # RERANKING
    # =========================

    reranked = rerank(
        question,
        retrieved
    )

    # =========================
    # BUILD CONTEXT
    # =========================

    context = "\n\n".join(

        item[0][0]

        for item in reranked[:5]

    )

    # =========================
    # COLLECT SOURCES
    # =========================

    sources = list(

        set(

            item[0][1]

            for item in reranked[:5]

        )

    )

    # =========================
    # PROMPT
    # =========================

    prompt = f"""
You are an internal company assistant.

Rules:

1. Answer only from the provided context.
2. Never use outside knowledge.
3. If information is missing, reply:
   I could not find this information in the company documents.

Chat History:

{chat_history}

Context:

{context}

Question:

{question}

Answer:
"""

    # =========================
    # LLM RESPONSE
    # =========================

    response = llm.invoke(
        prompt
    )

    # =========================
    # RETURN RESULT
    # =========================

    return {
        "answer": response.strip(),
        "sources": sources
    }