from retriever import hybrid_search
from reranker import rerank
from langchain_community.llms import Ollama
import time


# =========================
# LOAD LLM (LOCAL MODEL)
# =========================
# Initialize the Ollama LLM using Qwen3 model
# temperature=0 ensures deterministic and consistent outputs
llm = Ollama(
    model="qwen3:8b",
    temperature=0
)


# =========================
# MAIN RAG FUNCTION
# =========================
def ask_question(question, chat_history=""):

    # =========================
    # START RETRIEVAL TIMER
    # =========================
    t1 = time.time()

    # =========================
    # STEP 1: HYBRID RETRIEVAL
    # =========================
    # Combines:
    # - BM25 keyword-based search
    # - Dense vector semantic search
    retrieved = hybrid_search(
        question,
        top_k=10
    )

    # Log retrieval performance
    print("Retrieval time:", time.time() - t1)

    # =========================
    # HANDLE EMPTY RETRIEVAL RESULT
    # =========================
    if not retrieved:
        return {
            "answer": "I could not find this information in the company documents.",
            "sources": []
        }

    # =========================
    # START RERANK TIMER
    # =========================
    t2 = time.time()

    # =========================
    # STEP 2: RERANK RESULTS
    # =========================
    # Uses CrossEncoder model to re-score document relevance
    reranked = rerank(question, retrieved)

    print("Rerank time:", time.time() - t2)

    # =========================
    # BUILD FINAL CONTEXT
    # =========================
    # Take top 5 most relevant documents
    context = "\n\n".join(
        item[0][0]  # extract document text
        for item in reranked[:5]
    )

    # =========================
    # HANDLE EMPTY CONTEXT
    # =========================
    if not context.strip():
        return {
            "answer": "I could not find this information in the company documents.",
            "sources": []
        }

    # =========================
    # EXTRACT SOURCES
    # =========================
    # Collect unique document sources
    sources = list(
        set(
            item[0][1]  # extract source file name
            for item in reranked[:5]
        )
    )

    # =========================
    # BUILD PROMPT
    # =========================
    # Instruction set for the LLM
    prompt = f"""
You are a professional internal company assistant.

Your task is to answer questions strictly using ONLY the provided company documents.

Rules:
- Use only the given context.
- Do not use external knowledge.
- Do not hallucinate or invent information.
- If the answer is not found, respond exactly:
  I could not find this information in the company documents.

- Be concise and professional.
- Use bullet points if needed.
- Use chat history for continuity.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""

    # =========================
    # START LLM TIMER
    # =========================
    t3 = time.time()

    try:
        # =========================
        # STEP 3: GENERATE RESPONSE
        # =========================
        response = llm.invoke(prompt)

        print("LLM time:", time.time() - t3)

        # =========================
        # RETURN FINAL OUTPUT
        # =========================
        return {
            "answer": response.strip(),
            "sources": sources
        }

    except Exception as e:
        # =========================
        # HANDLE LLM ERROR
        # =========================
        print(f"LLM Error: {e}")

        return {
            "answer": "The language model is currently unavailable.",
            "sources": sources
        }