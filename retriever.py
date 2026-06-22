from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import numpy as np


# =========================
# LOAD EMBEDDING MODEL
# =========================
# This model is used for semantic (dense vector) search
embedding_model = SentenceTransformer("BAAI/bge-m3")


# =========================
# INIT QDRANT CLIENT
# =========================
# Connects to local Qdrant vector database
client = QdrantClient(path="./qdrant_data")


# =========================
# LOAD DOCUMENTS FROM QDRANT
# =========================
def load_documents():

    # =========================
    # SCROLL COLLECTION DATA
    # =========================
    # Retrieves all stored documents from Qdrant collection
    collection, _ = client.scroll(
        collection_name="company_docs",
        limit=10000,
        with_payload=True
    )

    # =========================
    # FORMAT DOCUMENTS
    # =========================
    # Converts Qdrant points into usable format:
    # (text, source)
    return [
        (
            item.payload["text"],
            item.payload.get("source", "Unknown")
        )
        for item in collection
    ]


# =========================
# BUILD BM25 INDEX
# =========================
def build_bm25(documents):

    # Tokenize each document by splitting words
    tokenized = [doc[0].split() for doc in documents]

    # Create BM25 model for keyword-based retrieval
    return BM25Okapi(tokenized)


# =========================
# HYBRID SEARCH FUNCTION
# =========================
def hybrid_search(query, top_k=10):

    # =========================
    # LOAD DOCUMENTS
    # =========================
    # Fetch all documents from vector database
    documents = load_documents()

    # =========================
    # BUILD BM25 MODEL
    # =========================
    bm25 = build_bm25(documents)

    # =========================
    # STEP 1: BM25 KEYWORD SCORING
    # =========================
    # Scores documents based on keyword matching
    bm25_scores = bm25.get_scores(query.split())

    # =========================
    # NORMALIZE BM25 SCORES
    # =========================
    # Converts scores to 0–1 range for fair hybrid weighting
    if bm25_scores.max() == bm25_scores.min():
        bm25_scores = np.zeros_like(bm25_scores)
    else:
        bm25_scores = (
            bm25_scores - bm25_scores.min()
        ) / (
            bm25_scores.max() - bm25_scores.min()
        )

    # =========================
    # STEP 2: DENSE VECTOR SEARCH
    # =========================
    # Convert query into embedding vector
    query_vector = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # Query Qdrant vector database
    dense_results = client.query_points(
        collection_name="company_docs",
        query=query_vector,
        limit=50
    )

    # Store dense scores mapped by document text
    dense_scores = {
        p.payload["text"]: p.score
        for p in dense_results.points
    }

    # =========================
    # STEP 3: HYBRID SCORING
    # =========================
    final_scores = []

    for idx, (doc_text, source) in enumerate(documents):

        # Combine BM25 + Dense scores
        hybrid_score = (
            0.3 * bm25_scores[idx] +                 # keyword relevance
            0.7 * dense_scores.get(doc_text, 0)     # semantic relevance
        )

        # Store final scored result
        final_scores.append(
            (doc_text, source, hybrid_score)
        )

    # =========================
    # STEP 4: SORT RESULTS
    # =========================
    # Highest relevance first
    final_scores.sort(
        key=lambda x: x[2],
        reverse=True
    )

    # =========================
    # RETURN TOP-K RESULTS
    # =========================
    return final_scores[:top_k]