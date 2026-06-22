from sentence_transformers import CrossEncoder


# =========================
# LOAD RERANKER MODEL
# =========================
# CrossEncoder is used for advanced semantic scoring
# It compares (query, document) together instead of separate embeddings
reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)


# =========================
# RERANK FUNCTION
# =========================
def rerank(query, docs):

    # =========================
    # SAFETY CHECK: EMPTY INPUT
    # =========================
    # If no documents are provided, return empty result immediately
    if not docs:
        return []

    # =========================
    # STEP 1: BUILD QUERY-DOCUMENT PAIRS
    # =========================
    # Convert retrieved docs into format required by CrossEncoder:
    # (query, document_text)
    pairs = [
        (query, doc_text)
        for doc_text, source, score in docs
    ]

    # =========================
    # STEP 2: PREDICT RELEVANCE SCORES
    # =========================
    # The model outputs a relevance score for each pair
    # Higher score = more relevant to the query
    scores = reranker.predict(pairs)

    # =========================
    # STEP 3: COMBINE DOCS WITH SCORES
    # =========================
    # Zip original docs with their reranker scores
    combined = zip(docs, scores)

    # =========================
    # STEP 4: SORT BY RELEVANCE SCORE
    # =========================
    # Sort documents based on reranker score (descending order)
    ranked = sorted(
        combined,
        key=lambda x: x[1],
        reverse=True
    )

    # =========================
    # STEP 5: RETURN RERANKED RESULTS
    # =========================
    # Output format:
    # [((text, source, bm25_score), rerank_score), ...]
    return ranked