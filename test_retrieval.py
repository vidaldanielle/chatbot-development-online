from retriever import hybrid_search   # Imports retrieval system (BM25 + Dense Search via Qdrant)
from reranker import rerank           # Imports reranker (Cross-Encoder for better ranking)

# =========================
# MAIN PROGRAM FUNCTION
# =========================
def main():

    # =========================
    # INFINITE LOOP (CHAT STYLE INTERFACE)
    # =========================
    # Keeps program running until user types 'exit'
    while True:

        # =========================
        # USER INPUT
        # =========================
        # Accepts user query from terminal
        query = input("\nEnter your search query (type 'exit' to stop): ").strip()

        # =========================
        # EXIT CONDITION
        # =========================
        # Stops the loop and ends program when user types 'exit'
        if query.lower() == "exit":
            print("Exiting program...")
            break

        # =========================
        # INPUT VALIDATION
        # =========================
        # Prevents empty queries from being processed
        if not query:
            print("Empty query. Try again.")
            continue

        try:
            # =========================
            # STEP 1: RETRIEVAL PHASE
            # =========================
            # Calls hybrid_search from retriever.py
            # Combines:
            # - BM25 (keyword-based search)
            # - Dense embedding search (semantic similarity using Qdrant + BGE-M3)
            results = hybrid_search(query, top_k=10)

            # =========================
            # STEP 2: RERANKING PHASE
            # =========================
            # Refines retrieved documents using CrossEncoder model
            # Improves relevance scoring by understanding full sentence meaning
            reranked = rerank(query, results)

            # =========================
            # OUTPUT HEADER
            # =========================
            print("\n===== FINAL RERANKED RESULTS =====\n")

            # =========================
            # DISPLAY TOP RESULTS
            # =========================
            # reranked format:
            # ((text, source, hybrid_score), rerank_score)
            for i, ((text, source, score), rerank_score) in enumerate(reranked[:5], 1):

                # Rank number of result
                print(f"Rank: {i}")

                # Source file of the document (e.g., Wikipedia.md)
                print(f"Source: {source}")

                # Score from hybrid retrieval (BM25 + Dense search)
                print(f"Hybrid Score: {score:.4f}")

                # Score from reranker (Cross-Encoder relevance score)
                print(f"Rerank Score: {rerank_score:.4f}")

                # Preview of retrieved text chunk
                print(f"Text: {text[:250]}")

                print("-" * 60)

        except Exception as e:
            # =========================
            # ERROR HANDLING
            # =========================
            # Catches any runtime errors from retrieval or reranking pipeline
            print("ERROR:", str(e))


# =========================
# PROGRAM ENTRY POINT
# =========================
# Ensures script runs only when executed directly (not imported)
if __name__ == "__main__":
    main()