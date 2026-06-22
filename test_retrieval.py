from retriever import hybrid_search


def main():

    # =========================
    # USER INPUT
    # =========================
    query = input("Enter your search query: ").strip()

    if not query:
        print("Empty query. Please enter a valid search term.")
        return

    # =========================
    # RUN RETRIEVAL
    # =========================
    try:
        results = hybrid_search(query, top_k=5)

    except Exception as e:
        print("\nERROR during retrieval:", str(e))
        return

    # =========================
    # PRINT RESULTS
    # =========================
    print("\n===== RETRIEVAL RESULTS =====\n")

    if not results:
        print("No results found.")
        return

    for i, (text, source, score) in enumerate(results, 1):

        print(f"Rank: {i}")
        print(f"Source: {source}")
        print(f"Score: {score:.4f}")
        print(f"Text: {text[:250]}")
        print("-" * 60)


if __name__ == "__main__":
    main()