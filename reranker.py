from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)


def rerank(
    query,
    docs
):

    # docs:
    # (
    #   doc_text,
    #   source,
    #   hybrid_score
    # )

    pairs = [

        [query, doc_text]

        for doc_text, source, score in docs

    ]

    scores = reranker.predict(
        pairs
    )

    ranked = sorted(

        zip(
            docs,
            scores
        ),

        key=lambda x: x[1],

        reverse=True

    )

    return ranked