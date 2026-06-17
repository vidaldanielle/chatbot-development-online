from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import numpy as np

embedding_model = SentenceTransformer(
    "BAAI/bge-m3"
)

client = QdrantClient(
    host="localhost",
    port=6333
)

collection, _ = client.scroll(
    collection_name="company_docs",
    limit=10000,
    with_payload=True
)

documents = [
    (
        item.payload["text"],
        item.payload.get(
            "source",
            "Unknown"
        )
    )
    for item in collection
]

tokenized_docs = [
    doc[0].split()
    for doc in documents
]

bm25 = BM25Okapi(
    tokenized_docs
)


def hybrid_search(
    query,
    top_k=10
):

    bm25_scores = bm25.get_scores(
        query.split()
    )

    query_vector = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    dense_results = client.query_points(
        collection_name="company_docs",
        query=query_vector,
        limit=100
    )

    dense_scores = {}

    for point in dense_results.points:

        dense_scores[
            point.payload["text"]
        ] = point.score

    bm25_scores = np.array(
        bm25_scores
    )

    bm25_scores = (
        bm25_scores - bm25_scores.min()
    ) / (
        bm25_scores.max()
        - bm25_scores.min()
        + 1e-8
    )

    final_scores = []

    for idx, (doc_text, source) in enumerate(
            documents
    ):

        hybrid_score = (
            0.4 * bm25_scores[idx]
            +
            0.6 * dense_scores.get(
                doc_text,
                0
            )
        )

        final_scores.append(
            (
                doc_text,
                source,
                hybrid_score
            )
        )

    final_scores.sort(
        key=lambda x: x[2],
        reverse=True
    )

    return final_scores[:top_k]