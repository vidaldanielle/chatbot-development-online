from utils.loader.pdf_loader import (
    load_documents
)

from utils.cleaner.text_cleaner import (
    clean_text
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from sentence_transformers import (
    SentenceTransformer
)

from qdrant_client import (
    QdrantClient
)

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

from datasketch import MinHash

import uuid


# =========================
# MINHASH FUNCTION
# =========================

def get_minhash(text):

    m = MinHash(
        num_perm=128
    )

    for word in text.split():

        m.update(
            word.encode("utf8")
        )

    return m


# =========================
# LOAD DOCUMENTS
# =========================

documents = load_documents()

print(
    f"Documents loaded: {len(documents)}"
)

print(
    documents[0].text[:3000]
)


# =========================
# TEXT SPLITTER
# =========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=150
)


# =========================
# CHUNKING + SOURCE TRACKING
# =========================

chunks_with_source = []

for doc in documents:

    source = doc.metadata.get(
        "file_name",
        "Unknown"
    )

    chunks = splitter.split_text(
        clean_text(
            doc.text
        )
    )

    for chunk in chunks:

        chunks_with_source.append(
            (
                chunk,
                source
            )
        )


# =========================
# DUPLICATE REMOVAL
# =========================

unique_texts = []
seen = []

for text, source in chunks_with_source:

    mh = get_minhash(
        text
    )

    duplicate = False

    for old in seen:

        if mh.jaccard(old) > 0.90:

            duplicate = True
            break

    if not duplicate:

        seen.append(
            mh
        )

        unique_texts.append(
            (
                text,
                source
            )
        )

texts = unique_texts

print(
    f"Chunks created: {len(texts)}"
)


# =========================
# EMBEDDING MODEL
# =========================

embedding_model = SentenceTransformer(
    "BAAI/bge-m3"
)

vectors = embedding_model.encode(
    [
        text
        for text, source in texts
    ],
    normalize_embeddings=True,
    show_progress_bar=True
)


# =========================
# QDRANT CONNECTION
# =========================

client = QdrantClient(
    path="./qdrant_data"
)


# =========================
# DELETE OLD COLLECTION
# =========================

try:

    client.delete_collection(
        collection_name="company_docs"
    )

except:

    pass


# =========================
# CREATE COLLECTION
# =========================

client.create_collection(
    collection_name="company_docs",
    vectors_config=VectorParams(
        size=1024,
        distance=Distance.COSINE
    )
)


# =========================
# BUILD POINTS
# =========================

points = []

for (text, source), vector in zip(
    texts,
    vectors
):

    points.append(

        PointStruct(

            id=str(
                uuid.uuid4()
            ),

            vector=vector.tolist(),

            payload={
                "text": text,
                "source": source
            }

        )

    )


# =========================
# UPLOAD TO QDRANT
# =========================

client.upsert(
    collection_name="company_docs",
    points=points
)


# =========================
# DONE
# =========================

print(
    f"Successfully indexed {len(points)} chunks."
)