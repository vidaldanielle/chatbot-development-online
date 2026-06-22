from utils.loader.file_loader import load_documents  # loads files from data folder (md/pdf)
from utils.cleaner.text_cleaner import clean_text    # cleans noise in text (pages, citations, etc.)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # splits docs into chunks
from sentence_transformers import SentenceTransformer  # embedding model (text → vectors)
from qdrant_client import QdrantClient                  # vector database client
from qdrant_client.models import Distance, VectorParams, PointStruct  # Qdrant config + data structure
from datasketch import MinHash  # used for near-duplicate detection
import uuid  # generates unique IDs for each chunk

# =========================
# MINHASH FUNCTION
# =========================
# Purpose: detect duplicate or very similar chunks
# (used to remove redundant text before embedding)

def get_minhash(text):

    m = MinHash(num_perm=128)  # creates MinHash object (128 hash permutations)

    for word in text.split():  # split text into words
        m.update(word.encode("utf8"))  # convert each word into hash input

    return m  # returns fingerprint of the text


# =========================
# LOAD DOCUMENTS
# =========================
# Load all documents from data folder using SimpleDirectoryReader (inside loader)

documents = load_documents()

print(f"Documents loaded: {len(documents)}")  # debug: number of loaded files

# safety check
if len(documents) == 0:
    raise ValueError("No documents loaded. Check your data folder.")


# =========================
# TEXT SPLITTER
# =========================
# Splits large documents into smaller chunks for embedding
# chunk_size = 1024 tokens/characters (approx)
# overlap = 150 ensures context continuity between chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=150
)


# =========================
# CHUNKING + SOURCE TRACKING
# =========================
# Stores all chunks together with their file source

chunks_with_source = []

for doc in documents:

    # =========================
    # SAFE SOURCE EXTRACTION
    # =========================
    # tries multiple metadata keys to avoid missing values

    source = (
        doc.metadata.get("file_name")
        or doc.metadata.get("file_path")
        or "Unknown"
    )

    # clean raw document text (removes noise, citations, whitespace)
    text = clean_text(doc.text)

    # split cleaned text into chunks
    chunks = splitter.split_text(text)

    print(f"Processing {source} → {len(chunks)} chunks")

    # store each chunk with its source file
    for chunk in chunks:
        chunks_with_source.append((chunk, source))


print(f"Total raw chunks: {len(chunks_with_source)}")


# =========================
# DUPLICATE REMOVAL (MinHash)
# =========================
# removes near-duplicate chunks using Jaccard similarity

unique_texts = []
seen = []

for text, source in chunks_with_source:

    mh = get_minhash(text)  # generate fingerprint

    duplicate = False

    # compare with previously seen chunks
    for old in seen:
        if mh.jaccard(old) > 0.90:  # threshold: 90% similarity
            duplicate = True
            break

    # if not duplicate, keep it
    if not duplicate:
        seen.append(mh)
        unique_texts.append((text, source))


print(f"Unique chunks after dedup: {len(unique_texts)}")


# =========================
# EMBEDDING MODEL
# =========================
# Converts text chunks into vector embeddings

embedding_model = SentenceTransformer("BAAI/bge-m3")

vectors = embedding_model.encode(
    [text for text, source in unique_texts],  # only text
    normalize_embeddings=True,  # important for cosine similarity
    show_progress_bar=True      # shows encoding progress
)


# =========================
# QDRANT SETUP
# =========================
# Initializes local vector database

client = QdrantClient(path="./qdrant_data")

# delete old collection if exists (fresh rebuild)
try:
    client.delete_collection("company_docs")
except:
    pass


# create new collection for embeddings
client.create_collection(
    collection_name="company_docs",
    vectors_config=VectorParams(
        size=1024,                 # vector size (must match embedding model output)
        distance=Distance.COSINE   # similarity metric
    )
)


# =========================
# BUILD POINTS
# =========================
# Convert each chunk into Qdrant Point (vector + metadata)

points = []

for (text, source), vector in zip(unique_texts, vectors):

    points.append(
        PointStruct(
            id=str(uuid.uuid4()),  # unique ID per chunk
            vector=vector.tolist(), # embedding vector
            payload={
                "text": text,       # actual chunk text
                "source": source,   # file origin
                "topic": "philippine_history"  # optional metadata tag
            }
        )
    )


# =========================
# UPLOAD TO QDRANT
# =========================
# store all vectors into vector database

client.upsert(
    collection_name="company_docs",
    points=points
)


# =========================
# DONE
# =========================
print(f"Successfully indexed {len(points)} chunks.")