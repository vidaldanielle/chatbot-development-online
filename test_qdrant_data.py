from qdrant_client import QdrantClient

client = QdrantClient(
    host="localhost",
    port=6333
)

count = client.count(
    collection_name="company_docs"
)

print(count)

points, _ = client.scroll(
    collection_name="company_docs",
    limit=10,
    with_payload=True
)

for p in points:
    print("\n=================")
    print(p.payload["text"])