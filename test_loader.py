from llama_index.core import SimpleDirectoryReader

docs = SimpleDirectoryReader(
    input_dir="data"
).load_data()

print(type(docs[0]))
print("=" * 50)
print(docs[0].text[:1000])