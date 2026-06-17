from llama_index.core import SimpleDirectoryReader


def load_documents():

    documents = SimpleDirectoryReader(
        input_dir="data"
    ).load_data()

    #debug purpose only
    print(
        f"Documents loaded: {len(documents)}"
    )

    return documents