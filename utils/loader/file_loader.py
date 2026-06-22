from llama_index.core import SimpleDirectoryReader  # Imports LlamaIndex reader for loading documents from a folder

def load_documents():
    """
    Loads all documents from the 'data' directory.
    Supports .pdf and .md files using LlamaIndex SimpleDirectoryReader.
    """

    # =========================
    # LOAD FILES FROM DIRECTORY
    # =========================

    documents = SimpleDirectoryReader(
        input_dir="data",  # Folder where your documents are stored
        required_exts=[".md", ".pdf"]  # Only include Markdown and PDF files
    ).load_data()  # Reads and converts files into Document objects

    # =========================
    # DEBUG INFORMATION
    # =========================

    # Print total number of documents loaded
    print(f"Documents loaded: {len(documents)}")

    # Show sample documents for verification (first 3 only)
    for doc in documents[:3]:

        print("\n--- SAMPLE DOC ---")

        # Print file name from metadata
        print("Source:", doc.metadata.get("file_name"))

        # Print first 200 characters of document text for preview
        print("Preview:", doc.text[:200])

    # =========================
    # RETURN DOCUMENTS
    # =========================

    return documents