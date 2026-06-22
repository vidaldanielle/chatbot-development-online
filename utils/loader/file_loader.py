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
    # RETURN DOCUMENTS
    # =========================

    return documents