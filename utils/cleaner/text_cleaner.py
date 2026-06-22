import re  # Import regex module for text pattern cleaning

def clean_text(text):
    """
    Cleans raw text from PDFs / Wikipedia / scraped documents.
    Removes noise like page numbers, citations, and extra spaces.
    """

    # =========================
    # REMOVE PDF ARTIFACTS
    # =========================

    # Remove patterns like "Page 1", "Page 2", etc.
    text = re.sub(
        r"Page\s+\d+",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove "Confidential" labels from documents
    text = re.sub(
        r"Confidential",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove "Internal Use Only" labels from documents
    text = re.sub(
        r"Internal Use Only",
        "",
        text,
        flags=re.IGNORECASE
    )

    # =========================
    # REMOVE WIKIPEDIA REFERENCES
    # =========================

    # Remove citation numbers like [1], [25], [149]
    text = re.sub(
        r"\[\d+\]",
        "",
        text
    )

    # Remove note-style citations like [note 18]
    text = re.sub(
        r"\[note\s*\d+\]",
        "",
        text,
        flags=re.IGNORECASE
    )

    # =========================
    # NORMALIZE WHITESPACE
    # =========================

    # Replace multiple spaces/newlines with a single space
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove leading and trailing spaces
    return text.strip()