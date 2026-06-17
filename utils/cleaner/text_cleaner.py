import re


def clean_text(text):

    text = re.sub(
        r"Page\s+\d+",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"Confidential",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"Internal Use Only",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()