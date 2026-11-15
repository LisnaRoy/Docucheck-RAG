
import re
def hit_rate_at_k(retrieved_pages, expected_pages, k=5):
    """
    Check whether at least one expected page
    appears in the top k retrieved results.
    """
    top_pages = retrieved_pages[:k]

    for page in top_pages:
        if page in expected_pages:
            return 1

    return 0


def reciprocal_rank(retrieved_pages, expected_pages, k=5):
    """
    Calculate reciprocal rank of the first
    relevant retrieved page.
    """
    top_pages = retrieved_pages[:k]

    for rank, page in enumerate(top_pages, start=1):
        if page in expected_pages:
            return 1 / rank

    return 0.0


def answer_correctness(generated_answer, expected_answer):
    """
    Compare answers after basic text normalization.

    Returns:
        1 if normalized answers match
        0 otherwise
    """

    def normalize(text):
        text = text.casefold().strip()

        # Remove common answer formatting
        text = re.sub(r"^\s*[-•]\s*", "", text)

        # Remove simple page citations
        text = re.sub(
            r"\(\s*page\s+[\d.]+\s*\)",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Standardize percentage wording
        text = re.sub(
            r"\bpercent\b",
            "%",
            text,
        )

        # Standardize basic time-period wording
        text = re.sub(
            r"\bweeks\b",
            "week",
            text,
        )

        # Remove punctuation except decimal points and %
        text = re.sub(r"[^\w\s.%]", " ", text)

        # Remove final punctuation
        text = text.strip(" .")

        # Normalize repeated whitespace
        text = " ".join(text.split())

        return text

    generated = normalize(generated_answer)
    expected = normalize(expected_answer)

    return int(generated == expected)
