
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
    Check whether the generated answer matches
    the expected answer after basic normalization.
    """
    generated = generated_answer.strip().casefold()
    expected = expected_answer.strip().casefold()

    return int(generated == expected)
