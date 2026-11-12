
from eval.metrics import (
    hit_rate_at_k,
    reciprocal_rank,
    answer_correctness,
)


def test_hit_rate_success():
    retrieved = [3, 8, 12, 15, 20]
    expected = [12]

    assert hit_rate_at_k(retrieved, expected, 5) == 1


def test_hit_rate_failure():
    retrieved = [3, 8, 15, 20, 25]
    expected = [12]

    assert hit_rate_at_k(retrieved, expected, 5) == 0


def test_reciprocal_rank():
    retrieved = [3, 8, 12, 15, 20]
    expected = [12]

    assert reciprocal_rank(retrieved, expected, 5) == 1 / 3


def test_reciprocal_rank_missing():
    retrieved = [3, 8, 15, 20, 25]
    expected = [12]

    assert reciprocal_rank(retrieved, expected, 5) == 0.0


def test_answer_correctness_success():
    assert answer_correctness("75%", "75%") == 1


def test_answer_correctness_normalization():
    assert answer_correctness("  YES ", "yes") == 1


def test_answer_correctness_failure():
    assert answer_correctness("80%", "75%") == 0
