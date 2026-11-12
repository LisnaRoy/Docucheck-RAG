
import json

from eval.run_eval import (
    run_evaluation, fake_ask,
    TESTSET_PATH,
)

def test_fake_ask_answerable():
    result = fake_ask(
        "What is the minimum attendance requirement?"
    )

    assert result["answer"] == "75%"
    assert result["found"] is True
    assert result["sources"][0]["page"] == 12


def test_fake_ask_unanswerable():
    result = fake_ask(
        "What is the university's annual profit?"
    )

    assert result["found"] is False
    assert result["sources"] == []


def test_evaluation_runner():
    with open(TESTSET_PATH, encoding="utf-8") as file:
        test_questions = json.load(file)

    results = run_evaluation()

    assert len(results) == len(test_questions)

    for result in results:
        assert "question" in result
        assert "generated_answer" in result
        assert "answer_correctness" in result
        assert "hit_rate_at_5" in result
        assert "mrr_at_5" in result
        assert "correct_refusal" in result
