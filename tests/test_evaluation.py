
from eval.run_eval import run_evaluation


def fake_ask(question):
    """Return predictable answers without running the real RAG pipeline."""
    if "attendance" in question.lower():
        return {
            "answer": "75%",
            "sources": [
                {
                    "chunk_id": 1,
                    "doc_name": "sample.pdf",
                    "page": 12,
                    "text": "Minimum attendance is 75%.",
                }
            ],
            "found": True,
        }

    return {
        "answer": "Not found in the document",
        "sources": [],
        "found": False,
    }


def fake_judge(context, answer):
    """Avoid calling the real Ollama server in automated tests."""
    return {
        "supported": True,
        "reason": "Mock judgment for automated testing",
    }


def test_evaluation_runner(tmp_path, monkeypatch):
    """Check evaluation results without using Ollama or a real PDF."""

    # Use a small test dataset, independent of the real evaluation dataset.
    testset = [
        {
            "question": "What is the attendance requirement?",
            "expected_answer": "75%",
            "expected_pages": [12],
            "answerable": True,
        },
        {
            "question": "What is the university's annual profit?",
            "expected_answer": "Not found in the document",
            "expected_pages": [],
            "answerable": False,
        },
    ]

    import json
    import eval.run_eval as runner

    testset_path = tmp_path / "testset.json"
    testset_path.write_text(
        json.dumps(testset),
        encoding="utf-8",
    )

    # Keep test outputs out of the real eval/results directory.
    monkeypatch.setattr(runner, "TESTSET_PATH", testset_path)
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")

    results = run_evaluation(
        ask_fn=fake_ask,
        judge_fn=fake_judge,
    )

    assert len(results) == 2

    # Answerable question
    first = results[0]
    assert first["hit_rate_at_5"] == 1
    assert first["mrr_at_5"] == 1.0
    assert first["answer_correctness"] == 1
    assert first["faithfulness"] is True
    assert first["faithfulness_reason"] == (
        "Mock judgment for automated testing"
    )

    # Unanswerable question
    second = results[1]
    assert second["correct_refusal"] is True
    assert second["faithfulness"] is None

    # Check that CSV output was created.
    output_file = tmp_path / "results" / "results.csv"
    assert output_file.exists()
