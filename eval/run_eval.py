
import csv
import json
from pathlib import Path

from eval.metrics import (
    hit_rate_at_k,
    reciprocal_rank,
    answer_correctness,
)

BASE_DIR = Path(__file__).resolve().parent
TESTSET_PATH = BASE_DIR / "testset.json"
RESULTS_DIR = BASE_DIR / "results"


def fake_ask(question):
    """
    Temporary function until the real RAG
    pipeline is ready.
    """
    if "attendance" in question.lower():
        return {
            "answer": "75%",
            "sources": [
                {
                    "chunk_id": 1,
                    "doc_name": "sample.pdf",
                    "page": 12,
                    "text": "Minimum attendance is 75%."
                }
            ],
            "found": True
        }

    return {
        "answer": "Not found in the document",
        "sources": [],
        "found": False
    }


def run_evaluation(ask_fn=fake_ask):
    """
    Run evaluation on all test questions.
    """

    with open(TESTSET_PATH, "r", encoding="utf-8") as file:
        test_questions = json.load(file)

    results = []

    for item in test_questions:
        question = item["question"]
        expected_answer = item["expected_answer"]
        expected_pages = item["expected_pages"]
        answerable = item["answerable"]

        response = ask_fn(question)

        generated_answer = response["answer"]

        retrieved_pages = [
            source["page"]
            for source in response["sources"]
        ]

        if answerable:
            hit_rate = hit_rate_at_k(
                retrieved_pages, expected_pages, k=5
            )

            mrr = reciprocal_rank(
                retrieved_pages, expected_pages, k=5
            )
        else:
            hit_rate = None
            mrr = None

        correctness = answer_correctness(
            generated_answer, expected_answer
        )

        refusal_correct = (
            not response["found"]
            if not answerable
            else None
        )

        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "expected_pages": str(expected_pages),
            "retrieved_pages": str(retrieved_pages),
            "answerable": answerable,
            "hit_rate_at_5": hit_rate,
            "mrr_at_5": mrr,
            "answer_correctness": correctness,
            "correct_refusal": refusal_correct,
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "results.csv"

    if results:
        with open(
            output_path, "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=results[0].keys()
            )
            writer.writeheader()
            writer.writerows(results)

    print(f"Evaluated {len(results)} questions.")
    print(f"Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_evaluation()
