
import csv
import json

from pathlib import Path

from eval.metrics import (
    hit_rate_at_k,
    reciprocal_rank,
    answer_correctness,
)

from eval.llm_judge import judge_faithfulness
from eval.semantic_judge import judge_semantic_correctness

BASE_DIR = Path(__file__).resolve().parent
TESTSET_PATH = BASE_DIR / "testset.json"
RESULTS_DIR = BASE_DIR / "results"


def fake_ask(question):
    """
    Temporary question-answering function for testing.
    """

    if "attendance" in question.lower():
        return {
            "answer": "75%",
            "sources": [
                {
                    "chunk_id": 1,
                    "doc_name": "sample.pdf",
                    "page": 12,
                    "text": "Minimum attendance is 75%",
                }
            ],
            "found": True,
        }

    return {
        "answer": "Not found in the document",
        "sources": [],
        "found": False,
    }


def run_evaluation(
    ask_fn=fake_ask,
    judge_fn=None,
    semantic_judge_fn=None,
    evaluate_semantic=False,
):
    """
    Evaluate a question-answering pipeline.

    ask_fn:
        Function accepting a question and returning
        answer, sources, and found.

    judge_fn:
        Function evaluating whether an answer is
        supported by retrieved context.
    """

    if judge_fn is None:
        judge_fn = judge_faithfulness

    with open(
        TESTSET_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        test_questions = json.load(file)

    results = []

    for item in test_questions:

        question = item["question"]
        expected_answer = item["expected_answer"]
        expected_pages = item["expected_pages"]
        answerable = item["answerable"]

        # Get the generated answer and retrieved sources
        response = ask_fn(question)

        generated_answer = response["answer"]
        sources = response["sources"]

        # Extract retrieved page numbers
        retrieved_pages = [
            source["page"]
            for source in sources
        ]

        # Combine retrieved PDF text into one context
        context = "\n\n".join(
            source.get("text", "")
            for source in sources
            if source.get("text")
        )

        # 1. Retrieval metrics
        if answerable:
            hit_rate = hit_rate_at_k(
                retrieved_pages,
                expected_pages,
                k=5,
            )

            mrr = reciprocal_rank(
                retrieved_pages,
                expected_pages,
                k=5,
            )

        else:
            hit_rate = None
            mrr = None

        # 2. Answer correctness
        correctness = answer_correctness(
            generated_answer,
            expected_answer,
        )

        # 3. Refusal correctness
        refusal_correct = (
            not response["found"]
            if not answerable
            else None
        )

        # 4. Ollama faithfulness evaluation
        faithfulness = None
        faithfulness_reason = "Not evaluated"

        if response["found"] and context.strip():

            judgment = judge_fn(
                context,
                generated_answer,
            )

            faithfulness = judgment["supported"]
            faithfulness_reason = judgment["reason"]

        # 5. Semantic answer correctness
        semantic_correctness = None
        semantic_reason = "Not evaluated"

        if evaluate_semantic and answerable:
            semantic_judge = (
                semantic_judge_fn
                if semantic_judge_fn is not None
                else judge_semantic_correctness
            )

            semantic_result = semantic_judge(
                question,
                expected_answer,
                generated_answer,
            )

            semantic_correctness = semantic_result["correct"]
            semantic_reason = semantic_result["reason"]

        # Save evaluation result for this question
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
            "faithfulness": faithfulness,
            "faithfulness_reason": faithfulness_reason,
            "semantic_answer_correctness": semantic_correctness,
            "semantic_correctness_reason": semantic_reason,
        })

    # Create results directory if necessary
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = RESULTS_DIR / "results.csv"

    # Write results to CSV
    if results:

        with open(
            output_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=results[0].keys(),
            )

            writer.writeheader()
            writer.writerows(results)

    print(f"Evaluated {len(results)} questions.")
    print(f"Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    from docucheck.pipeline import ask

    run_evaluation(
        ask_fn=ask,
        evaluate_semantic=True,
    )
