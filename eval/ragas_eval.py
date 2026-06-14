# eval/ragas_eval.py
"""
Faithfulness (0-1): Does the answer only contain claims supported by retrieved context?
  1.0 = every claim grounded in documents
  0.5 = mix of grounded and model knowledge
  0.0 = contradicts or ignores retrieved context

Answer Relevance (0-1): Does the answer directly address the question?
  1.0 = fully addresses the question
  0.5 = partially addresses it
  0.0 = doesn't address it at all
"""

import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.graph import run_agent

TEST_QUERIES = [
    {
        "id": 1,
        "query": "What was Apple's total revenue in 2025?",
        "expected_type": "numerical",
        "notes": "Should use execute_sql for exact figure from Apple's 2025 filing"
    },
    {
        "id": 2,
        "query": "What are the main risk factors Amazon mentions in their 2025 filing?",
        "expected_type": "qualitative",
        "notes": "Should use search_vector_db with Amazon + 2025 filter"
    },
    {
        "id": 3,
        "query": "Compare Apple and Google's net income in 2025. Which was more profitable?",
        "expected_type": "comparison",
        "notes": "Should use both tools: SQL for numbers, vector DB for context"
    },
    {
        "id": 4,
        "query": "How does Amazon describe its cloud computing strategy in 2025?",
        "expected_type": "qualitative",
        "notes": "Should use search_vector_db with Amazon + 2025 filter, focus on AWS"
    },
    {
        "id": 5,
        "query": "Which of the 4 companies had the highest R&D spending as a percentage of revenue in 2025?",
        "expected_type": "analytical",
        "notes": "Should use SQL, compute the R&D/revenue ratio across all companies with 2025 financial data"
    },

]


def run_evaluation():
    print("=" * 70)
    print("RAGAS EVALUATION — 7 Test Queries")
    print("=" * 70)

    results = []

    for test in TEST_QUERIES:
        print(f"\n{'─' * 70}")
        print(f"Query {test['id']}: {test['query']}")
        print(f"Type: {test['expected_type']}")

        result = run_agent(test["query"])

        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nCost: ${result['cost']['total_cost_usd']:.6f} | "
              f"Loops: {result['cost']['total_loops']} | "
              f"Tier: {result['cost']['tier']}")

        eval_result = {
            "query_id": test["id"],
            "query": test["query"],
            "answer": result["answer"],
            "cost": result["cost"],
            # Fill in these scores after reading each answer:
            "faithfulness": None,        # 0.0 to 1.0
            "answer_relevance": None,    # 0.0 to 1.0
            "notes": "",
        }
        results.append(eval_result)

    os.makedirs("eval", exist_ok=True)
    with open("eval/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Results saved to eval/results.json")
    print("Open the file and fill in faithfulness + answer_relevance scores.")
    # print("Then add the table to REPORT.md")


if __name__ == "__main__":
    run_evaluation()