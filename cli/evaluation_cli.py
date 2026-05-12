import argparse
from lib.evaluation import evaluate_command


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    evaluation = evaluate_command(limit)

    for result in evaluation:
        print(f"k={limit}\n\n")
        print(f"-Query: {result["query"]}")
        print(f"    -Precision@{limit}: {result["precision"]:.4f}")
        print(f"    -Recall@{limit}: {result["recall"]:.4f}")
        print(f"    -F1 Score: {result["f1_score"]:.4f}")
        print(f"    -Retrieved: {result["retrieved"]}")
        print(f"    -Relevant: {result["relevant"]}")

if __name__ == "__main__":
    main()