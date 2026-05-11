import argparse
from lib.hybrid_search import normalize_scores, weighted_search_command, rrf_search_command
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument("scores", nargs='*',type=float, help="List of scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search")
    weighted_search_parser.add_argument("query", type=str, help="Query to search")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Weight of semantic search vs keyworrd search")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Search results limit")

    rrf_search_parser = subparsers.add_parser("rrf-search")
    rrf_search_parser.add_argument("query", type=str, help="Query to search")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="The k weighting parameter")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Search limit number")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method",type=str, choices=["individual", "batch", "cross_encoder"], help="Option to rerank the search results")

    args = parser.parse_args()

    match args.command:
        case "rrf-search":
            results = rrf_search_command(args.query, args.enhance, args.rerank_method, args.k, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")

                if args.rerank_method and args.rerank_method == "individual":
                    print(f"   Re-rank Score: {res.get('rerank_score', 0):.3f}")
                elif args.rerank_method and args.rerank_method == "batch":
                    print(f"   Re-rank Rank: {res.get('rerank_score', 0):.3f}")

                print(f"   RRF Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "keyword_rank" in metadata and "semantic_rank" in metadata:
                    print(
                        f"   BM25: {metadata['keyword_rank']}, Semantic: {metadata['semantic_rank']}"
                    )
                print(f"   {res['document'][:100]}...")
        case "weighted-search":
            results = weighted_search_command(args.query, args.alpha, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")
        case "normalize":
            scores = normalize_scores(args.scores)
            for score in scores:
                print(f"* {score:.4f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
