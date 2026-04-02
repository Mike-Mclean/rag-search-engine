import argparse
from query_process import *
from inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res}")
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
            test_doc = index.index["merida"]
            print(f"First document for token 'merida' = {list(test_doc)[0]}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()