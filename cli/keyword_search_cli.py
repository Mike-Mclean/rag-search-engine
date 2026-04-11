import argparse
from key_word_search import *
from search_movies import *
import math

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build")

    frequency_parser = subparsers.add_parser("tf")
    frequency_parser.add_argument("doc_id", type=int, help="Document ID")
    frequency_parser.add_argument("term", type=str, help="Search Term")

    idf_parser = subparsers.add_parser("idf")
    idf_parser.add_argument("term", type=str, help="Search term")

    tf_idf_parser = subparsers.add_parser("tfidf")
    tf_idf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_idf_parser.add_argument("term", type=str, help="Search Term")

    args = parser.parse_args()

    index = InvertedIndex()

    match args.command:
        case "tfidf":
            index.load()
            tf_idf = index.get_tf_idf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "idf":
            index.load()
            idf = index.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tf":
            index.load()
            print(index.get_tf(args.doc_id, args.term))
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res}")
        case "build":
            index.build()
            index.save()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()