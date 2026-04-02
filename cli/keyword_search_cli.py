import argparse
import json
import os
from query_process import *

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_PATH, "data", "movies.json")
SEARCH_LIMIT = 5

def load_movies(path):
    with open(path, "r") as json_file:
        data = json.load(json_file)
    return data["movies"]

def compare_tokens(query_tokens, movie_tokens):
    for query_token in query_tokens:
        for movie_token in movie_tokens:
            if query_token in movie_token:
                return True
    return False

def search_command(query, limit = SEARCH_LIMIT):
    movies = load_movies(DATA_PATH)
    results = []
    preprocessed_query = preprocess_text(query)
    for movie in movies:
        preprocessed_title = preprocess_text(movie["title"])
        if (compare_tokens(preprocessed_query, preprocessed_title)):
            results.append(movie['title'])
        if len(results) >= limit:
            break
    return results

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()