#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    serarch_command
    )

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify")

    embed_text_parser = subparsers.add_parser("embed_text")
    embed_text_parser.add_argument("text", type= str, help="Text to embed")

    embed_query_parser = subparsers.add_parser("embedquery")
    embed_query_parser.add_argument("query", type= str, help="Query to embed")

    subparsers.add_parser("verify_embeddings")

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query", type= str, help="Search term")
    search_parser.add_argument("--limit", type= int, default= 5, help="Maxiumum results returned from the search")

    args = parser.parse_args()

    match args.command:
        case "search":
            serarch_command(args.query, args.limit)
        case "embedquery":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "verify":
            verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()