#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_command,
    chunk_command,
    semantic_chunk_parser_command,

    )

from lib.chunked_semantic_search import embed_chunks_command, search_chunked_command

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

    chunk_parser = subparsers.add_parser("chunk")
    chunk_parser.add_argument("text", type= str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type= int, default=200, help="Size of text chunk")
    chunk_parser.add_argument("--overlap", type= int,help="Chunk text overlap")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk")
    semantic_chunk_parser.add_argument("text", type= str, help="String to turn into chunks")
    semantic_chunk_parser.add_argument("--max-chunk-size", type= int, default=4, help="Maximum size of each chunk")
    semantic_chunk_parser.add_argument("--overlap", type= int, default= 0, help="Overlap of each chunk")

    subparsers.add_parser("embed_chunks")

    semantic_search_parser = subparsers.add_parser("search_chunked")
    semantic_search_parser.add_argument("query", type= str, help= "Search query")
    semantic_search_parser.add_argument("--limit", type= int, default=5,help= "Maximum number of movies in result")

    args = parser.parse_args()

    match args.command:
        case "search_chunked":
            result = search_chunked_command(args.query, args.limit)
            print(f"Query: {result['query']}")
            print("Results:")
            for i, res in enumerate(result["results"], 1):
                print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['document']}...")
        case "embed_chunks":
            embeddings = embed_chunks_command()
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "semantic_chunk":
            semantic_chunk_parser_command(args.text, args.max_chunk_size, args.overlap)
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "search":
            search_command(args.query, args.limit)
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