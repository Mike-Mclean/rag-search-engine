import argparse
from lib.augmented_generation import rag_command, summarize_command, citations_command, question_command

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize the search results")
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument("--limit", type=int, default = 5, help="Total number of results to be displayed")

    citations_parser = subparsers.add_parser("citations", help="Summarize the search results with citations")
    citations_parser.add_argument("query", type=str, help="Search query for RAG")
    citations_parser.add_argument("--limit", type=int, default = 5, help="Total number of results to be displayed")

    question_parser = subparsers.add_parser("question", help="Answer a question from the user")
    question_parser.add_argument("question", type=str, help="Question for the LLM")
    question_parser.add_argument("--limit", type=int, default = 5, help="Total number of results to be displayed")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag_command(query)
        case "summarize":
            query = args.query
            limit = args.limit
            summarize_command(query, limit)
        case "citations":
            query = args.query
            limit = args.limit
            citations_command(query, limit)
        case "question":
            question = args.question
            limit = args.limit
            question_command(question, limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()