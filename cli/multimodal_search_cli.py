import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command

def main():
    parser = argparse.ArgumentParser(description="Multimodal search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verifies the embedding of the image"
    )
    verify_image_parser.add_argument("image_path", type=str, help="Path to the image")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Searches the embeddings with the provided image"
    )
    image_search_parser.add_argument("image_path", type=str, help="Path to the image")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            search_results = image_search_command(args.image_path)[:10]
            for i, res in enumerate(search_results):
                print(f"{i + 1}. {res["title"]} (similarity: {res["score"]:.3f})")
                print(f"{res["document"][:100]}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()